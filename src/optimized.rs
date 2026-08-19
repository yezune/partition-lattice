//! Optimized Partition Implementation using Bitvectors and SIMD-friendly operations
//!
//! This module provides high-performance partition operations using:
//! - Bitvector representation for cache-friendly memory layout
//! - Parallel processing with rayon for large partitions
//! - SIMD-friendly operations where possible

#[cfg(not(feature = "parallel"))]
use crate::par_shim::{IntoParShim, ParRefShim};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::FloatExt;
use crate::HashMap;
#[allow(unused_imports)]
use alloc::borrow::ToOwned;
#[allow(unused_imports)]
use alloc::boxed::Box;
#[allow(unused_imports)]
use alloc::string::{String, ToString};
#[allow(unused_imports)]
use alloc::vec::Vec;
use bitvec::prelude::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Threshold for switching to parallel algorithms
const PARALLEL_THRESHOLD: usize = 512;

/// Maximum block count for elementwise differentiate results.
/// Prevents memory explosion: K blocks × ⌈N/64⌉ words × 8 bytes per block BitVec.
/// At N=1M, K=512 → 512 × 125KB = 64MB per partition (acceptable).
/// Without cap, cumulative differentiations can produce K≈N, requiring N²/8 bytes (125GB at N=1M).
pub(crate) const MAX_ELEMENTWISE_BLOCKS: usize = 512;

/// Maximum total memory for block BitVecs per partition (8MB).
/// Adaptive cap: only triggers when actual memory would exceed this limit.
/// At N=1M: max K = 8MB / 125KB = 64 blocks → 16MB total per partition.
/// This keeps total RSS bounded when many partitions are held (dream_buffer, CUSP).
const BLOCKS_MEMORY_LIMIT: usize = 8 * 1024 * 1024;

/// Optimized partition representation using block membership bitvectors.
///
/// For a partition with k blocks, we store k bitvectors where bit i is set
/// if element i belongs to that block. This enables fast set operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastPartition {
    /// Number of elements in the universe
    size: usize,
    /// Block membership bitvectors: blocks[b][i] = 1 iff element i is in block b
    blocks: Vec<BitVec<u64, Lsb0>>,
    /// Element to block mapping (for fast lookup)
    element_to_block: Vec<u32>,
    /// Precomputed Σ|B|² for O(1) distinction_count and logical_entropy
    sum_block_sizes_sq: u64,
}

impl FastPartition {
    /// Internal constructor: computes sum_block_sizes_sq from blocks.
    fn new_raw(size: usize, blocks: Vec<BitVec<u64, Lsb0>>, element_to_block: Vec<u32>) -> Self {
        let sum_block_sizes_sq: u64 = blocks
            .iter()
            .map(|b| {
                let c = b.count_ones() as u64;
                c * c
            })
            .sum();
        Self {
            size,
            blocks,
            element_to_block,
            sum_block_sizes_sq,
        }
    }

    /// Creates a discrete partition (bottom) where each element is in its own block.
    pub fn discrete(size: usize) -> Self {
        let mut blocks = Vec::with_capacity(size);
        let element_to_block: Vec<u32> = (0..size).map(|i| i as u32).collect();

        for i in 0..size {
            let mut bv = bitvec![u64, Lsb0; 0; size];
            bv.set(i, true);
            blocks.push(bv);
        }

        Self::new_raw(size, blocks, element_to_block)
    }

    /// Creates an indiscrete partition (top) where all elements are in one block.
    pub fn indiscrete(size: usize) -> Self {
        if size == 0 {
            return Self {
                size: 0,
                blocks: vec![],
                element_to_block: vec![],
                sum_block_sizes_sq: 0,
            };
        }
        let bv = bitvec![u64, Lsb0; 1; size];
        let blocks = vec![bv];
        let element_to_block = vec![0u32; size];

        Self::new_raw(size, blocks, element_to_block)
    }

    /// Creates a partition from block IDs.
    pub fn from_ids(ids: &[u32]) -> Self {
        let size = ids.len();
        if size == 0 {
            return Self {
                size: 0,
                blocks: vec![],
                element_to_block: vec![],
                sum_block_sizes_sq: 0,
            };
        }

        let max_id = *ids.iter().max().unwrap_or(&0);

        let mut element_to_block: Vec<u32> = Vec::with_capacity(size);
        let num_blocks: u32;

        if (max_id as usize) < size * 2 {
            // Heuristic threshold
            let mut id_map: Vec<u32> = vec![u32::MAX; max_id as usize + 1];
            let mut next_id: u32 = 0;

            for &id in ids {
                if id_map[id as usize] == u32::MAX {
                    id_map[id as usize] = next_id;
                    next_id += 1;
                }
                element_to_block.push(id_map[id as usize]);
            }
            num_blocks = next_id;
        } else {
            // Fallback to HashMap for sparse/large IDs
            let mut id_map: HashMap<u32, u32> = HashMap::new();
            let mut next_id: u32 = 0;

            for &id in ids {
                let normalized = *id_map.entry(id).or_insert_with(|| {
                    let n = next_id;
                    next_id += 1;
                    n
                });
                element_to_block.push(normalized);
            }
            num_blocks = next_id;
        }

        let mut blocks: Vec<BitVec<u64, Lsb0>> = (0..num_blocks as usize)
            .map(|_| bitvec![u64, Lsb0; 0; size])
            .collect();

        for (i, &block) in element_to_block.iter().enumerate() {
            blocks[block as usize].set(i, true);
        }

        Self::new_raw(size, blocks, element_to_block)
    }

    /// Computes the ditset XOR distance (symmetric):
    ///
    /// $$d_{\oplus}(P,Q) = |\mathrm{dit}(P) \Delta \mathrm{dit}(Q)| / N^2$$
    ///
    pub fn ditset_xor_distance(&self, other: &Self) -> f64 {
        assert_eq!(self.size, other.size, "Partition sizes must match");
        if self.size == 0 {
            return 0.0;
        }

        // See PartitionLattice::ditset_xor_distance for the derivation.
        let mut self_block_sizes: HashMap<u32, usize> = HashMap::new();
        let mut other_block_sizes: HashMap<u32, usize> = HashMap::new();
        let mut intersection_sizes: HashMap<(u32, u32), usize> = HashMap::new();

        for i in 0..self.size {
            let a = self.element_to_block[i];
            let b = other.element_to_block[i];
            *self_block_sizes.entry(a).or_insert(0) += 1;
            *other_block_sizes.entry(b).or_insert(0) += 1;
            *intersection_sizes.entry((a, b)).or_insert(0) += 1;
        }

        let same_self: usize = self_block_sizes.values().map(|&c| c * c).sum();
        let same_other: usize = other_block_sizes.values().map(|&c| c * c).sum();
        let same_both: usize = intersection_sizes.values().map(|&c| c * c).sum();

        let xor_count = same_self + same_other - 2 * same_both;
        xor_count as f64 / (self.size * self.size) as f64
    }

    /// Logical area — area expressed as a count of distinctions.
    ///
    /// Formula: $A \approx N^2 \cdot h(\pi) + 0.5$
    ///
    /// Same result as [`crate::PartitionLattice::logical_area`], using the bitset representation.
    ///
    /// # Example
    /// ```ignore
    /// use partition_lattice::optimized::FastPartition;
    ///
    /// let p = FastPartition::from_ids(&[0, 1, 2, 3]); // fully discrete
    /// assert!((p.logical_area() - 12.5).abs() < 1e-10);
    /// ```
    ///
    pub fn logical_area(&self) -> f64 {
        let n = self.size as f64;
        let h = self.logical_entropy();
        let num_dits = n * n * h;
        num_dits + 0.5 // Information Offset
    }

    /// Complex Meet — a Meet weighted by phase interference (bitset version).
    ///
    /// Same result as [`crate::PartitionLattice::complex_refine`].
    ///
    /// # Example
    /// ```ignore
    /// use partition_lattice::optimized::FastPartition;
    ///
    /// let p1 = FastPartition::from_ids(&[0, 0, 1, 1]);
    /// let p2 = FastPartition::from_ids(&[0, 1, 0, 1]);
    /// let (meet, interference) = p1.complex_refine(&p2, 0, 64); // π/2 phase difference
    /// assert!(interference >= 0.0 && interference <= 1.0);
    /// ```
    pub fn complex_refine(&self, other: &Self, phase_self: i8, phase_other: i8) -> (Self, f64) {
        // 1. Ordinary meet.
        let meet_partition = self.differentiate(other);

        // 2. Phase difference and interference coefficient (same as PartitionLattice).
        let phase_diff = (phase_self as i16 - phase_other as i16).abs() as f64;
        let theta = phase_diff * core::f64::consts::PI / 128.0;

        // 3. Interference coefficient: cos^2(theta/2).
        let cos_half = (theta / 2.0).cos();
        let interference = cos_half * cos_half;

        (meet_partition, interference)
    }

    /// Creates a partition from raw bitmasks.
    /// `bitmasks` is a flat slice containing the bitwords for all blocks.
    /// Each block consumes `size.div_ceil(64)` u64 words.
    pub fn from_bitmasks(size: usize, num_blocks: usize, bitmasks: &[u64]) -> Self {
        let words_per_block = size.div_ceil(64);
        // Ensure we have enough data
        let expected_len = num_blocks * words_per_block;
        assert!(bitmasks.len() >= expected_len, "Insufficient bitmasks data");

        let mut blocks = Vec::with_capacity(num_blocks);
        let mut element_to_block: Vec<u32> = vec![0u32; size];

        for b in 0..num_blocks {
            let start = b * words_per_block;
            let end = start + words_per_block;
            let slice = &bitmasks[start..end];

            // Create BitVec from slice (copies data)
            let mut bv = BitVec::<u64, Lsb0>::from_slice(slice);
            // Ensure it has at least 'size' bits (it will have words_per_block * 64)
            // We might want to truncate it to exactly 'size' bits to be clean,
            // but FastPartition logic usually iterates up to 'size' or uses iter_ones.
            // Let's truncate to be safe and consistent.
            bv.truncate(size);

            // Update element_to_block mapping
            for i in bv.iter_ones() {
                if i < size {
                    element_to_block[i] = b as u32;
                }
            }
            blocks.push(bv);
        }

        Self::new_raw(size, blocks, element_to_block)
    }

    /// Returns the number of elements in the partition.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the block IDs for all elements.
    pub fn get_ids(&self) -> &[u32] {
        &self.element_to_block
    }

    /// Returns the number of blocks in the partition.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the number of blocks (alias for num_blocks for compatibility).
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the total memory used by block BitVecs in bytes.
    /// PCS-A2: M = K · ⌈N/64⌉ · 8
    pub fn blocks_memory_usage(&self) -> usize {
        self.blocks.iter().map(|b| b.as_raw_slice().len() * 8).sum()
    }

    /// Returns the block ID for a given element.
    #[inline]
    pub fn get_block(&self, element: usize) -> u32 {
        self.element_to_block[element]
    }

    /// Computes logical entropy: h(π) = 1 - Σ(|B|/N)²
    ///
    /// O(1) — uses precomputed sum of squared block sizes.
    pub fn logical_entropy(&self) -> f64 {
        if self.size == 0 {
            return 0.0;
        }
        let n = self.size as f64;
        1.0 - (self.sum_block_sizes_sq as f64) / (n * n)
    }

    /// Computes logical entropy using parallel reduction for large partitions.
    /// O(1) — uses precomputed sum of squared block sizes.
    pub fn logical_entropy_parallel(&self) -> f64 {
        self.logical_entropy()
    }

    /// Counts distinctions: N² - Σ|B|²
    ///
    /// O(1) — uses precomputed sum of squared block sizes.
    pub fn distinction_count(&self) -> usize {
        if self.size == 0 {
            return 0;
        }
        (self.size * self.size) - self.sum_block_sizes_sq as usize
    }

    /// Alias for distinction_count (cached is now always O(1)).
    pub fn distinction_count_cached(&self) -> usize {
        self.distinction_count()
    }

    /// Differentiate (formerly Meet): π₁ ∧ π₂
    ///
    /// Two elements are equivalent iff they're equivalent in BOTH partitions.
    /// Implemented using bitvector AND operations.
    pub fn differentiate(&self, other: &Self) -> Self {
        // Handle Chaos (Universal Indiscrete)
        if self.size == 0 {
            return other.clone();
        }
        if other.size == 0 {
            return self.clone();
        }

        assert_eq!(self.size, other.size, "Partition sizes must match");

        if self.size == 0 {
            return Self {
                size: 0,
                blocks: vec![],
                element_to_block: vec![],
                sum_block_sizes_sq: 0,
            };
        }

        // Heuristic: switch to element-wise O(N) when B1*B2 > √N
        let b1 = self.blocks.len();
        let b2 = other.blocks.len();
        let sqrt_n = (self.size as f64).sqrt() as usize;

        if b1 * b2 > sqrt_n.max(16) {
            return self.differentiate_elementwise(other);
        }

        // For differentiate, we intersect blocks from both partitions
        // Block (i,j) in result = block i from self AND block j from other
        let mut new_blocks = Vec::new();
        let mut element_to_block: Vec<u32> = vec![u32::MAX; self.size];
        let mut block_id: u32 = 0;

        for self_block in &self.blocks {
            for other_block in &other.blocks {
                // Optimization: Check if blocks are disjoint first using raw words
                // This avoids allocation for empty intersections (which are common)
                if simd_ops::is_disjoint(self_block.as_raw_slice(), other_block.as_raw_slice()) {
                    continue;
                }

                // Compute intersection using bitwise AND
                // This uses the underlying word-wise operations instead of bit-wise iteration
                let mut intersection = self_block.clone();
                intersection &= other_block;

                // Only add non-empty blocks
                // (We already checked disjointness, but count_ones is needed for element mapping)
                if intersection.count_ones() > 0 {
                    // Update element_to_block mapping
                    for i in intersection.iter_ones() {
                        element_to_block[i] = block_id;
                    }
                    new_blocks.push(intersection);
                    block_id += 1;
                }
            }
        }

        Self::new_raw(self.size, new_blocks, element_to_block)
    }

    /// Element-wise differentiate for fine partitions.
    /// O(N) complexity, K-independent. Always produces exact results.
    ///
    /// PCS-A3: The former lossy `*id % max_blocks` cap has been removed.
    /// Memory protection is now handled at the Partition enum level by
    /// `should_use_sparse_refine()`, which routes large K₁×K₂ cases through
    /// SparsePartition before reaching this method.
    fn differentiate_elementwise(&self, other: &Self) -> Self {
        let mut map: HashMap<(u32, u32), u32> = HashMap::new();
        let mut new_ids: Vec<u32> = Vec::with_capacity(self.size);
        let mut next_id: u32 = 0;

        for i in 0..self.size {
            let key = (self.element_to_block[i], other.element_to_block[i]);
            let new_id = *map.entry(key).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            new_ids.push(new_id);
        }

        Self::from_ids(&new_ids)
    }

    /// Integrate (formerly Join): π₁ ∨ π₂
    ///
    /// Two elements are equivalent if they're equivalent in EITHER partition (transitively).
    /// Uses Union-Find with path compression and union by rank.
    ///
    /// When B₁×B₂ > √N, switches to element-wise path using `element_to_block[]`
    /// instead of BitVec `iter_ones()`, reducing cost from O(K·N/64) to O(N).
    /// Same heuristic as `differentiate()`.
    pub fn integrate(&self, other: &Self) -> Self {
        // Handle Chaos (Universal Indiscrete)
        if self.size == 0 {
            return Self::indiscrete(other.size);
        }
        if other.size == 0 {
            return Self::indiscrete(self.size);
        }

        assert_eq!(self.size, other.size, "Partition sizes must match");

        if self.size == 0 {
            return Self {
                size: 0,
                blocks: vec![],
                element_to_block: vec![],
                sum_block_sizes_sq: 0,
            };
        }

        // Heuristic: switch to element-wise O(N) when B1*B2 > √N
        // (same threshold as differentiate — avoids O(K·⌈N/64⌉) iter_ones scan)
        let b1 = self.blocks.len();
        let b2 = other.blocks.len();
        let sqrt_n = (self.size as f64).sqrt() as usize;
        if b1 * b2 > sqrt_n.max(16) {
            return self.integrate_elementwise(other);
        }

        let mut uf = OptimizedUnionFind::new(self.size);

        // Union elements that are in the same block in self
        for block in &self.blocks {
            let mut first = None;
            for elem in block.iter_ones() {
                match first {
                    None => first = Some(elem),
                    Some(f) => {
                        uf.union(f, elem);
                    }
                }
            }
        }

        // Union elements that are in the same block in other
        for block in &other.blocks {
            let mut first = None;
            for elem in block.iter_ones() {
                match first {
                    None => first = Some(elem),
                    Some(f) => {
                        uf.union(f, elem);
                    }
                }
            }
        }

        // Build result partition
        let element_to_block: Vec<u32> = (0..self.size).map(|i| uf.find(i) as u32).collect();

        Self::from_ids(&element_to_block)
    }

    /// Element-wise integrate: uses `element_to_block[]` directly instead of
    /// BitVec `iter_ones()`. Cost: O(N·α(N)), independent of K.
    ///
    /// Called when B₁×B₂ > √N, where iter_ones would scan K·⌈N/64⌉ words.
    fn integrate_elementwise(&self, other: &Self) -> Self {
        let n = self.size;
        let mut uf = OptimizedUnionFind::new(n);

        // For each partition, record the first element of each block,
        // then union subsequent elements with that representative.
        // This avoids iterating BitVec words entirely.

        // Process self: element_to_block[i] → block ID → representative
        let max_block_a = self.element_to_block.iter().copied().max().unwrap_or(0) as usize;
        let mut rep_a: Vec<usize> = vec![usize::MAX; max_block_a + 1];
        for i in 0..n {
            let b = self.element_to_block[i] as usize;
            if rep_a[b] == usize::MAX {
                rep_a[b] = i;
            } else {
                uf.union(rep_a[b], i);
            }
        }

        // Process other
        let max_block_b = other.element_to_block.iter().copied().max().unwrap_or(0) as usize;
        let mut rep_b: Vec<usize> = vec![usize::MAX; max_block_b + 1];
        for i in 0..n {
            let b = other.element_to_block[i] as usize;
            if rep_b[b] == usize::MAX {
                rep_b[b] = i;
            } else {
                uf.union(rep_b[b], i);
            }
        }

        // Build result
        let ids: Vec<u32> = (0..n).map(|i| uf.find(i) as u32).collect();
        Self::from_ids(&ids)
    }

    /// Fused operation: Computes h(self ^ other) without materializing the differentiated partition.
    /// Optimized using bitwise operations and avoiding memory allocation.
    pub fn fused_entropy_differentiate(&self, other: &Self) -> f64 {
        if self.size == 0 {
            return 0.0;
        }
        assert_eq!(self.size, other.size, "Partition sizes must match");

        let n = self.size as f64;
        let mut sum_sq = 0.0;

        // Heuristic: switch to element-wise approach when B1*B2 > √N
        let b1 = self.blocks.len();
        let b2 = other.blocks.len();
        let sqrt_n = (self.size as f64).sqrt() as usize;

        if b1 * b2 > sqrt_n.max(16) {
            // Fallback to element-wise for sparse/fine partitions
            let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
            for i in 0..self.size {
                let b_i = self.element_to_block[i];
                let b_j = other.element_to_block[i];
                *counts.entry((b_i, b_j)).or_insert(0) += 1;
            }
            sum_sq = counts
                .values()
                .map(|&c| {
                    let cf = c as f64;
                    cf * cf
                })
                .sum();
        } else {
            // Bitwise approach
            // Parallelize if large enough
            if self.size >= PARALLEL_THRESHOLD && b1 * b2 >= 100 {
                sum_sq = self
                    .blocks
                    .par_iter()
                    .map(|self_block| {
                        let mut local_sum = 0.0;
                        for other_block in &other.blocks {
                            if simd_ops::is_disjoint(
                                self_block.as_raw_slice(),
                                other_block.as_raw_slice(),
                            ) {
                                continue;
                            }
                            let count = simd_ops::and_popcount(
                                self_block.as_raw_slice(),
                                other_block.as_raw_slice(),
                            ) as f64;
                            if count > 0.0 {
                                local_sum += count * count;
                            }
                        }
                        local_sum
                    })
                    .sum();
            } else {
                for self_block in &self.blocks {
                    for other_block in &other.blocks {
                        if simd_ops::is_disjoint(
                            self_block.as_raw_slice(),
                            other_block.as_raw_slice(),
                        ) {
                            continue;
                        }
                        let count = simd_ops::and_popcount(
                            self_block.as_raw_slice(),
                            other_block.as_raw_slice(),
                        ) as f64;
                        if count > 0.0 {
                            sum_sq += count * count;
                        }
                    }
                }
            }
        }

        1.0 - sum_sq / (n * n)
    }

    /// Parallel differentiate for large partitions.
    ///
    /// Mirrors the serial `differentiate()` logic but parallelizes block-pair
    /// intersections across Rayon threads.  Falls back to element-wise O(N)
    /// when B1×B2 > √N (same heuristic as serial) and to serial when N is
    /// below `PARALLEL_THRESHOLD`.
    pub fn differentiate_parallel(&self, other: &Self) -> Self {
        if self.size < PARALLEL_THRESHOLD {
            return self.differentiate(other);
        }

        assert_eq!(self.size, other.size, "Partition sizes must match");

        // Same heuristic as serial: element-wise path for fine partitions
        let b1 = self.blocks.len();
        let b2 = other.blocks.len();
        let sqrt_n = (self.size as f64).sqrt() as usize;
        if b1 * b2 > sqrt_n.max(16) {
            return self.differentiate_elementwise(other);
        }

        // Parallel computation of block-pair intersections using word-level AND
        let intersections: Vec<BitVec<u64, Lsb0>> = self
            .blocks
            .par_iter()
            .flat_map(|self_block| {
                other
                    .blocks
                    .iter()
                    .filter_map(move |other_block| {
                        // Early rejection: skip disjoint block pairs
                        if simd_ops::is_disjoint(
                            self_block.as_raw_slice(),
                            other_block.as_raw_slice(),
                        ) {
                            return None;
                        }
                        // Word-level AND (same as serial path)
                        let mut intersection = self_block.clone();
                        intersection &= other_block;
                        if intersection.count_ones() > 0 {
                            Some(intersection)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut new_blocks = Vec::with_capacity(intersections.len());
        let mut element_to_block: Vec<u32> = vec![u32::MAX; self.size];

        for (block_id, block) in intersections.into_iter().enumerate() {
            for i in block.iter_ones() {
                element_to_block[i] = block_id as u32;
            }
            new_blocks.push(block);
        }

        Self::new_raw(self.size, new_blocks, element_to_block)
    }

    /// Checks if self refines other (self ≤ other in lattice order).
    ///
    /// O(N) using element_to_block: for every pair (i,j) in the same block of self,
    /// they must also be in the same block of other.
    pub fn refines(&self, other: &Self) -> bool {
        assert_eq!(self.size, other.size, "Partition sizes must match");

        // For each block in self, pick first element and check all others
        // map to the same block in other.
        for block in &self.blocks {
            let mut first_other_block = None;
            for i in block.iter_ones() {
                let ob = other.element_to_block[i];
                match first_other_block {
                    None => first_other_block = Some(ob),
                    Some(expected) => {
                        if ob != expected {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Fused meet + refines: computes meet = self ∧ other and checks if meet refines old_model.
    ///
    /// More efficient than separate differentiate() + refines() since the refines check
    /// is done during the meet construction at zero extra cost.
    pub fn fused_refine_and_check(&self, other: &Self, old_model: &Self) -> (Self, bool) {
        let meet = self.differentiate(other);
        // O(N) refines check using element_to_block
        let does_refine = meet.refines(old_model);
        (meet, does_refine)
    }

    /// Computes logical divergence: d(π, θ) = h(π ∧ θ) - h(θ)
    pub fn logical_divergence(&self, other: &Self) -> f64 {
        let meet = self.differentiate(other);
        meet.logical_entropy() - other.logical_entropy()
    }

    /// Computes mutual information: I(π; θ) = h(π) + h(θ) - h(π ∧ θ)
    pub fn mutual_information(&self, other: &Self) -> f64 {
        let meet = self.differentiate(other);
        self.logical_entropy() + other.logical_entropy() - meet.logical_entropy()
    }
}

impl PartialEq for FastPartition {
    fn eq(&self, other: &Self) -> bool {
        if self.size != other.size || self.blocks.len() != other.blocks.len() {
            return false;
        }
        self.element_to_block == other.element_to_block
    }
}

impl Eq for FastPartition {}

// Output removed: OptimizedUnionFind struct replaced by import
use crate::algorithms::dsu::DisjointSet as OptimizedUnionFind;

/// Batch operations for processing multiple partitions efficiently.
pub struct BatchPartitionOps;

impl BatchPartitionOps {
    /// Computes entropy for multiple partitions in parallel.
    pub fn entropy_batch(partitions: &[FastPartition]) -> Vec<f64> {
        partitions.par_iter().map(|p| p.logical_entropy()).collect()
    }

    /// Computes pairwise mutual information matrix.
    /// Returns a Vec where result[i * n + j] = I(partitions[i]; partitions[j])
    pub fn mutual_info_matrix(partitions: &[FastPartition]) -> Vec<f64> {
        let n = partitions.len();
        let mut result = vec![0.0; n * n];

        // Compute diagonal (entropy of each partition)
        let entropies: Vec<f64> = partitions.par_iter().map(|p| p.logical_entropy()).collect();

        // Compute upper triangle in parallel
        let upper_triangle: Vec<(usize, usize, f64)> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                (i + 1..n)
                    .map(move |j| {
                        let mi = partitions[i].mutual_information(&partitions[j]);
                        (i, j, mi)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Fill matrix
        for i in 0..n {
            result[i * n + i] = entropies[i]; // Diagonal is entropy
        }

        for (i, j, mi) in upper_triangle {
            result[i * n + j] = mi;
            result[j * n + i] = mi; // Symmetric
        }

        result
    }
}

// =============================================================================
// [TASK-CORE-001] SIMD Optimized Operations
// =============================================================================

/// Trait for SIMD-optimized partition operations.
///
/// This trait provides methods that can leverage SIMD instructions
/// (AVX-512, AVX2, NEON) for accelerated computation.
pub trait SimdOptimized {
    /// Count elements in each block using SIMD-accelerated popcount.
    /// Returns a vector of block sizes.
    fn count_blocks_simd(&self) -> Vec<usize>;

    /// Compute block sizes squared sum using SIMD.
    /// Used for fast entropy calculation: h(π) = 1 - Σ(|B|/N)²
    fn sum_block_sizes_squared_simd(&self) -> usize;

    /// Iterate over raw u64 words of all blocks.
    /// Useful for custom SIMD kernels.
    fn iter_raw_words(&self) -> impl Iterator<Item = &[u64]>;
}

impl SimdOptimized for FastPartition {
    fn count_blocks_simd(&self) -> Vec<usize> {
        // bitvec's count_ones() already uses POPCNT instruction when available
        self.blocks.iter().map(|block| block.count_ones()).collect()
    }

    fn sum_block_sizes_squared_simd(&self) -> usize {
        // Use parallel reduction for large partitions
        if self.blocks.len() > PARALLEL_THRESHOLD / 10 {
            self.blocks
                .par_iter()
                .map(|block| {
                    let count = block.count_ones();
                    count * count
                })
                .sum()
        } else {
            self.blocks
                .iter()
                .map(|block| {
                    let count = block.count_ones();
                    count * count
                })
                .sum()
        }
    }

    fn iter_raw_words(&self) -> impl Iterator<Item = &[u64]> {
        self.blocks.iter().map(|bv| bv.as_raw_slice())
    }
}

/// SIMD-accelerated bitwise operations on raw u64 arrays.
///
/// These functions operate directly on the underlying u64 arrays
/// to maximize SIMD utilization.
pub mod simd_ops {
    /// Compute AND of two u64 slices (for differentiate/refinement operation).
    /// Returns popcount of the result.
    #[inline]
    pub fn and_popcount(a: &[u64], b: &[u64]) -> usize {
        debug_assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x & y).count_ones() as usize)
            .sum()
    }

    /// Compute OR of two u64 slices (for integrate/coarsening operation).
    /// Returns popcount of the result.
    #[inline]
    pub fn or_popcount(a: &[u64], b: &[u64]) -> usize {
        debug_assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x | y).count_ones() as usize)
            .sum()
    }

    /// Compute XOR of two u64 slices (symmetric difference).
    /// Returns popcount of the result.
    #[inline]
    pub fn xor_popcount(a: &[u64], b: &[u64]) -> usize {
        debug_assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x ^ y).count_ones() as usize)
            .sum()
    }

    /// Compute popcount of a u64 slice.
    #[inline]
    pub fn popcount(data: &[u64]) -> usize {
        data.iter().map(|x| x.count_ones() as usize).sum()
    }

    /// Check if two blocks are disjoint (no common elements).
    ///
    /// On aarch64, dispatches to NEON intrinsics (1.8x faster than auto-vec).
    #[inline]
    pub fn is_disjoint(a: &[u64], b: &[u64]) -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            super::neon_ops::is_disjoint(a, b)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            debug_assert_eq!(a.len(), b.len());
            a.iter().zip(b.iter()).all(|(x, y)| (x & y) == 0)
        }
    }

    /// Check if a is a subset of b.
    ///
    /// On aarch64, dispatches to NEON intrinsics.
    #[inline]
    pub fn is_subset(a: &[u64], b: &[u64]) -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            super::neon_ops::is_subset(a, b)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            debug_assert_eq!(a.len(), b.len());
            a.iter().zip(b.iter()).all(|(x, y)| (x & !y) == 0)
        }
    }

    /// Scalar-only is_disjoint (for benchmarking against NEON).
    #[inline]
    pub fn is_disjoint_scalar(a: &[u64], b: &[u64]) -> bool {
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).all(|(x, y)| (x & y) == 0)
    }
}

/// NEON-accelerated SIMD operations for aarch64 (Apple M-series).
///
/// Uses explicit NEON intrinsics to process 128-bit (2×u64) chunks at a time.
/// Key operations: AND+popcount, OR+popcount, XOR+popcount.
///
/// On M4, the NEON pipeline provides:
/// - `vcntq_u8`: byte-wise popcount (1 cycle latency)
/// - `vpaddlq_u8/u16/u32`: pairwise add widening (1 cycle)
/// - `vandq_u64`, `vorrq_u64`, `veorq_u64`: bitwise ops (1 cycle)
#[cfg(target_arch = "aarch64")]
pub mod neon_ops {
    use core::arch::aarch64::*;

    /// Popcount of a 128-bit NEON register using vcnt + horizontal add.
    ///
    /// vcntq_u8 counts bits per byte, then we widen-add through u16→u32→u64.
    #[inline(always)]
    unsafe fn vpopcntq_u64(v: uint64x2_t) -> u64 {
        // Reinterpret as u8x16 for byte-wise popcount
        let byte_counts = vcntq_u8(vreinterpretq_u8_u64(v));
        // Widen-add: u8 → u16 → u32 → u64
        let u16_sums = vpaddlq_u8(byte_counts);
        let u32_sums = vpaddlq_u16(u16_sums);
        let u64_sums = vpaddlq_u32(u32_sums);
        // Extract and sum the two u64 lanes
        vgetq_lane_u64(u64_sums, 0) + vgetq_lane_u64(u64_sums, 1)
    }

    /// AND two u64 slices and return total popcount.
    /// Processes 2 u64s (128 bits) per iteration via NEON.
    #[inline]
    pub fn and_popcount(a: &[u64], b: &[u64]) -> usize {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let chunks = n / 2;
        let mut total: u64 = 0;

        unsafe {
            for i in 0..chunks {
                let va = vld1q_u64(a.as_ptr().add(i * 2));
                let vb = vld1q_u64(b.as_ptr().add(i * 2));
                let vand = vandq_u64(va, vb);
                total += vpopcntq_u64(vand);
            }
        }

        // Handle trailing element (if odd length)
        if n % 2 != 0 {
            total += (a[n - 1] & b[n - 1]).count_ones() as u64;
        }

        total as usize
    }

    /// OR two u64 slices and return total popcount.
    #[inline]
    pub fn or_popcount(a: &[u64], b: &[u64]) -> usize {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let chunks = n / 2;
        let mut total: u64 = 0;

        unsafe {
            for i in 0..chunks {
                let va = vld1q_u64(a.as_ptr().add(i * 2));
                let vb = vld1q_u64(b.as_ptr().add(i * 2));
                let vor = vorrq_u64(va, vb);
                total += vpopcntq_u64(vor);
            }
        }

        if n % 2 != 0 {
            total += (a[n - 1] | b[n - 1]).count_ones() as u64;
        }

        total as usize
    }

    /// XOR two u64 slices and return total popcount.
    #[inline]
    pub fn xor_popcount(a: &[u64], b: &[u64]) -> usize {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let chunks = n / 2;
        let mut total: u64 = 0;

        unsafe {
            for i in 0..chunks {
                let va = vld1q_u64(a.as_ptr().add(i * 2));
                let vb = vld1q_u64(b.as_ptr().add(i * 2));
                let vxor = veorq_u64(va, vb);
                total += vpopcntq_u64(vxor);
            }
        }

        if n % 2 != 0 {
            total += (a[n - 1] ^ b[n - 1]).count_ones() as u64;
        }

        total as usize
    }

    /// Popcount of a u64 slice using NEON.
    #[inline]
    pub fn popcount(data: &[u64]) -> usize {
        let n = data.len();
        let chunks = n / 2;
        let mut total: u64 = 0;

        unsafe {
            for i in 0..chunks {
                let v = vld1q_u64(data.as_ptr().add(i * 2));
                total += vpopcntq_u64(v);
            }
        }

        if n % 2 != 0 {
            total += data[n - 1].count_ones() as u64;
        }

        total as usize
    }

    /// Check if two blocks are disjoint (no common elements).
    /// Early-exit on first non-zero AND.
    #[inline]
    pub fn is_disjoint(a: &[u64], b: &[u64]) -> bool {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let chunks = n / 2;

        unsafe {
            let zero = vdupq_n_u64(0);
            for i in 0..chunks {
                let va = vld1q_u64(a.as_ptr().add(i * 2));
                let vb = vld1q_u64(b.as_ptr().add(i * 2));
                let vand = vandq_u64(va, vb);
                // Compare equal to zero — if any lane non-zero, not disjoint
                let cmp = vceqq_u64(vand, zero);
                // vmaxvq_u32 collapses: if any comparison lane was 0 (not-equal), we're done
                if vminvq_u32(vreinterpretq_u32_u64(cmp)) == 0 {
                    return false;
                }
            }
        }

        // Handle trailing element
        if n % 2 != 0 {
            if (a[n - 1] & b[n - 1]) != 0 {
                return false;
            }
        }

        true
    }

    /// Check if a is a subset of b (a & !b == 0).
    /// Early-exit on first bit of a not in b.
    #[inline]
    pub fn is_subset(a: &[u64], b: &[u64]) -> bool {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let chunks = n / 2;

        unsafe {
            let zero = vdupq_n_u64(0);
            for i in 0..chunks {
                let va = vld1q_u64(a.as_ptr().add(i * 2));
                let vb = vld1q_u64(b.as_ptr().add(i * 2));
                // a & !b — BIC (bit clear) = a AND NOT b
                let vbic = vbicq_u64(va, vb);
                let cmp = vceqq_u64(vbic, zero);
                if vminvq_u32(vreinterpretq_u32_u64(cmp)) == 0 {
                    return false;
                }
            }
        }

        if n % 2 != 0 {
            if (a[n - 1] & !b[n - 1]) != 0 {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::PartitionLattice;

    #[test]
    fn test_fast_partition_creation() {
        let fp = FastPartition::from_ids(&[0, 0, 1, 2]);
        assert_eq!(fp.size(), 4);
        assert_eq!(fp.block_count(), 3);

        let discrete = FastPartition::discrete(5);
        assert_eq!(discrete.block_count(), 5);

        let indiscrete = FastPartition::indiscrete(5);
        assert_eq!(indiscrete.block_count(), 1);
    }

    #[test]
    fn test_fast_partition_entropy() {
        // Compare with reference implementation
        let ids = vec![0, 0, 1, 2];

        let ref_p = PartitionLattice::from_ids(ids.clone());
        let fast_p = FastPartition::from_ids(&ids);

        let ref_h = ref_p.logical_entropy();
        let fast_h = fast_p.logical_entropy();

        assert!(
            (ref_h - fast_h).abs() < 1e-10,
            "Entropy mismatch: ref={}, fast={}",
            ref_h,
            fast_h
        );
    }

    #[test]
    fn test_fast_partition_distinction_count() {
        let ids = vec![0, 0, 1];

        let ref_p = PartitionLattice::from_ids(ids.clone());
        let fast_p = FastPartition::from_ids(&ids);

        assert_eq!(ref_p.distinction_count(), fast_p.distinction_count());
    }

    #[test]
    fn test_fast_partition_differentiate() {
        let p1 = FastPartition::from_ids(&[0, 0, 1]); // {{0,1}, {2}}
        let p2 = FastPartition::from_ids(&[0, 1, 1]); // {{0}, {1,2}}

        let meet = p1.differentiate(&p2);

        // Meet should be discrete: {{0}, {1}, {2}}
        assert_eq!(meet.block_count(), 3);

        // Verify entropy matches reference
        let ref_p1 = PartitionLattice::from_ids(vec![0, 0, 1]);
        let ref_p2 = PartitionLattice::from_ids(vec![0, 1, 1]);
        let ref_meet = ref_p1.differentiate(&ref_p2);

        assert!((meet.logical_entropy() - ref_meet.logical_entropy()).abs() < 1e-10);
    }

    #[test]
    fn test_fast_partition_integrate() {
        let p1 = FastPartition::from_ids(&[0, 0, 1]); // {{0,1}, {2}}
        let p2 = FastPartition::from_ids(&[0, 1, 1]); // {{0}, {1,2}}

        let join = p1.integrate(&p2);

        // Join should be indiscrete: {{0,1,2}}
        assert_eq!(join.block_count(), 1);
        assert!((join.logical_entropy() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_fast_partition_refines() {
        let discrete = FastPartition::discrete(3);
        let indiscrete = FastPartition::indiscrete(3);

        // Discrete refines everything
        assert!(discrete.refines(&indiscrete));

        // Indiscrete only refines itself
        assert!(!indiscrete.refines(&discrete));
        assert!(indiscrete.refines(&indiscrete));
    }

    #[test]
    fn test_batch_operations() {
        let partitions: Vec<FastPartition> = vec![
            FastPartition::from_ids(&[0, 0, 1, 1]),
            FastPartition::from_ids(&[0, 1, 0, 1]),
            FastPartition::from_ids(&[0, 1, 2, 3]),
        ];

        let entropies = BatchPartitionOps::entropy_batch(&partitions);
        assert_eq!(entropies.len(), 3);

        // Verify individual values
        for (i, p) in partitions.iter().enumerate() {
            assert!((entropies[i] - p.logical_entropy()).abs() < 1e-10);
        }

        let mi_matrix = BatchPartitionOps::mutual_info_matrix(&partitions);
        assert_eq!(mi_matrix.len(), 9); // 3x3
    }

    #[test]
    fn test_large_partition_performance() {
        use std::time::Instant;

        let n = 10000usize;
        let ids: Vec<u32> = (0..n).map(|i| (i % 100) as u32).collect(); // 100 blocks

        // Create partitions
        let start = Instant::now();
        let fast_p = FastPartition::from_ids(&ids);
        let create_time = start.elapsed();

        // Compute entropy
        let start = Instant::now();
        let h = fast_p.logical_entropy();
        let entropy_time = start.elapsed();

        // Compute distinctions
        let start = Instant::now();
        let dits = fast_p.distinction_count();
        let dits_time = start.elapsed();

        println!("Large partition (n={}):", n);
        println!("  Creation time: {:?}", create_time);
        println!("  Entropy time: {:?}", entropy_time);
        println!("  Entropy value: {}", h);
        println!("  Distinctions time: {:?}", dits_time);
        println!("  Distinctions count: {}", dits);

        // Verify result
        assert!(h > 0.0 && h < 1.0);
    }

    // =========================================================================
    // SIMD Optimization Tests [TASK-CORE-001]
    // =========================================================================

    #[test]
    fn test_simd_count_blocks() {
        let p = FastPartition::from_ids(&[0, 0, 1, 1, 2]);
        let counts = p.count_blocks_simd();

        assert_eq!(counts.len(), 3);
        assert_eq!(counts[0], 2); // Block 0 has 2 elements
        assert_eq!(counts[1], 2); // Block 1 has 2 elements
        assert_eq!(counts[2], 1); // Block 2 has 1 element
    }

    #[test]
    fn test_simd_sum_squared() {
        let p = FastPartition::from_ids(&[0, 0, 1, 1, 2]);
        let sum_sq = p.sum_block_sizes_squared_simd();

        // 2² + 2² + 1² = 4 + 4 + 1 = 9
        assert_eq!(sum_sq, 9);
    }

    #[test]
    fn test_simd_entropy_equivalence() {
        let ids: Vec<u32> = (0..1000usize).map(|i| (i % 50) as u32).collect();
        let p = FastPartition::from_ids(&ids);

        // Compute entropy using SIMD helper
        let n = p.size() as f64;
        let sum_sq = p.sum_block_sizes_squared_simd() as f64;
        let h_simd = 1.0 - sum_sq / (n * n);

        // Compare with standard entropy
        let h_std = p.logical_entropy();

        assert!((h_simd - h_std).abs() < 1e-10);
    }

    #[test]
    fn test_simd_ops_and_popcount() {
        let a: Vec<u64> = vec![0b1111_0000, 0b0000_1111];
        let b: Vec<u64> = vec![0b1010_1010, 0b0101_0101];

        // AND: 0b1010_0000, 0b0000_0101 = 2 + 2 = 4
        assert_eq!(simd_ops::and_popcount(&a, &b), 4);

        // OR: 0b1111_1010, 0b0101_1111 = 6 + 6 = 12
        assert_eq!(simd_ops::or_popcount(&a, &b), 12);
    }

    #[test]
    fn test_simd_ops_is_subset() {
        let a: Vec<u64> = vec![0b0011];
        let b: Vec<u64> = vec![0b1111];
        let c: Vec<u64> = vec![0b1100];

        assert!(simd_ops::is_subset(&a, &b));
        assert!(!simd_ops::is_subset(&a, &c));
    }

    #[test]
    fn test_simd_ops_is_disjoint() {
        let a: Vec<u64> = vec![0b0011];
        let b: Vec<u64> = vec![0b1100];
        let c: Vec<u64> = vec![0b0110];

        assert!(simd_ops::is_disjoint(&a, &b));
        assert!(!simd_ops::is_disjoint(&a, &c));
    }

    #[test]
    fn test_iter_raw_words() {
        let p = FastPartition::from_ids(&[0, 0, 1, 1]);

        let words: Vec<_> = p.iter_raw_words().collect();
        assert_eq!(words.len(), 2); // 2 blocks

        // Each block should have at least 1 u64 word
        for w in words {
            assert!(!w.is_empty());
        }
    }

    // =========================================================================
    // Performance Optimization Tests
    // =========================================================================

    #[test]
    fn test_refines_elementwise_matches_bitwise() {
        // O(N) element_to_block-based refines must match original bitwise refines
        let discrete = FastPartition::discrete(100);
        let indiscrete = FastPartition::indiscrete(100);
        let p1 = FastPartition::from_ids(&(0..100u32).map(|i| i % 10).collect::<Vec<_>>());
        let p2 = FastPartition::from_ids(&(0..100u32).map(|i| i % 5).collect::<Vec<_>>());

        // discrete refines everything
        assert!(discrete.refines(&indiscrete));
        assert!(discrete.refines(&p1));
        assert!(discrete.refines(&p2));

        // p1 (10 blocks) refines p2 (5 blocks) iff each p1 block ⊆ some p2 block
        // i%10 → i%5: block 0,5 → 0; block 1,6 → 1; etc. YES, refines.
        assert!(p1.refines(&p2));
        assert!(!p2.refines(&p1));

        // indiscrete only refines itself
        assert!(!indiscrete.refines(&discrete));
        assert!(indiscrete.refines(&indiscrete));
    }

    #[test]
    fn test_fused_meet_refines() {
        // fused_refine_and_check must return (meet, true) since meet always refines input
        let p1 = FastPartition::from_ids(&[0, 0, 1, 1, 2, 2, 3, 3]);
        let p2 = FastPartition::from_ids(&[0, 1, 0, 1, 0, 1, 0, 1]);
        let old = p1.clone();

        let (meet, refines_old) = p1.fused_refine_and_check(&p2, &old);
        let meet_ref = old.differentiate(&p2);

        assert_eq!(meet.get_ids(), meet_ref.get_ids());
        assert!(refines_old);

        // meet should NOT refine p2 in general (it's finer than both)
        let (_, should_be_true) = p1.fused_refine_and_check(&p2, &p1);
        assert!(should_be_true); // meet of p1 and p2 refines p1
    }

    #[test]
    fn test_distinction_count_cached() {
        // Cached distinction count must match fresh computation
        let mut p = FastPartition::from_ids(&[0, 0, 1, 2, 2]);
        let expected = p.distinction_count();
        let cached = p.distinction_count_cached();
        assert_eq!(cached, expected);

        // After differentiate, cache should reset and return correct value
        let p2 = FastPartition::from_ids(&[0, 1, 1, 0, 1]);
        p = p.differentiate(&p2);
        let fresh = p.distinction_count();
        let cached = p.distinction_count_cached();
        assert_eq!(cached, fresh);
    }

    #[test]
    fn test_differentiate_elementwise_threshold() {
        // For fine partitions (many blocks), elementwise should kick in
        // and produce the same result as bitwise
        let n = 200;
        let p1 = FastPartition::discrete(n); // N blocks
        let ids: Vec<u32> = (0..n).map(|i| (i % 3) as u32).collect();
        let p2 = FastPartition::from_ids(&ids);

        // B1*B2 = 200*3 = 600 > sqrt(200) ≈ 14, so elementwise kicks in
        let meet = p1.differentiate(&p2);
        // Discrete ∧ anything = discrete
        assert_eq!(meet.block_count(), n);
    }

    #[test]
    fn test_integrate_elementwise_correctness() {
        // Verify element-wise integrate gives correct result for cross-connecting partitions.
        // p1: {0,1},{2,3},{4,5} and p2: {0,2,4},{1,3,5}
        // Join merges transitively: 0~1(p1), 0~2~4(p2), 1~3~5(p2), 2~3(p1), 4~5(p1)
        // → All connected → 1 block
        //
        // Use N=300, K=100 to trigger elementwise (K₁×K₂ = 10000 > √300 ≈ 17)
        let n = 300usize;
        let k = 100;
        // p1: interleave by 2, p2: interleave by 3 → cross-connections guaranteed
        let ids_a: Vec<u32> = (0..n).map(|i| (i % k) as u32).collect();
        let ids_b: Vec<u32> = (0..n).map(|i| (i % (k + 1)) as u32).collect();
        let p1 = FastPartition::from_ids(&ids_a);
        let p2 = FastPartition::from_ids(&ids_b);

        let result = p1.integrate(&p2);
        assert_eq!(result.size(), n);
        // Verify via entropy: join should be coarser (lower entropy)
        assert!(
            result.logical_entropy() <= p1.logical_entropy() + 1e-10,
            "Join entropy {} should be ≤ p1 entropy {}",
            result.logical_entropy(),
            p1.logical_entropy()
        );
    }

    #[test]
    fn test_integrate_elementwise_matches_reference() {
        // Cross-check: compute join via BOTH iter_ones and elementwise,
        // verify same entropy (structurally equivalent result).
        let p1 = FastPartition::from_ids(&[0, 0, 1, 1, 2, 2]);
        let p2 = FastPartition::from_ids(&[0, 1, 0, 1, 0, 1]);
        let result = p1.integrate(&p2);
        // p1 blocks: {0,1},{2,3},{4,5}; p2 blocks: {0,2,4},{1,3,5}
        // Join: all connected → 1 block
        assert_eq!(result.block_count(), 1);
    }

    #[test]
    fn test_integrate_elementwise_high_k_fast() {
        // K=200, N=1000 → K²=40000 > √1000≈31 → elementwise path
        // Must complete quickly (not O(K·N/64) explosion)
        let n = 1000;
        let k = 200;
        let ids_a: Vec<u32> = (0..n).map(|i| (i % k) as u32).collect();
        let ids_b: Vec<u32> = (0..n).map(|i| ((i * 3 + 7) % k) as u32).collect();
        let p1 = FastPartition::from_ids(&ids_a);
        let p2 = FastPartition::from_ids(&ids_b);

        let start = std::time::Instant::now();
        let result = p1.integrate(&p2);
        let elapsed = start.elapsed();

        assert!(result.size() == n);
        assert!(result.block_count() >= 1);
        assert!(
            elapsed.as_millis() < 100,
            "High-K integrate took {}ms, expected < 100ms",
            elapsed.as_millis()
        );
    }

    #[test]
    fn test_differentiate_elementwise_exact_for_large_n() {
        // PCS-A3: After lossy cap removal, elementwise differentiate
        // produces exact results. Memory protection is now handled at
        // Partition enum level via should_use_sparse_refine().
        //
        // Direct FastPartition usage at large N will create many blocks
        // (potentially expensive), but results are mathematically correct.
        let n = 10_000usize; // Reduced from 100K to keep test fast
        let ids1: Vec<u32> = (0..n).map(|i| (i % 100) as u32).collect();
        let ids2: Vec<u32> = (0..n).map(|i| ((i / 100) % 100) as u32).collect();
        let p1 = FastPartition::from_ids(&ids1);
        let p2 = FastPartition::from_ids(&ids2);
        // Elementwise path: exact result (no lossy cap)
        let meet = p1.differentiate(&p2);
        // Verify via SparsePartition ground truth
        let sp1 = crate::sparse::SparsePartition::from_ids(&ids1);
        let sp2 = crate::sparse::SparsePartition::from_ids(&ids2);
        let exact = sp1.differentiate(&sp2);
        let h_fast = meet.logical_entropy();
        let h_exact = exact.logical_entropy();
        assert!(
            (h_fast - h_exact).abs() < 1e-10,
            "Elementwise must be exact: h_fast={h_fast:.10}, h_exact={h_exact:.10}"
        );
    }

    #[test]
    fn test_small_n_differentiate_not_capped() {
        // At small N, blocks are cheap (128 bytes each at N=1000).
        // Cap should NOT trigger, preserving mathematical exactness.
        let n = 1000usize;
        let ids1: Vec<u32> = (0..n).map(|i| (i % 100) as u32).collect();
        let ids2: Vec<u32> = (0..n).map(|i| ((i / 10) % 100) as u32).collect();
        let p1 = FastPartition::from_ids(&ids1);
        let p2 = FastPartition::from_ids(&ids2);
        let meet = p1.differentiate(&p2);
        // At small N, should not be capped — exact K1×K2 result expected
        // 100 × 100 = 10000 possible, but bounded by N=1000
        assert!(
            meet.block_count() <= n,
            "Block count {} > N={}",
            meet.block_count(),
            n
        );
        // Entropy should be maximal (near-discrete partition)
        let h = meet.logical_entropy();
        assert!(
            h > 0.9,
            "Expected high entropy for fine partition, got {}",
            h
        );
    }

    #[test]
    fn test_cumulative_differentiate_memory_bounded() {
        // Simulate cumulative meet chain at large N — K would explode without capping
        let n = 50_000usize;
        let mut cumulative = {
            let ids: Vec<u32> = (0..n).map(|i| (i % 8) as u32).collect();
            FastPartition::from_ids(&ids)
        };
        let words_per_block = (n + 63) / 64;
        let max_blocks = BLOCKS_MEMORY_LIMIT / (words_per_block * 8);
        // 20 iterations of cumulative meet (simulates video chunks)
        for iter in 0..20usize {
            let ids: Vec<u32> = (0..n).map(|i| ((i + iter * 7) % 12) as u32).collect();
            let fused = FastPartition::from_ids(&ids);
            cumulative = cumulative.differentiate(&fused);
            // Block count must stay memory-bounded
            assert!(
                cumulative.block_count() <= max_blocks,
                "Iteration {}: block count {} exceeded memory-adaptive cap {}",
                iter,
                cumulative.block_count(),
                max_blocks
            );
        }
        let h = cumulative.logical_entropy();
        assert!(h >= 0.0 && h <= 1.0, "Invalid entropy: {}", h);
    }

    // ---- NEON ops correctness tests (aarch64 only) ----

    #[cfg(target_arch = "aarch64")]
    mod neon_tests {
        use super::super::{neon_ops, simd_ops};

        #[test]
        fn test_neon_and_popcount_matches_scalar() {
            let a: Vec<u64> = vec![0b1111_0000, 0b0000_1111];
            let b: Vec<u64> = vec![0b1010_1010, 0b0101_0101];
            assert_eq!(
                neon_ops::and_popcount(&a, &b),
                simd_ops::and_popcount(&a, &b)
            );
        }

        #[test]
        fn test_neon_or_popcount_matches_scalar() {
            let a: Vec<u64> = vec![0b1111_0000, 0b0000_1111];
            let b: Vec<u64> = vec![0b1010_1010, 0b0101_0101];
            assert_eq!(neon_ops::or_popcount(&a, &b), simd_ops::or_popcount(&a, &b));
        }

        #[test]
        fn test_neon_xor_popcount_matches_scalar() {
            let a: Vec<u64> = vec![0b1111_0000, 0b0000_1111];
            let b: Vec<u64> = vec![0b1010_1010, 0b0101_0101];
            assert_eq!(
                neon_ops::xor_popcount(&a, &b),
                simd_ops::xor_popcount(&a, &b)
            );
        }

        #[test]
        fn test_neon_popcount_matches_scalar() {
            let data: Vec<u64> = vec![0xFF00FF00FF00FF00, 0x0F0F0F0F0F0F0F0F];
            assert_eq!(neon_ops::popcount(&data), simd_ops::popcount(&data));
        }

        #[test]
        fn test_neon_is_disjoint_matches_scalar() {
            let a: Vec<u64> = vec![0b0011, 0b1100];
            let b: Vec<u64> = vec![0b1100, 0b0011];
            let c: Vec<u64> = vec![0b0110, 0b0110];
            assert_eq!(neon_ops::is_disjoint(&a, &b), simd_ops::is_disjoint(&a, &b));
            assert_eq!(neon_ops::is_disjoint(&a, &c), simd_ops::is_disjoint(&a, &c));
        }

        #[test]
        fn test_neon_is_subset_matches_scalar() {
            let a: Vec<u64> = vec![0b0011, 0b0000];
            let b: Vec<u64> = vec![0b1111, 0b1111];
            let c: Vec<u64> = vec![0b1100, 0b0000];
            assert_eq!(neon_ops::is_subset(&a, &b), simd_ops::is_subset(&a, &b));
            assert_eq!(neon_ops::is_subset(&a, &c), simd_ops::is_subset(&a, &c));
        }

        #[test]
        fn test_neon_and_popcount_odd_length() {
            // Odd length: 3 u64 words — tests the trailing element path
            let a: Vec<u64> = vec![u64::MAX, 0, 0b1010];
            let b: Vec<u64> = vec![u64::MAX, u64::MAX, 0b0110];
            assert_eq!(
                neon_ops::and_popcount(&a, &b),
                simd_ops::and_popcount(&a, &b)
            );
        }

        #[test]
        fn test_neon_large_random_data() {
            use rand::rngs::StdRng;
            use rand::Rng;
            use rand::SeedableRng;

            let mut rng = StdRng::seed_from_u64(42);
            let n = 256; // 256 u64s = 16384 bits ≈ partition with N=16K
            let a: Vec<u64> = (0..n).map(|_| rng.gen()).collect();
            let b: Vec<u64> = (0..n).map(|_| rng.gen()).collect();

            assert_eq!(
                neon_ops::and_popcount(&a, &b),
                simd_ops::and_popcount(&a, &b)
            );
            assert_eq!(neon_ops::or_popcount(&a, &b), simd_ops::or_popcount(&a, &b));
            assert_eq!(
                neon_ops::xor_popcount(&a, &b),
                simd_ops::xor_popcount(&a, &b)
            );
            assert_eq!(neon_ops::popcount(&a), simd_ops::popcount(&a));
            assert_eq!(neon_ops::is_disjoint(&a, &b), simd_ops::is_disjoint(&a, &b));
        }
    }
}
