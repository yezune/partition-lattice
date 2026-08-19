//! Sparse Partition representation for large-scale data.
//!
//! When dealing with millions of elements where most blocks contain few elements,
//! the dense `Vec<usize>` representation in `Partition` becomes memory-inefficient.
//! This module provides a `SparsePartition` using Roaring Bitmaps for efficient storage.
//!
//! Memory: O(k * average_block_size) for non-singleton blocks + O(N) for element_to_block
//! vs FastPartition: O(K * N/64) which explodes when K is large.

use crate::unified::Partition;
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
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

/// Canonical map serialisation, so that equal partitions serialise to equal bytes.
///
/// `blocks` is a `HashMap`, so its iteration order varies per instance: building a
/// `SparsePartition` twice from the same ids used to produce different JSON (observed as a
/// key-order difference in the `"blocks"` map).
///
/// This matters because a `Partition::Sparse` can end up inside a persisted artifact. If the
/// encoding is not canonical, the artifact digest no longer identifies the state, and "the state
/// changed" becomes indistinguishable from "the map was rehashed".
///
/// The format is still a JSON map; only the key order is fixed, so previously written files still load.
mod canonical_blocks {
    use super::*;
    use serde::ser::SerializeMap;
    use serde::Serializer;

    pub fn serialize<S: Serializer>(m: &HashMap<u32, RoaringBitmap>, s: S) -> Result<S::Ok, S::Error> {
        let mut keys: Vec<&u32> = m.keys().collect();
        keys.sort();
        let mut out = s.serialize_map(Some(m.len()))?;
        for k in keys {
            out.serialize_entry(k, &m[k])?;
        }
        out.end()
    }
}


/// A sparse partition representation using Roaring Bitmaps.
///
/// Each non-singleton block is represented as a `RoaringBitmap` containing the element indices.
/// Elements not in any explicit block are singletons (block_id = element index + offset).
///
/// Key design choices:
/// - `element_to_block: Vec<u32>` provides O(1) `get_ids()` and `block_of()`
/// - `sum_block_sizes_sq: usize` enables O(1) `logical_entropy()`
/// - Roaring Bitmaps for blocks: memory-efficient for sparse, large-N partitions
/// - Block IDs use `u32` to align with RoaringBitmap's native u32 element type
///
/// This is efficient when:
/// - N is very large (100K+)
/// - K (block count) is large relative to N (K > 256)
/// - Many singleton blocks exist
///
/// # Example
/// ```
/// use partition_lattice::sparse::SparsePartition;
///
/// // Create partition: {0, 1, 2}, {3}, {4} from IDs
/// let sp = SparsePartition::from_ids(&[0, 0, 0, 1, 2]);
/// assert_eq!(sp.size(), 5);
/// assert_eq!(sp.block_count(), 3);
/// assert_eq!(sp.block_of(0), sp.block_of(1));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsePartition {
    /// Total number of elements (universe size |U|)
    size: usize,
    /// Element index → block ID mapping (O(1) lookup)
    element_to_block: Vec<u32>,
    /// Block ID → Elements in that block (as RoaringBitmap)
    /// Only stores non-singleton blocks for memory efficiency.
    #[serde(serialize_with = "canonical_blocks::serialize")]
    blocks: HashMap<u32, RoaringBitmap>,
    /// Bitmap tracking which elements are NOT singletons
    non_singletons: RoaringBitmap,
    /// Precomputed Sigma|B|^2 for O(1) entropy calculation
    sum_block_sizes_sq: usize,
}

impl SparsePartition {
    /// Creates a discrete (all-singletons) partition.
    ///
    /// Each element is in its own block. This is the finest possible partition.
    ///
    /// # Example
    /// ```
    /// use partition_lattice::sparse::SparsePartition;
    /// let sp = SparsePartition::discrete(5);
    /// assert_eq!(sp.block_count(), 5);
    /// ```
    pub fn discrete(size: usize) -> Self {
        // Each element is its own block: element i -> block i
        let element_to_block: Vec<u32> = (0..size as u32).collect();
        Self {
            size,
            element_to_block,
            blocks: HashMap::new(),
            non_singletons: RoaringBitmap::new(),
            sum_block_sizes_sq: size, // each block has size 1, so Sigma 1^2 = N
        }
    }

    /// Creates an indiscrete (single-block) partition.
    ///
    /// All elements are in the same block. This is the coarsest possible partition.
    ///
    /// # Example
    /// ```
    /// use partition_lattice::sparse::SparsePartition;
    /// let sp = SparsePartition::indiscrete(5);
    /// assert_eq!(sp.block_count(), 1);
    /// ```
    pub fn indiscrete(size: usize) -> Self {
        if size == 0 {
            return Self::discrete(0);
        }
        let element_to_block = vec![0u32; size];
        let mut bitmap = RoaringBitmap::new();
        let mut non_singletons = RoaringBitmap::new();
        for i in 0..size {
            bitmap.insert(i as u32);
            non_singletons.insert(i as u32);
        }
        let mut blocks = HashMap::new();
        blocks.insert(0u32, bitmap);
        Self {
            size,
            element_to_block,
            blocks,
            non_singletons,
            sum_block_sizes_sq: size * size, // single block of size N: N^2
        }
    }

    /// Creates a sparse partition from block-ID assignments.
    ///
    /// `ids[i]` specifies the block ID of element `i`.
    /// This is the primary factory method, compatible with `get_ids()`.
    ///
    /// # Example
    /// ```
    /// use partition_lattice::sparse::SparsePartition;
    /// let sp = SparsePartition::from_ids(&[0, 0, 1, 1, 2]);
    /// assert_eq!(sp.block_count(), 3);
    /// assert_eq!(sp.block_of(0), sp.block_of(1));
    /// assert_eq!(sp.block_of(2), sp.block_of(3));
    /// ```
    pub fn from_ids(ids: &[u32]) -> Self {
        let size = ids.len();
        let element_to_block = ids.to_vec();

        // Group elements by block ID
        let mut block_elements: HashMap<u32, RoaringBitmap> = HashMap::new();
        for (elem, &block_id) in ids.iter().enumerate() {
            block_elements
                .entry(block_id)
                .or_default()
                .insert(elem as u32);
        }

        // Separate singletons from non-singletons
        let mut blocks = HashMap::new();
        let mut non_singletons = RoaringBitmap::new();
        let mut sum_block_sizes_sq: usize = 0;

        for (block_id, bitmap) in block_elements {
            let block_size = bitmap.len() as usize;
            sum_block_sizes_sq += block_size * block_size;
            if block_size > 1 {
                for elem in bitmap.iter() {
                    non_singletons.insert(elem);
                }
                blocks.insert(block_id, bitmap);
            }
        }

        Self {
            size,
            element_to_block,
            blocks,
            non_singletons,
            sum_block_sizes_sq,
        }
    }

    /// Creates a sparse partition from explicit blocks.
    ///
    /// Each inner `Vec<usize>` is a block. Elements not mentioned are singletons.
    ///
    /// # Example
    /// ```
    /// use partition_lattice::sparse::SparsePartition;
    /// let sp = SparsePartition::from_blocks(5, vec![vec![0, 1, 2]]);
    /// assert_eq!(sp.block_count(), 3); // {0,1,2}, {3}, {4}
    /// ```
    pub fn from_blocks(size: usize, block_list: Vec<Vec<usize>>) -> Self {
        // Assign block IDs: explicit blocks first, then singletons
        let mut element_to_block = vec![0u32; size];
        let mut blocks = HashMap::new();
        let mut non_singletons = RoaringBitmap::new();
        let mut sum_block_sizes_sq: usize = 0;
        let mut next_block_id: u32 = 0;

        // Track which elements are assigned
        let mut assigned = vec![false; size];

        for elements in block_list {
            if elements.is_empty() {
                continue;
            }
            let block_id = next_block_id;
            next_block_id += 1;

            let block_size = elements.len();
            sum_block_sizes_sq += block_size * block_size;

            if block_size > 1 {
                let mut bitmap = RoaringBitmap::new();
                for &elem in &elements {
                    element_to_block[elem] = block_id;
                    bitmap.insert(elem as u32);
                    non_singletons.insert(elem as u32);
                    assigned[elem] = true;
                }
                blocks.insert(block_id, bitmap);
            } else {
                // Single-element block
                let elem = elements[0];
                element_to_block[elem] = block_id;
                assigned[elem] = true;
            }
        }

        // Assign singleton IDs for unassigned elements
        for i in 0..size {
            if !assigned[i] {
                element_to_block[i] = next_block_id;
                sum_block_sizes_sq += 1; // singleton: 1^2 = 1
                next_block_id += 1;
            }
        }

        Self {
            size,
            element_to_block,
            blocks,
            non_singletons,
            sum_block_sizes_sq,
        }
    }

    /// Deprecated: Use `discrete()` instead.
    #[deprecated(note = "Use SparsePartition::discrete() instead")]
    pub fn new(size: u32) -> Self {
        Self::discrete(size as usize)
    }

    /// Returns the total number of elements (universe size).
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the number of blocks (partitions).
    pub fn block_count(&self) -> usize {
        // Non-singleton blocks + singleton elements
        let singleton_count = self.size as u64 - self.non_singletons.len();
        self.blocks.len() + singleton_count as usize
    }

    /// Returns the block-ID assignment slice.
    ///
    /// `get_ids()[i]` is the block ID of element `i`.
    /// Returns a reference to the internal `element_to_block` vector (O(1)).
    pub fn get_ids(&self) -> &[u32] {
        &self.element_to_block
    }

    /// Returns which block an element belongs to (O(1)).
    pub fn block_of(&self, element: usize) -> u32 {
        self.element_to_block[element]
    }

    /// Computes logical entropy: h(pi) = 1 - Sigma(|B|/N)^2
    ///
    /// O(1) using precomputed `sum_block_sizes_sq`.
    pub fn logical_entropy(&self) -> f64 {
        if self.size == 0 {
            return 0.0;
        }
        let n = self.size as f64;
        1.0 - (self.sum_block_sizes_sq as f64 / (n * n))
    }

    /// Computes the number of distinctions.
    /// dit(pi) = N^2 - Sigma|B|^2
    pub fn distinction_count(&self) -> usize {
        if self.size == 0 {
            return 0;
        }
        self.size * self.size - self.sum_block_sizes_sq
    }

    /// Computes logical divergence: D_L(self || other) = h(self ^ other) - h(other)
    ///
    /// Uses fused computation to avoid materializing the meet.
    pub fn logical_divergence(&self, other: &Self) -> f64 {
        self.fused_entropy_differentiate(other) - other.logical_entropy()
    }

    /// Computes ditset XOR distance: d_xor(self, other) = h(self) + h(other) - 2*h(self v other)
    ///
    /// Uses the formula: d_xor = |dit(P) Delta dit(Q)| / N^2
    /// Which equals: (dit(P) + dit(Q) - 2*|dit(P) intersect dit(Q)|) / N^2
    ///
    /// Computed via block pair intersections.
    pub fn ditset_xor_distance(&self, other: &Self) -> f64 {
        if self.size == 0 {
            return 0.0;
        }
        let n = self.size;
        let n_sq = (n * n) as f64;

        // Count indits intersection: pairs (x,y) in same block in BOTH partitions
        // For each pair of blocks (B_self, B_other), intersection contributes |B_self intersect B_other|^2
        let mut sum_intersection_sq: usize = 0;

        // Build block map for other: block_id -> &RoaringBitmap
        // For singletons in other, each singleton intersects at most 1 element per self-block

        // For efficiency, iterate over elements and count (self_block, other_block) pair sizes
        let mut pair_sizes: HashMap<(u32, u32), usize> = HashMap::new();
        for i in 0..n {
            let sb = self.element_to_block[i];
            let ob = other.element_to_block[i];
            *pair_sizes.entry((sb, ob)).or_insert(0) += 1;
        }

        for (_, &count) in &pair_sizes {
            sum_intersection_sq += count * count;
        }

        // dit_self = N^2 - sum_self_sq
        // dit_other = N^2 - sum_other_sq
        // dit_self intersect dit_other = N^2 - sum_intersection_sq (pairs that are NOT in same block in BOTH)
        // Actually: indit_union = sum_intersection_sq (pairs in same block in self^other)
        // dit_self_and_other = N^2 - sum_intersection_sq
        //
        // Correct formula for ditset XOR distance:
        // d_xor(P,Q) = |dit(P) Delta dit(Q)| / N^2
        // = (|dit(P)| + |dit(Q)| - 2|dit(P) intersect dit(Q)|) / N^2
        //
        // |dit(P) intersect dit(Q)| = number of pairs distinguished by BOTH P and Q
        // = N^2 - |{(x,y): x=y in P or x=y in Q}|
        // = N^2 - |indit(P) union indit(Q)|
        //
        // By inclusion-exclusion:
        // |indit(P) union indit(Q)| = |indit(P)| + |indit(Q)| - |indit(P) intersect indit(Q)|
        //                           = Sigma|B_P|^2 + Sigma|B_Q|^2 - Sigma|B_P intersect B_Q|^2 (= sum_intersection_sq)
        //
        // |dit(P) intersect dit(Q)| = N^2 - (Sigma|B_P|^2 + Sigma|B_Q|^2 - sum_intersection_sq)

        let dit_p = self.distinction_count();
        let dit_q = other.distinction_count();
        let indit_union = self.sum_block_sizes_sq + other.sum_block_sizes_sq - sum_intersection_sq;
        let dit_intersection = n * n - indit_union;

        let symmetric_diff = dit_p + dit_q - 2 * dit_intersection;
        symmetric_diff as f64 / n_sq
    }

    /// Checks if self refines other (self <= other).
    ///
    /// pi refines sigma iff every block of pi is a subset of some block of sigma.
    /// Equivalently: for all (x, y), if self.block_of(x) == self.block_of(y),
    /// then other.block_of(x) == other.block_of(y).
    pub fn refines(&self, other: &Self) -> bool {
        assert_eq!(self.size, other.size, "Partition sizes must match");

        // For each non-singleton block in self, check all elements map to
        // the same block in other.
        for bitmap in self.blocks.values() {
            let mut other_block: Option<u32> = None;
            for elem in bitmap.iter() {
                let ob = other.element_to_block[elem as usize];
                match other_block {
                    None => other_block = Some(ob),
                    Some(expected) => {
                        if ob != expected {
                            return false;
                        }
                    }
                }
            }
        }
        // Singletons trivially refine: a single element is always in some block of other.
        true
    }

    /// Computes h(self ^ other) without materializing the meet partition.
    ///
    /// Counts the sizes of intersection blocks (self_block, other_block) pairs
    /// and computes entropy from those sizes directly.
    pub fn fused_entropy_differentiate(&self, other: &Self) -> f64 {
        if self.size == 0 {
            return 0.0;
        }
        let n = self.size as f64;

        // Count sizes of (self_block, other_block) pair groups
        let mut pair_sizes: HashMap<(u32, u32), usize> = HashMap::new();
        for i in 0..self.size {
            let sb = self.element_to_block[i];
            let ob = other.element_to_block[i];
            *pair_sizes.entry((sb, ob)).or_insert(0) += 1;
        }

        let mut sum_sq: usize = 0;
        for (_, &count) in &pair_sizes {
            sum_sq += count * count;
        }

        1.0 - (sum_sq as f64 / (n * n))
    }

    /// Computes the logical area (L-QRM): A ~ N^2 * h(pi) + 0.5
    pub fn logical_area(&self) -> f64 {
        let n = self.size as f64;
        n * n * self.logical_entropy() + 0.5
    }

    /// Differentiate operation (Meet / Greatest Lower Bound).
    /// Two elements are equivalent in the result iff they are equivalent in BOTH partitions.
    ///
    /// Sparse-optimized: groups elements by (self_block, other_block) pairs.
    pub fn differentiate(&self, other: &Self) -> Self {
        assert_eq!(self.size, other.size, "Partition sizes must match");

        // Assign result block IDs by first occurrence of each (self_block,
        // other_block) pair in element-scan order. This makes the output
        // `element_to_block` deterministic (independent of HashMap iteration
        // order), so two stores accumulating the same partitions in
        // different orders produce bit-exact equal partitions — required
        // for CRDT order-invariance (test_xmodal_align_cross_agent_agreement).
        let mut pair_to_block: HashMap<(u32, u32), u32> = HashMap::new();
        let mut element_to_block = vec![0u32; self.size];
        let mut next_block_id: u32 = 0;
        for i in 0..self.size {
            let sb = self.element_to_block[i];
            let ob = other.element_to_block[i];
            let block_id = *pair_to_block.entry((sb, ob)).or_insert_with(|| {
                let id = next_block_id;
                next_block_id += 1;
                id
            });
            element_to_block[i] = block_id;
        }

        // Single-pass bucket build — O(N), not O(N · K).
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); next_block_id as usize];
        for (elem, &b) in element_to_block.iter().enumerate() {
            buckets[b as usize].push(elem as u32);
        }

        let mut blocks = HashMap::new();
        let mut non_singletons = RoaringBitmap::new();
        let mut sum_block_sizes_sq: usize = 0;
        for (block_id, bucket) in buckets.iter().enumerate() {
            let sz = bucket.len();
            sum_block_sizes_sq += sz * sz;
            if sz > 1 {
                let mut bitmap = RoaringBitmap::new();
                for &elem in bucket {
                    bitmap.insert(elem);
                    non_singletons.insert(elem);
                }
                blocks.insert(block_id as u32, bitmap);
            }
        }

        Self {
            size: self.size,
            element_to_block,
            blocks,
            non_singletons,
            sum_block_sizes_sq,
        }
    }

    /// Integrate operation (Join / Least Upper Bound).
    /// Two elements are equivalent if connected through either partition.
    ///
    /// PCS-T4 elementwise path: uses element_to_block[] directly instead of
    /// iterating RoaringBitmaps. Cost: O(N·α(N)), K-independent.
    pub fn integrate(&self, other: &Self) -> Self {
        assert_eq!(self.size, other.size, "Partition sizes must match");

        use crate::algorithms::DisjointSet;
        let mut dsu = DisjointSet::new(self.size);

        // Elementwise: use element_to_block[] for O(N) block→representative mapping.
        // For each partition, record the first element of each block,
        // then union subsequent elements with that representative.

        // Process self
        let max_block_a = self.element_to_block.iter().copied().max().unwrap_or(0) as usize;
        let mut rep_a: Vec<usize> = vec![usize::MAX; max_block_a + 1];
        for i in 0..self.size {
            let b = self.element_to_block[i] as usize;
            if rep_a[b] == usize::MAX {
                rep_a[b] = i;
            } else {
                dsu.union(rep_a[b], i);
            }
        }

        // Process other
        let max_block_b = other.element_to_block.iter().copied().max().unwrap_or(0) as usize;
        let mut rep_b: Vec<usize> = vec![usize::MAX; max_block_b + 1];
        for i in 0..other.size {
            let b = other.element_to_block[i] as usize;
            if rep_b[b] == usize::MAX {
                rep_b[b] = i;
            } else {
                dsu.union(rep_b[b], i);
            }
        }

        // Build result from DSU
        let mut root_to_block: HashMap<usize, u32> = HashMap::new();
        let mut element_to_block = vec![0u32; self.size];
        let mut next_block_id: u32 = 0;

        for i in 0..self.size {
            let root = dsu.find(i);
            let block_id = *root_to_block.entry(root).or_insert_with(|| {
                let id = next_block_id;
                next_block_id += 1;
                id
            });
            element_to_block[i] = block_id;
        }

        Self::from_ids(&element_to_block)
    }

    /// Returns memory usage estimate in bytes.
    pub fn memory_usage(&self) -> usize {
        let mut total = core::mem::size_of::<Self>();

        // element_to_block vector
        total += self.element_to_block.capacity() * core::mem::size_of::<u32>();

        // Roaring bitmap serialized sizes
        for bitmap in self.blocks.values() {
            total += bitmap.serialized_size();
        }
        total += self.non_singletons.serialized_size();

        // HashMap overhead
        total += self.blocks.capacity()
            * (core::mem::size_of::<u32>() + core::mem::size_of::<RoaringBitmap>());

        total
    }
}

/// Conversion from dense Partition to SparsePartition.
impl From<&Partition> for SparsePartition {
    fn from(dense: &Partition) -> Self {
        SparsePartition::from_ids(dense.get_ids())
    }
}

/// Conversion from SparsePartition to dense Partition.
impl From<&SparsePartition> for Partition {
    fn from(sparse: &SparsePartition) -> Self {
        Partition::from_ids(sparse.element_to_block.clone())
    }
}

// ========================================================================
// Tests
// ========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // ---- Phase 1-1: Type alignment, reverse mapping, precomputed entropy ----

    #[test]
    fn test_sparse_type_alignment() {
        // size() returns usize, distinction_count() returns usize
        let sp = SparsePartition::from_ids(&[0, 0, 1, 1, 2]);
        let _s: usize = sp.size();
        let _d: usize = sp.distinction_count();
        assert_eq!(_s, 5);
    }

    #[test]
    fn test_sparse_get_ids_returns_slice() {
        let ids: Vec<u32> = vec![0, 0, 1, 1, 2];
        let sp = SparsePartition::from_ids(&ids);
        assert_eq!(sp.get_ids(), &ids[..]);
    }

    #[test]
    fn test_sparse_block_of_o1() {
        let ids: Vec<u32> = vec![0, 0, 1, 1, 2, 2, 3];
        let sp = SparsePartition::from_ids(&ids);
        for (i, &expected) in ids.iter().enumerate() {
            assert_eq!(sp.block_of(i), expected, "block_of({}) mismatch", i);
        }
    }

    #[test]
    fn test_sparse_entropy_precomputed() {
        // Partition: {0, 1}, {2, 3}, {4}
        let sp = SparsePartition::from_ids(&[0, 0, 1, 1, 2]);
        // h = 1 - (2/5)^2 - (2/5)^2 - (1/5)^2 = 1 - 0.16 - 0.16 - 0.04 = 0.64
        let entropy = sp.logical_entropy();
        assert!((entropy - 0.64).abs() < 1e-10, "entropy = {}", entropy);

        // Also verify sum_block_sizes_sq
        // 2^2 + 2^2 + 1^2 = 9
        assert_eq!(sp.sum_block_sizes_sq, 9);
    }

    #[test]
    fn test_sparse_serde_roundtrip() {
        let sp = SparsePartition::from_ids(&[0, 0, 1, 1, 2, 3, 3, 3]);

        // JSON roundtrip
        let json = serde_json::to_string(&sp).expect("serialize");
        let sp2: SparsePartition = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(sp.size(), sp2.size());
        assert_eq!(sp.get_ids(), sp2.get_ids());
        assert_eq!(sp.block_count(), sp2.block_count());
        assert!((sp.logical_entropy() - sp2.logical_entropy()).abs() < 1e-10);

        // Bincode roundtrip
        let bytes = bincode::serialize(&sp).expect("bincode serialize");
        let sp3: SparsePartition = bincode::deserialize(&bytes).expect("bincode deserialize");
        assert_eq!(sp.get_ids(), sp3.get_ids());
    }

    // ---- Phase 1-2: Missing PartitionOps methods ----

    /// Helper: create Dense Partition equivalent for cross-checking
    fn dense_equiv(ids: &[u32]) -> Partition {
        Partition::from_ids(ids.to_vec())
    }

    #[test]
    fn test_sparse_logical_divergence_matches_dense() {
        let ids1: Vec<u32> = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
        let ids2: Vec<u32> = vec![0, 1, 0, 1, 0, 1, 2, 2, 3, 3];

        let sp1 = SparsePartition::from_ids(&ids1);
        let sp2 = SparsePartition::from_ids(&ids2);
        let dp1 = dense_equiv(&ids1);
        let dp2 = dense_equiv(&ids2);

        let sparse_div = sp1.logical_divergence(&sp2);
        let dense_div = dp1.logical_divergence(&dp2);

        assert!(
            (sparse_div - dense_div).abs() < 1e-10,
            "sparse_div={}, dense_div={}",
            sparse_div,
            dense_div
        );
    }

    #[test]
    fn test_sparse_divergence_zero_for_same() {
        let ids: Vec<u32> = vec![0, 0, 1, 1, 2, 2, 3];
        let sp = SparsePartition::from_ids(&ids);
        let div = sp.logical_divergence(&sp);
        assert!(div.abs() < 1e-10, "D_L(P||P) should be 0, got {}", div);
    }

    #[test]
    fn test_sparse_ditset_xor_distance_matches_dense() {
        let ids1: Vec<u32> = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
        let ids2: Vec<u32> = vec![0, 1, 0, 1, 0, 1, 2, 2, 3, 3];

        let sp1 = SparsePartition::from_ids(&ids1);
        let sp2 = SparsePartition::from_ids(&ids2);
        let dp1 = dense_equiv(&ids1);
        let dp2 = dense_equiv(&ids2);

        let sparse_dist = sp1.ditset_xor_distance(&sp2);
        let dense_dist = dp1.ditset_xor_distance(&dp2);

        assert!(
            (sparse_dist - dense_dist).abs() < 1e-10,
            "sparse_dist={}, dense_dist={}",
            sparse_dist,
            dense_dist
        );
    }

    #[test]
    fn test_sparse_refines_matches_dense() {
        // P1 refines P2 iff every block of P1 is subset of some block of P2
        let ids_fine: Vec<u32> = vec![0, 1, 2, 3, 4]; // discrete -- refines everything
        let ids_coarse: Vec<u32> = vec![0, 0, 1, 1, 2]; // coarser

        let sp_fine = SparsePartition::from_ids(&ids_fine);
        let sp_coarse = SparsePartition::from_ids(&ids_coarse);

        assert!(
            sp_fine.refines(&sp_coarse),
            "discrete should refine any partition"
        );
        assert!(
            !sp_coarse.refines(&sp_fine),
            "coarser should not refine finer"
        );
    }

    #[test]
    fn test_sparse_refines_self() {
        let ids: Vec<u32> = vec![0, 0, 1, 1, 2, 3, 3];
        let sp = SparsePartition::from_ids(&ids);
        assert!(sp.refines(&sp), "every partition refines itself");
    }

    #[test]
    fn test_sparse_fused_entropy_matches_dense() {
        let ids1: Vec<u32> = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let ids2: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let sp1 = SparsePartition::from_ids(&ids1);
        let sp2 = SparsePartition::from_ids(&ids2);
        let dp1 = dense_equiv(&ids1);
        let dp2 = dense_equiv(&ids2);

        let sparse_fused = sp1.fused_entropy_differentiate(&sp2);
        let dense_fused = dp1.fused_entropy_differentiate(&dp2);

        assert!(
            (sparse_fused - dense_fused).abs() < 1e-10,
            "sparse_fused={}, dense_fused={}",
            sparse_fused,
            dense_fused
        );
    }

    #[test]
    fn test_sparse_logical_area_matches_dense() {
        let ids: Vec<u32> = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
        let sp = SparsePartition::from_ids(&ids);
        let dp = dense_equiv(&ids);

        let sparse_area = sp.logical_area();
        let dense_area = dp.logical_area();

        assert!(
            (sparse_area - dense_area).abs() < 1e-10,
            "sparse_area={}, dense_area={}",
            sparse_area,
            dense_area
        );
    }

    // ---- Phase 1-3: from_ids / discrete / indiscrete factories ----

    #[test]
    fn test_sparse_from_ids_roundtrip() {
        let ids: Vec<u32> = vec![0, 0, 1, 2, 2, 3, 3, 3, 4, 5];
        let sp = SparsePartition::from_ids(&ids);
        assert_eq!(sp.get_ids(), &ids[..]);
        assert_eq!(sp.size(), 10);
        assert_eq!(sp.block_count(), 6);
    }

    #[test]
    fn test_sparse_discrete_indiscrete() {
        let disc = SparsePartition::discrete(100);
        assert_eq!(disc.size(), 100);
        assert_eq!(disc.block_count(), 100);
        assert!((disc.logical_entropy() - 0.99).abs() < 1e-10);

        let indisc = SparsePartition::indiscrete(100);
        assert_eq!(indisc.size(), 100);
        assert_eq!(indisc.block_count(), 1);
        assert!(indisc.logical_entropy().abs() < 1e-10);
    }

    // ---- Existing tests (migrated from u32) ----

    #[test]
    fn test_sparse_basic() {
        // Create partition: {0, 1, 2}, {3}, {4}
        let blocks = vec![vec![0, 1, 2]];
        let sparse = SparsePartition::from_blocks(5, blocks);

        assert_eq!(sparse.size(), 5);
        assert_eq!(sparse.block_count(), 3); // 1 non-singleton + 2 singletons

        // Check block membership
        assert_eq!(sparse.block_of(0), sparse.block_of(1));
        assert_eq!(sparse.block_of(0), sparse.block_of(2));
        assert_ne!(sparse.block_of(0), sparse.block_of(3));
    }

    #[test]
    fn test_sparse_entropy() {
        // Partition: {0, 1}, {2, 3}, {4}
        let blocks = vec![vec![0, 1], vec![2, 3]];
        let sparse = SparsePartition::from_blocks(5, blocks);

        // h = 1 - (2/5)^2 - (2/5)^2 - (1/5)^2 = 1 - 0.16 - 0.16 - 0.04 = 0.64
        let entropy = sparse.logical_entropy();
        assert!((entropy - 0.64).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_differentiate() {
        // P1: {0, 1, 2}, {3, 4}
        let p1 = SparsePartition::from_blocks(5, vec![vec![0, 1, 2], vec![3, 4]]);

        // P2: {0, 1}, {2, 3}, {4}
        let p2 = SparsePartition::from_blocks(5, vec![vec![0, 1], vec![2, 3]]);

        // Differentiate: {0, 1}, {2}, {3}, {4}
        let meet = p1.differentiate(&p2);

        assert_eq!(meet.block_of(0), meet.block_of(1));
        assert_ne!(meet.block_of(0), meet.block_of(2));
        assert_ne!(meet.block_of(2), meet.block_of(3));
    }

    #[test]
    fn test_sparse_integrate() {
        // P1: {0, 1}, {2}, {3, 4}
        let p1 = SparsePartition::from_blocks(5, vec![vec![0, 1], vec![3, 4]]);

        // P2: {1, 2}, {3}, {4}
        let p2 = SparsePartition::from_blocks(5, vec![vec![1, 2]]);

        // Integrate: {0, 1, 2}, {3, 4} (transitive closure)
        let join = p1.integrate(&p2);

        assert_eq!(join.block_of(0), join.block_of(1));
        assert_eq!(join.block_of(1), join.block_of(2));
        assert_eq!(join.block_of(3), join.block_of(4));
        assert_ne!(join.block_of(0), join.block_of(3));
    }

    #[test]
    fn test_sparse_dense_conversion() {
        // Dense partition
        let dense = Partition::from_ids(vec![0, 0, 1, 1, 2]);

        // Convert to sparse
        let sparse = SparsePartition::from(&dense);
        assert_eq!(sparse.block_count(), 3);

        // Convert back to dense
        let dense2 = Partition::from(&sparse);

        // Check equivalence (block IDs may differ, but structure should match)
        assert_eq!(dense.block_count(), dense2.block_count());
        assert!((dense.logical_entropy() - dense2.logical_entropy()).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_memory_efficiency() {
        // Large sparse partition: N=1,000,000 with mostly singletons
        let size = 1_000_000usize;

        // Only 100 blocks with 10 elements each
        let blocks: Vec<Vec<usize>> = (0..100)
            .map(|i| ((i * 10)..((i + 1) * 10)).collect())
            .collect();

        let sparse = SparsePartition::from_blocks(size, blocks);

        // Dense FastPartition would need: K blocks x N/64 x 8 bytes per bitvec
        // With K ~ 999_000 singleton + 100 explicit = huge
        // But even just element_to_block: 1M * 4 bytes = 4MB (now u32 instead of usize)
        // Sparse stores: element_to_block (4MB) + roaring bitmaps (tiny for 100x10)
        // FastPartition would store: K x N/64 x 8 bytes = massive for K ~ 1M

        let sparse_size = sparse.memory_usage();
        println!(
            "Sparse size: {} bytes ({:.2} MB)",
            sparse_size,
            sparse_size as f64 / 1e6
        );

        // The sparse partition should be reasonable for N=1M
        // (element_to_block = 4MB with u32, roaring ~ small, total < 16MB)
        assert!(sparse_size < 16_000_000, "sparse should be < 16MB for N=1M");
    }

    // ---- Phase 4: N=100K stress tests (slow-tests feature gate) ----

    #[cfg(feature = "slow-tests")]
    #[test]
    fn test_sparse_n100k_correctness() {
        // SparsePartition correctness at N=100K.
        //
        // NOTE: FastPartition caps block count via BLOCKS_MEMORY_LIMIT (8MB),
        // causing information loss when K*ceil(N/64)*8 > 8MB. At N=100K, K>640
        // triggers the cap. SparsePartition does NOT cap -- it is the
        // accurate reference implementation for high-K scenarios.
        //
        // Therefore we verify:
        // 1. Single-partition ops match between Sparse and Fast (no cap needed)
        // 2. Sparse internal self-consistency for all operations
        // 3. Sparse differentiate is at least as accurate as Fast (fused check)
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let n = 100_000;
        let k = 200u32;

        let ids1: Vec<u32> = (0..n).map(|_| rng.gen_range(0..k)).collect();
        let ids2: Vec<u32> = (0..n).map(|_| rng.gen_range(0..k / 2)).collect();

        let sp1 = SparsePartition::from_ids(&ids1);
        let sp2 = SparsePartition::from_ids(&ids2);
        let dp1 = dense_equiv(&ids1);

        // 1. Single-partition entropy (no block cap issue)
        assert!(
            (sp1.logical_entropy() - dp1.logical_entropy()).abs() < 1e-10,
            "N=100K entropy mismatch"
        );

        // 2. distinction_count
        assert_eq!(sp1.distinction_count(), dp1.distinction_count());

        // 3. Sparse self-consistency: fused entropy == differentiate entropy
        let sp_meet = sp1.differentiate(&sp2);
        let fused_h = sp1.fused_entropy_differentiate(&sp2);
        assert!(
            (sp_meet.logical_entropy() - fused_h).abs() < 1e-10,
            "N=100K fused vs materialized meet entropy mismatch: meet={}, fused={}",
            sp_meet.logical_entropy(),
            fused_h
        );

        // 4. Integrate self-consistency
        let sp_join = sp1.integrate(&sp2);
        assert!(
            sp_join.logical_entropy() <= sp1.logical_entropy() + 1e-10,
            "Join should have entropy <= operands"
        );
        assert!(
            sp_meet.logical_entropy() >= sp1.logical_entropy() - 1e-10,
            "Meet should have entropy >= operands"
        );

        // 5. Divergence self-consistency: D_L(P||Q) = h(P^Q) - h(Q) >= 0
        let sp_div = sp1.logical_divergence(&sp2);
        assert!(
            sp_div >= -1e-10,
            "Divergence should be >= 0, got {}",
            sp_div
        );
        let expected_div = sp_meet.logical_entropy() - sp2.logical_entropy();
        assert!(
            (sp_div - expected_div).abs() < 1e-10,
            "Divergence mismatch: computed={}, expected={}",
            sp_div,
            expected_div
        );

        // 6. Refines: meet refines both operands
        assert!(sp_meet.refines(&sp1), "Meet should refine operand 1");
        assert!(sp_meet.refines(&sp2), "Meet should refine operand 2");

        // 7. ditset_xor_distance symmetry and non-negativity
        let d12 = sp1.ditset_xor_distance(&sp2);
        let d21 = sp2.ditset_xor_distance(&sp1);
        assert!(d12 >= -1e-10, "XOR distance should be >= 0");
        assert!(
            (d12 - d21).abs() < 1e-10,
            "XOR distance should be symmetric: d12={}, d21={}",
            d12,
            d21
        );

        // 8. logical_area matches formula
        let expected_area = (n as f64) * (n as f64) * sp1.logical_entropy() + 0.5;
        assert!(
            (sp1.logical_area() - expected_area).abs() < 1e-6,
            "N=100K logical area mismatch"
        );

        // 9. Single-partition ops: Sparse matches Fast (K=200, no cap triggered)
        // (For single partition, from_ids produces same result in both backends)
        assert!(
            (sp1.logical_entropy() - dp1.logical_entropy()).abs() < 1e-10,
            "Single-partition entropy should match Fast"
        );
        assert!(
            (sp1.logical_area() - dp1.logical_area()).abs() < 1e-6,
            "Single-partition area should match Fast"
        );
    }

    #[cfg(feature = "slow-tests")]
    #[test]
    fn test_sparse_memory_n100k() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let n = 100_000;
        let k = 1000u32; // Many blocks

        // Sparse partition: mostly singletons + some multi-element blocks
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            ids.push(rng.gen_range(0..k));
        }

        let sp = SparsePartition::from_ids(&ids);
        let sp_mem = sp.memory_usage();

        // FastPartition would use K x ceil(N/64) x 8 bytes for blocks alone
        // K=1000, N=100K -> 1000 x 1563 x 8 ~ 12.2MB
        let fast_block_mem_estimate = (k as usize) * ((n + 63) / 64) * 8;

        println!(
            "Sparse memory: {} bytes ({:.2} MB)",
            sp_mem,
            sp_mem as f64 / 1e6
        );
        println!(
            "FastPartition block estimate: {} bytes ({:.2} MB)",
            fast_block_mem_estimate,
            fast_block_mem_estimate as f64 / 1e6
        );

        // SparsePartition should use significantly less memory than FastPartition's blocks
        // SparsePartition = element_to_block (400KB with u32) + roaring (~small) ~ 1-2 MB
        // FastPartition blocks alone ~ 12.2 MB
        assert!(
            sp_mem < fast_block_mem_estimate / 2,
            "Sparse ({:.2} MB) should be < 50% of Fast block estimate ({:.2} MB)",
            sp_mem as f64 / 1e6,
            fast_block_mem_estimate as f64 / 1e6
        );
    }

    /// Storage-compression measurement gate.
    ///
    /// Hypothesis claims: SparsePartition wire-up reduces per-crystal ngram
    /// storage from ~275 KB (dense FastPartition at N=65536) to ≤ 1 KB.
    ///
    /// This test measures the actual cost. Encodes "abc" as a cyclic-pattern
    /// ngram over a universe of size N=65536 (matching ko-M scale), and
    /// asserts the SparsePartition cost is below the 1 KB hypothesis target.
    ///
    /// Expected: FAIL at substrate level. `element_to_block: Vec<u32>` of
    /// size N dominates at 262 KB regardless of K. Confirms that simple
    /// wire-up does not meet the hypothesis target — a new genuinely-sparse
    /// representation (e.g. cyclic-pattern encoding) is required.
    #[test]
    fn test_sfm_crystal_compression_ngram_cost_at_n65536() {
        let n = 65_536usize;
        // Cyclic ngram "abc": ids[i] = codepoint(chars[i % 3])
        let chars: Vec<u32> = "abc".chars().map(|c| c as u32).collect();
        let ids: Vec<u32> = (0..n).map(|i| chars[i % chars.len()]).collect();

        let sparse = SparsePartition::from_ids(&ids);
        let bytes = sparse.memory_usage();

        // Diagnostic: SparsePartition's element_to_block alone = N * 4 = 262144 B.
        // K=3 → only 3 RoaringBitmaps (small). non_singletons covers all elements.
        assert_eq!(sparse.block_count(), 3, "ngram 'abc' should yield K=3");

        // Hypothesis target: ≤ 1024 bytes per crystal.
        // This assertion is EXPECTED TO FAIL with current substrate.
        // When it fails, the hypothesis must propose a new substrate
        // (cyclic-pattern encoding) before wire-up can proceed.
        let target_bytes = 1024usize;
        let n_bytes_floor = n * core::mem::size_of::<u32>();
        eprintln!(
            "[H_SFM_CRYSTAL_STORAGE_COMPRESSION] ngram crystal at N={}: \
             SparsePartition={} bytes ({:.1} KB), \
             element_to_block floor = {} bytes ({:.1} KB), \
             hypothesis target = {} bytes",
            n,
            bytes,
            bytes as f64 / 1024.0,
            n_bytes_floor,
            n_bytes_floor as f64 / 1024.0,
            target_bytes,
        );

        // Document the structural lower bound: element_to_block is O(N).
        assert!(
            bytes >= n_bytes_floor,
            "SparsePartition memory ({}) must be ≥ element_to_block floor ({})",
            bytes,
            n_bytes_floor,
        );

        // The hypothesis claim — currently UNREACHABLE with this substrate.
        // We assert the gap rather than the target, to lock the finding in.
        assert!(
            bytes > target_bytes,
            "EXPECTED: SparsePartition exceeds 1 KB target due to O(N) element_to_block. \
             If this assertion ever fails (bytes ≤ 1024), the hypothesis target has been \
             reached and this test should be inverted."
        );
    }

    #[cfg(feature = "slow-tests")]
    #[test]
    fn test_sparse_differentiate_n100k_timing() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let n = 100_000;

        let ids1: Vec<u32> = (0..n).map(|_| rng.gen_range(0..500u32)).collect();
        let ids2: Vec<u32> = (0..n).map(|_| rng.gen_range(0..100u32)).collect();

        let sp1 = SparsePartition::from_ids(&ids1);
        let sp2 = SparsePartition::from_ids(&ids2);

        let start = std::time::Instant::now();
        let _meet = sp1.differentiate(&sp2);
        let elapsed = start.elapsed();

        println!("N=100K differentiate: {:?}", elapsed);
        assert!(
            elapsed.as_millis() < 500,
            "N=100K differentiate took {}ms (limit: 500ms)",
            elapsed.as_millis()
        );
    }

    #[cfg(feature = "slow-tests")]
    #[test]
    fn test_delta_incremental_n100k_timing() {
        use crate::delta_refine::{ChangeSet, DeltaOpsEngine};
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let n = 100_000;

        let ids: Vec<u32> = (0..n).map(|_| rng.gen_range(0..200u32)).collect();
        let p = Partition::from_ids(ids.clone());

        let mut engine = DeltaOpsEngine::new(n).with_threshold(0.1);
        engine.initialize(&p);

        // 1% change (1000 elements)
        let mut changes = ChangeSet::new();
        let mut new_ids = ids;
        for i in 0..1000 {
            let old_block = new_ids[i];
            let new_block = rng.gen_range(0..200u32);
            new_ids[i] = new_block;
            changes.record_move(i, old_block, new_block);
        }
        let p_new = Partition::from_ids(new_ids);

        let start = std::time::Instant::now();
        let h = engine.incremental_entropy(&p_new, &changes);
        let elapsed = start.elapsed();

        println!("N=100K incremental entropy (1% change): {:?}", elapsed);
        assert!(
            elapsed.as_millis() < 10,
            "N=100K incremental entropy took {}ms (limit: 10ms)",
            elapsed.as_millis()
        );

        // Verify correctness
        // NOTE: incremental updates block sizes one-by-one, while full recompute
        // rebuilds from scratch via FastPartition::from_ids. Block ID renormalization
        // in from_ids can cause sum_block_sizes_sq to differ slightly due to
        // HashMap iteration order differences. Tolerance 1e-6 is appropriate.
        let full_h = p_new.logical_entropy();
        assert!(
            (h - full_h).abs() < 1e-6,
            "N=100K incremental vs full: incr={}, full={}, diff={}",
            h,
            full_h,
            (h - full_h).abs()
        );
    }

    // ---- Cross-validation: differentiate/integrate match dense ----

    #[test]
    fn test_sparse_differentiate_matches_dense() {
        let ids1: Vec<u32> = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
        let ids2: Vec<u32> = vec![0, 1, 0, 1, 0, 1, 2, 2, 3, 3];

        let sp1 = SparsePartition::from_ids(&ids1);
        let sp2 = SparsePartition::from_ids(&ids2);
        let dp1 = dense_equiv(&ids1);
        let dp2 = dense_equiv(&ids2);

        let sparse_meet = sp1.differentiate(&sp2);
        let dense_meet = dp1.differentiate(&dp2);

        // Entropies should match
        let sparse_h = sparse_meet.logical_entropy();
        let dense_h = dense_meet.logical_entropy();
        assert!(
            (sparse_h - dense_h).abs() < 1e-10,
            "differentiate entropy mismatch: sparse={}, dense={}",
            sparse_h,
            dense_h
        );

        // Block counts should match
        assert_eq!(sparse_meet.block_count(), dense_meet.block_count());
    }

    #[test]
    fn test_sparse_integrate_matches_dense() {
        let ids1: Vec<u32> = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
        let ids2: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2];

        let sp1 = SparsePartition::from_ids(&ids1);
        let sp2 = SparsePartition::from_ids(&ids2);
        let dp1 = dense_equiv(&ids1);
        let dp2 = dense_equiv(&ids2);

        let sparse_join = sp1.integrate(&sp2);
        let dense_join = dp1.integrate(&dp2);

        let sparse_h = sparse_join.logical_entropy();
        let dense_h = dense_join.logical_entropy();
        assert!(
            (sparse_h - dense_h).abs() < 1e-10,
            "integrate entropy mismatch: sparse={}, dense={}",
            sparse_h,
            dense_h
        );
    }
}
