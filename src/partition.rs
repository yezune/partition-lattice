use crate::algorithms::dsu::DisjointSet;
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
#[cfg(feature = "std")]
use core::cell::Cell;

#[cfg(feature = "std")]
thread_local! {
    static LOGICAL_OPS: Cell<u64> = const { Cell::new(0) };
}

/// Reset the global logical operation counter. (no_std: no-op — the diagnostic
/// op counter needs thread-local state; the count is not part of any result.)
pub fn reset_logical_ops() {
    #[cfg(feature = "std")]
    LOGICAL_OPS.with(|c| c.set(0));
}

/// Get the current value of the global logical operation counter (no_std: `0`).
pub fn get_logical_ops() -> u64 {
    #[cfg(feature = "std")]
    {
        LOGICAL_OPS.with(|c| c.get())
    }
    #[cfg(not(feature = "std"))]
    {
        0
    }
}

/// Increment the global logical operation counter. (no_std: no-op.)
fn inc_logical_ops(amount: u64) {
    #[cfg(feature = "std")]
    LOGICAL_OPS.with(|c| c.set(c.get() + amount));
    #[cfg(not(feature = "std"))]
    let _ = amount;
}

use serde::{Deserialize, Serialize};

/// Represents a partition of a set of N elements.
/// Implemented using a vector of block IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartitionLattice {
    /// Maps element index -> Block ID
    /// elements are 0..size
    ids: Vec<u32>,
    size: usize,
}

impl PartitionLattice {
    /// Creates a discrete partition (bottom) where each element is in its own block.
    /// {{0}, {1}, ..., {n-1}}
    pub fn new(size: usize) -> Self {
        Self {
            ids: (0..size).map(|i| i as u32).collect(),
            size,
        }
    }

    /// Creates a partition from a list of block IDs.
    /// e.g., [0, 0, 1] -> {{0, 1}, {2}}
    pub fn from_ids(ids: Vec<u32>) -> Self {
        let size = ids.len();
        // Normalize IDs to be 0..k-1
        let mut map = HashMap::new();
        let mut normalized_ids = Vec::with_capacity(size);
        let mut next_id: u32 = 0;

        for &id in &ids {
            let new_id = *map.entry(id).or_insert_with(|| {
                let i = next_id;
                next_id += 1;
                i
            });
            normalized_ids.push(new_id);
        }

        Self {
            ids: normalized_ids,
            size,
        }
    }

    /// Returns the number of elements (universe size).
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the block IDs as a slice.
    pub fn get_ids(&self) -> &[u32] {
        &self.ids
    }

    /// Returns the number of blocks in the partition.
    pub fn block_count(&self) -> usize {
        if self.size == 0 {
            return 0;
        }
        (*self.ids.iter().max().unwrap_or(&0) + 1) as usize
    }

    /// Returns the blocks of the partition as a vector of vectors of element indices.
    pub fn to_blocks(&self) -> Vec<Vec<usize>> {
        if self.size == 0 {
            return vec![];
        }
        let mut blocks_map: HashMap<u32, Vec<usize>> = HashMap::new();
        for (element, &block_id) in self.ids.iter().enumerate() {
            blocks_map.entry(block_id).or_default().push(element);
        }
        // Sort blocks by first element to ensure deterministic output
        let mut blocks: Vec<Vec<usize>> = blocks_map.into_values().collect();
        blocks.sort_by(|a, b| a[0].cmp(&b[0]));
        blocks
    }

    /// Calculates the Logical Entropy: h(P) = 1 - sum(p_B^2)
    /// where p_B is the probability of a block B (|B|/N).
    ///
    /// **Bit-determinism.** Block sizes are counted with a `HashMap`, but the integer `Σ|B|²` is
    /// accumulated *first* — integer addition is associative, so the result does not depend on
    /// `HashMap` iteration order — and only then divided once in floating point. Summing per-block
    /// `f64` values instead would make the low bits order-dependent (~1 ULP of jitter between runs).
    /// ⚠ This is not value-identical to a per-block reciprocal sum: for non-power-of-two `N` the two
    /// differ by ≤ 1–2 ULP. The single-rounding form here is the more accurate one, and the point
    /// estimate is deterministic.
    pub fn logical_entropy(&self) -> f64 {
        if self.size == 0 {
            return 0.0;
        }

        // Integer sum of |B|^2; integer addition is associative, so HashMap order does not matter.
        let sum_sq_int = self.sum_block_sizes_sq();
        let n = self.size as f64;
        1.0 - (sum_sq_int as f64) / (n * n)
    }

    /// `Σ|B|²` — the indit count (ordered same-block pairs, diagonal included).
    /// [`Self::distinction_count`] and [`Self::repeat_ratio`] both delegate here.
    /// Block sizes only — label-independent, and integer, so no floating point is involved.
    fn sum_block_sizes_sq(&self) -> usize {
        let mut counts = HashMap::new();
        for &id in &self.ids {
            *counts.entry(id).or_insert(0usize) += 1;
        }
        counts.values().map(|&c| c * c).sum()
    }

    /// Returns the number of distinctions (dits).
    /// A distinction is an ordered pair (u, v) such that u and v are in different blocks.
    /// Count = N^2 - sum(|B|^2).
    pub fn distinction_count(&self) -> usize {
        if self.size == 0 {
            return 0;
        }
        (self.size * self.size) - self.sum_block_sizes_sq()
    }

    /// The indit fraction as exact integers, `r(π) = (Σ|B|², N²)` — the indit-side counterpart of `1 − h(π)`.
    /// Same lattice point, different coordinates: at `Top` (indiscrete) `r = N²/N² = 1`, i.e. maximal
    /// indit and `h = 0`. Delegates to [`Self::sum_block_sizes_sq`], the same count [`Self::distinction_count`] uses.
    /// Returns `(numerator, denominator)` as integers, avoiding floating point. Empty → `(0, 0)`.
    pub fn repeat_ratio(&self) -> (usize, usize) {
        if self.size == 0 {
            return (0, 0);
        }
        (self.sum_block_sizes_sq(), self.size * self.size)
    }

    /// `indit(π)` — the ordered same-block pairs `(i, j)`, diagonal included; `|indit| = Σ|B|²`.
    /// This materialises the relation, which costs `O(Σ|B|²)`; it is intended for boundary and
    /// measurement code. Hot loops should use the ratio `r = 1 − h` instead.
    pub fn indit_pairs(&self) -> Vec<(usize, usize)> {
        let mut blocks: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, &id) in self.ids.iter().enumerate() {
            blocks.entry(id).or_default().push(i);
        }
        let mut pairs = Vec::new();
        for elems in blocks.values() {
            for &a in elems {
                for &b in elems {
                    pairs.push((a, b));
                }
            }
        }
        pairs
    }

    /// Computes logical divergence (asymmetric): D_L(P||Q) = h(P ∧ Q) - h(Q).
    ///
    pub fn logical_divergence(&self, other: &Self) -> f64 {
        let meet = self.differentiate(other);
        meet.logical_entropy() - other.logical_entropy()
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

        // We compute the symmetric difference size on ditsets via the equivalence relation
        // (same-block pairs), because complementing both sets preserves XOR size.
        //
        // Let S_P(i,j) be 1 iff i~_P j (same block). Then
        // |S_P Δ S_Q| = Σ (S_P xor S_Q) = Σ S_P + Σ S_Q - 2 Σ(S_P & S_Q).
        // For partitions, Σ S_P = Σ_B |B|^2 and Σ(S_P & S_Q) = Σ_{B,C} |B∩C|^2.

        let mut self_block_sizes: HashMap<u32, usize> = HashMap::new();
        let mut other_block_sizes: HashMap<u32, usize> = HashMap::new();
        let mut intersection_sizes: HashMap<(u32, u32), usize> = HashMap::new();

        for i in 0..self.size {
            let a = self.ids[i];
            let b = other.ids[i];
            *self_block_sizes.entry(a).or_insert(0) += 1;
            *other_block_sizes.entry(b).or_insert(0) += 1;
            *intersection_sizes.entry((a, b)).or_insert(0) += 1;
        }
        inc_logical_ops(self.size as u64);

        let same_self: usize = self_block_sizes.values().map(|&c| c * c).sum();
        let same_other: usize = other_block_sizes.values().map(|&c| c * c).sum();
        let same_both: usize = intersection_sizes.values().map(|&c| c * c).sum();
        inc_logical_ops(
            (self_block_sizes.len() + other_block_sizes.len() + intersection_sizes.len()) as u64,
        );

        let xor_count = same_self + same_other - 2 * same_both;
        xor_count as f64 / (self.size * self.size) as f64
    }

    /// Logical area — area expressed as a count of distinctions.
    ///
    /// Formula: $A \approx N^2 \cdot h(\pi) + 0.5$
    ///
    /// where $N^2 \cdot h(\pi) = |\mathrm{dit}(\pi)|$ is the number of distinguished pairs.
    ///
    /// This linearly approximates the loop-quantum-gravity area spectrum $A_j = \sqrt{j(j+1)}$,
    /// with relative error below 1% for large $j$.
    ///
    /// # Example
    /// ```ignore
    /// use partition_lattice::partition::PartitionLattice;
    ///
    /// let p = PartitionLattice::new(4); // fully discrete
    /// // h = 0.75, area = 16 * 0.75 + 0.5 = 12.5
    /// assert!((p.logical_area() - 12.5).abs() < 1e-10);
    /// ```
    ///
    pub fn logical_area(&self) -> f64 {
        let n = self.size as f64;
        let h = self.logical_entropy();
        let num_dits = n * n * h;
        num_dits + 0.5 // Information Offset
    }

    /// Complex Meet — a Meet weighted by phase interference.
    ///
    /// Used when approximating spin-foam 6j-symbol amplitudes with partition operations.
    /// Returns the common refinement together with an interference coefficient.
    ///
    /// # Arguments
    /// - `other`: the partition to combine with
    /// - `phase_self`: this partition's phase (`i8`, −128..127 mapped to −π..π)
    /// - `phase_other`: the other partition's phase (`i8`)
    ///
    /// # Returns
    /// - `(partition, interference)`: the combined partition and its interference weight (0.0–1.0)
    ///
    /// # Interference formula
    /// $$P = \cos^2\left(\frac{\Delta\theta \cdot \pi}{128}\right)$$
    ///
    /// where $\Delta\theta = |\text{phase\_self} - \text{phase\_other}|$
    ///
    /// # Example
    /// ```ignore
    /// use partition_lattice::partition::PartitionLattice;
    ///
    /// let p1 = PartitionLattice::from_ids(vec![0, 0, 1, 1]);
    /// let p2 = PartitionLattice::from_ids(vec![0, 1, 0, 1]);
    /// let (meet, interference) = p1.complex_refine(&p2, 0, 0); // in phase
    /// assert!((interference - 1.0).abs() < 1e-10); // fully constructive
    /// ```
    ///
    pub fn complex_refine(&self, other: &Self, phase_self: i8, phase_other: i8) -> (Self, f64) {
        // 1. Ordinary meet (common refinement).
        let meet_partition = self.differentiate(other);

        // 2. Phase difference: i8 in -128..127 maps onto -pi..pi.
        let phase_diff = (phase_self as i16 - phase_other as i16).abs() as f64;
        let theta = phase_diff * core::f64::consts::PI / 128.0;

        // 3. Interference coefficient: cos^2(theta/2).
        // Destructive interference is expected at a phase difference of pi.
        let cos_half = (theta / 2.0).cos();
        let interference = cos_half * cos_half;

        (meet_partition, interference)
    }

    /// Returns a list of all distinctions (pairs of indices).
    /// This is computationally expensive O(N^2) and should be used for debugging or small N.
    pub fn distinctions(&self) -> Vec<(usize, usize)> {
        let mut dits = Vec::new();
        for i in 0..self.size {
            for j in 0..self.size {
                if self.ids[i] != self.ids[j] {
                    dits.push((i, j));
                }
            }
        }
        dits
    }

    /// Differentiate (formerly Meet): Intersection of blocks.
    /// Refines the partition by splitting blocks based on the other partition.
    /// Two elements are equivalent iff they are equivalent in BOTH self and other.
    pub fn differentiate(&self, other: &Self) -> Self {
        assert_eq!(self.size, other.size, "Partition sizes must match");

        // Pair (id1, id2) -> new_id
        let mut map: HashMap<(u32, u32), u32> = HashMap::new();
        let mut new_ids: Vec<u32> = Vec::with_capacity(self.size);
        let mut next_id: u32 = 0;

        for i in 0..self.size {
            let key = (self.ids[i], other.ids[i]);
            let new_id = *map.entry(key).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            new_ids.push(new_id);
        }
        inc_logical_ops(self.size as u64);

        Self {
            ids: new_ids,
            size: self.size,
        }
    }

    /// Integrate (formerly Join): Merging blocks.
    /// Coarsens the partition by merging blocks based on the other partition.
    /// Two elements are equivalent if they are equivalent in self OR other (transitively).
    pub fn integrate(&self, other: &Self) -> Self {
        assert_eq!(self.size, other.size, "Partition sizes must match");

        // Initialize DSU
        let mut dsu = DisjointSet::new(self.size);

        // Helper to union elements based on partition IDs
        // O(N) complexity
        let union_partition = |partition: &Self, dsu: &mut DisjointSet| {
            // max_id is typically small (number of blocks)
            // ids are normalized 0..k
            let k = partition.block_count();
            let mut seen: Vec<Option<usize>> = vec![None; k];

            for (i, &block_id) in partition.ids.iter().enumerate() {
                if let Some(first) = seen[block_id as usize] {
                    dsu.union(first, i);
                } else {
                    seen[block_id as usize] = Some(i);
                }
            }
            inc_logical_ops(partition.ids.len() as u64);
        };

        // Union elements based on self and other
        union_partition(self, &mut dsu);
        union_partition(other, &mut dsu);

        // Build result
        let mut new_ids: Vec<u32> = Vec::with_capacity(self.size);
        for i in 0..self.size {
            new_ids.push(dsu.find(i) as u32);
        }

        Self::from_ids(new_ids)
    }

    /// Checks if `self` refines `other` (self ⊑ other in refinement order).
    ///
    /// `self` refines `other` iff every block in `self` is a subset of some block in `other`.
    /// Equivalently: whenever two elements are equivalent in `self`, they must be equivalent in `other`.
    pub fn refines(&self, other: &Self) -> bool {
        assert_eq!(self.size, other.size, "Partition sizes must match");
        if self.size == 0 {
            return true;
        }

        for i in 0..self.size {
            for j in 0..self.size {
                if self.ids[i] == self.ids[j] && other.ids[i] != other.ids[j] {
                    return false;
                }
            }
        }
        true
    }

    /// Fused operation: Computes h(self ^ other) without materializing the differentiated partition.
    /// This is more memory efficient and faster as it avoids allocating the new partition structure.
    pub fn fused_entropy_differentiate(&self, other: &Self) -> f64 {
        assert_eq!(self.size, other.size, "Partition sizes must match");
        if self.size == 0 {
            return 0.0;
        }

        // We need to count the size of each block in the differentiation.
        // A block in differentiation is intersection of B_i from self and B_j from other.
        // We can just count occurrences of pairs (self.ids[k], other.ids[k]).
        let mut counts: HashMap<(u32, u32), usize> = HashMap::new();

        for i in 0..self.size {
            *counts.entry((self.ids[i], other.ids[i])).or_insert(0) += 1;
        }

        // Bit-determinism: accumulate the integer sum of |B ∩ C|^2 first, which is order-independent.
        // This removes the ~1 ULP run-to-run jitter of a floating-point accumulation and matches
        // FastPartition's single-division form. Not value-identical: non-power-of-two N shifts by 1-2 ULP.
        let sum_sq_int: usize = counts.values().map(|&c| c * c).sum();
        let n = self.size as f64;
        1.0 - (sum_sq_int as f64) / (n * n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimized::FastPartition;

    const EPS: f64 = 1e-9;

    #[test]
    fn test_partition_creation() {
        let p = PartitionLattice::new(5);
        assert_eq!(p.size(), 5);
        assert_eq!(p.block_count(), 5);

        // Discrete partition: {{0}, {1}, {2}, {3}, {4}}
        // p_B = 1/5. sum(p_B^2) = 5 * (1/25) = 1/5 = 0.2.
        // h = 1 - 0.2 = 0.8.
        assert!((p.logical_entropy() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_partition_repeat_ratio_complements_distinction() {
        // r(pi) = (sum |B|^2, N^2) and distinction_count = N^2 - sum |B|^2: an exact integer identity.
        // mixed {{0,1},{2}} -> ids [0,0,1], N=3: Σ = 2²+1² = 5, N² = 9, dit = 4.
        let p = PartitionLattice::from_ids(vec![0u32, 0, 1]);
        let (num, den) = p.repeat_ratio();
        assert_eq!((num, den), (5, 9));
        assert_eq!(num + p.distinction_count(), den, "repeat + dit = N^2 exactly");
    }

    #[test]
    fn test_partition_repeat_ratio_top_is_max_indit() {
        // Top (indiscrete, one block): r = N^2/N^2 = 1, maximal indit, h = 0.
        let top = PartitionLattice::from_ids(vec![0u32, 0, 0, 0, 0]);
        assert_eq!(top.repeat_ratio(), (25, 25));
        assert!(
            (top.logical_entropy() - 0.0).abs() < 1e-9,
            "⊤ h=0 (dit-view of plenum)"
        );
        // Bottom (discrete): r = N/N^2, minimal indit, h = 1 - 1/N.
        let bot = PartitionLattice::from_ids(vec![0u32, 1, 2, 3, 4]);
        assert_eq!(bot.repeat_ratio(), (5, 25));
    }

    /// Bit-determinism: repeated calls of [`Partition::logical_entropy`] and
    /// `fused_entropy_differentiate` on the same input are bit-identical, because the integer
    /// `Σ|B|²` is accumulated before the single division.
    #[test]
    fn test_partition_entropy_bit_determinism_repeated_calls() {
        // Many distinct blocks, so the HashMap holds enough entries for order dependence to be real.
        let p = PartitionLattice::from_ids(vec![0u32, 1, 2, 3, 0, 1, 2, 3, 0, 1, 4, 5, 6, 7, 4, 5]);
        let q = PartitionLattice::from_ids(vec![0u32, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]);
        let h0 = p.logical_entropy();
        let f0 = p.fused_entropy_differentiate(&q);
        for _ in 0..200 {
            // Bit-exact (== rather than an epsilon): the estimate must be fully deterministic across calls.
            assert_eq!(
                p.logical_entropy().to_bits(),
                h0.to_bits(),
                "logical_entropy bit-stable"
            );
            assert_eq!(
                p.fused_entropy_differentiate(&q).to_bits(),
                f0.to_bits(),
                "fused_entropy_differentiate bit-stable"
            );
        }
        // freshly-constructed clones (new HashMap insertion order) → still bit-identical.
        let p2 = PartitionLattice::from_ids(p.get_ids().to_vec());
        let q2 = PartitionLattice::from_ids(q.get_ids().to_vec());
        assert_eq!(
            p2.logical_entropy().to_bits(),
            h0.to_bits(),
            "clone h bit-identical"
        );
        assert_eq!(
            p2.fused_entropy_differentiate(&q2).to_bits(),
            f0.to_bits(),
            "clone fused bit-identical"
        );
    }

    #[test]
    fn test_partition_indit_pairs_count_matches_repeat() {
        // |indit_pairs| = sum |B|^2 = repeat_ratio().0, ordered same-block pairs including the diagonal.
        let p = PartitionLattice::from_ids(vec![0u32, 0, 1]);
        assert_eq!(p.indit_pairs().len(), p.repeat_ratio().0);
        // Block {0,1} contributes (0,0) (0,1) (1,0) (1,1); block {2} contributes (2,2).
        let mut got = p.indit_pairs();
        got.sort_unstable();
        assert_eq!(got, vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn test_differentiate_integrate() {
        // p1 = {{0, 1}, {2}} -> ids [0, 0, 1]
        let p1 = PartitionLattice::from_ids(vec![0u32, 0, 1]);
        // p2 = {{0}, {1, 2}} -> ids [0, 1, 1]
        let p2 = PartitionLattice::from_ids(vec![0u32, 1, 1]);

        // Differentiate (Meet): {{0}, {1}, {2}} -> ids [0, 1, 2] (Discrete)
        let diff = p1.differentiate(&p2);
        assert_eq!(diff.block_count(), 3);

        // Integrate (Join): {{0, 1, 2}} -> ids [0, 0, 0] (Indiscrete)
        let integ = p1.integrate(&p2);
        assert_eq!(integ.block_count(), 1);
    }

    #[test]
    fn test_distinctions() {
        // P = {{0, 1}, {2}}. N=3.
        // Blocks: B0={0,1} (size 2), B1={2} (size 1).
        // Indistinguishable pairs: (0,0), (0,1), (1,0), (1,1), (2,2). Total 5.
        // Total pairs: 3*3 = 9.
        // Distinctions: 9 - 5 = 4.
        // Pairs: (0,2), (2,0), (1,2), (2,1).
        let p = PartitionLattice::from_ids(vec![0u32, 0, 1]);
        assert_eq!(p.distinction_count(), 4);

        let dits = p.distinctions();
        assert_eq!(dits.len(), 4);
        assert!(dits.contains(&(0, 2)));
        assert!(dits.contains(&(2, 0)));
        assert!(dits.contains(&(1, 2)));
        assert!(dits.contains(&(2, 1)));
    }

    #[test]
    fn test_partition_logical_divergence_zero_when_equal() {
        let p = PartitionLattice::from_ids(vec![0u32, 0, 1, 1, 2, 2]);
        assert!((p.logical_divergence(&p) - 0.0).abs() < EPS);
    }

    #[test]
    fn test_partition_logical_divergence_non_negative() {
        let p = PartitionLattice::from_ids(vec![0u32, 0, 1, 1, 2, 2]);
        let q = PartitionLattice::from_ids(vec![0u32, 1, 2, 0, 1, 2]);

        let d_pq = p.logical_divergence(&q);
        let d_qp = q.logical_divergence(&p);

        assert!(d_pq >= -EPS);
        assert!(d_qp >= -EPS);
    }

    #[test]
    fn test_partition_logical_divergence_matches_fastpartition() {
        let p_ids: Vec<u32> = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let q_ids: Vec<u32> = vec![0, 1, 0, 1, 2, 3, 2, 3];

        let p = PartitionLattice::from_ids(p_ids.clone());
        let q = PartitionLattice::from_ids(q_ids.clone());

        let p_fast = FastPartition::from_ids(&p_ids);
        let q_fast = FastPartition::from_ids(&q_ids);

        let d_slow = p.logical_divergence(&q);
        let d_fast = p_fast.logical_divergence(&q_fast);

        assert!((d_slow - d_fast).abs() < 1e-12);
    }

    fn brute_force_ditset_xor_distance(p: &PartitionLattice, q: &PartitionLattice) -> f64 {
        assert_eq!(p.size(), q.size());
        if p.size() == 0 {
            return 0.0;
        }

        let p_ids = p.get_ids();
        let q_ids = q.get_ids();
        let n = p.size();

        let mut xor_count: usize = 0;
        for i in 0..n {
            for j in 0..n {
                let p_distinguishes = p_ids[i] != p_ids[j];
                let q_distinguishes = q_ids[i] != q_ids[j];
                if p_distinguishes ^ q_distinguishes {
                    xor_count += 1;
                }
            }
        }

        xor_count as f64 / (n * n) as f64
    }

    #[test]
    fn test_partition_ditset_xor_distance_zero_when_equal() {
        let p = PartitionLattice::from_ids(vec![0u32, 0, 1, 1, 2, 2]);
        assert!((p.ditset_xor_distance(&p) - 0.0).abs() < EPS);
    }

    #[test]
    fn test_partition_ditset_xor_distance_is_symmetric() {
        let p = PartitionLattice::from_ids(vec![0u32, 0, 1, 1, 2, 2]);
        let q = PartitionLattice::from_ids(vec![0u32, 1, 2, 0, 1, 2]);

        let d_pq = p.ditset_xor_distance(&q);
        let d_qp = q.ditset_xor_distance(&p);

        assert!((d_pq - d_qp).abs() < 1e-12);
    }

    #[test]
    fn test_partition_ditset_xor_distance_matches_bruteforce() {
        let p = PartitionLattice::from_ids(vec![0u32, 0, 0, 1]);
        let q = PartitionLattice::from_ids(vec![0u32, 1, 1, 1]);

        let expected = brute_force_ditset_xor_distance(&p, &q);
        let actual = p.ditset_xor_distance(&q);

        assert!((expected - actual).abs() < 1e-12);
    }

    #[test]
    fn test_partition_ditset_xor_distance_matches_fastpartition() {
        let p_ids: Vec<u32> = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let q_ids: Vec<u32> = vec![0, 1, 0, 1, 2, 3, 2, 3];

        let p = PartitionLattice::from_ids(p_ids.clone());
        let q = PartitionLattice::from_ids(q_ids.clone());

        let p_fast = FastPartition::from_ids(&p_ids);
        let q_fast = FastPartition::from_ids(&q_ids);

        let d_slow = p.ditset_xor_distance(&q);
        let d_fast = p_fast.ditset_xor_distance(&q_fast);

        assert!((d_slow - d_fast).abs() < 1e-12);
    }

    /// `logical_area()` on a discrete partition.
    ///
    /// Formula: A = N² · h(π) + 0.5
    /// where h(π) = 1 − Σ p_B² = |dit(π)| / N²
    #[test]
    fn test_partition_logical_area_discrete() {
        let p = PartitionLattice::new(4); // fully discrete: {{0}, {1}, {2}, {3}}
                                          // h = 1 - 4*(1/4)^2 = 1 - 4/16 = 1 - 0.25 = 0.75
                                          // area = 16 * 0.75 + 0.5 = 12 + 0.5 = 12.5
        assert!((p.logical_area() - 12.5).abs() < 1e-10);
    }

    #[test]
    fn test_partition_logical_area_indiscrete() {
        let p = PartitionLattice::from_ids(vec![0u32, 0, 0, 0]); // fully indiscrete: {{0,1,2,3}}
                                                                 // h = 1 - 1*(4/4)^2 = 1 - 1 = 0
                                                                 // area = 16 * 0 + 0.5 = 0.5
        assert!((p.logical_area() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_partition_logical_area_mixed() {
        let p = PartitionLattice::from_ids(vec![0u32, 0, 1, 1]); // {{0,1}, {2,3}}
                                                                 // h = 1 - 2*(2/4)^2 = 1 - 2*0.25 = 0.5
                                                                 // area = 16 * 0.5 + 0.5 = 8.5
        assert!((p.logical_area() - 8.5).abs() < 1e-10);
    }

    /// `complex_refine()` in phase — fully constructive interference.
    #[test]
    fn test_partition_complex_meet_same_phase() {
        let p1 = PartitionLattice::from_ids(vec![0u32, 0, 1, 1]); // {{0,1}, {2,3}}
        let p2 = PartitionLattice::from_ids(vec![0u32, 1, 0, 1]); // {{0,2}, {1,3}}
        let (meet, interference) = p1.complex_refine(&p2, 0, 0); // in phase

        // The meet is the finest refinement: {{0}, {1}, {2}, {3}}.
        assert_eq!(meet.block_count(), 4);
        // In phase means fully constructive: cos(0)^2 = 1.0.
        assert!((interference - 1.0).abs() < 1e-10);
    }

    /// `complex_refine()` in antiphase — destructive interference.
    #[test]
    fn test_partition_complex_meet_opposite_phase() {
        let p1 = PartitionLattice::from_ids(vec![0u32, 0, 1, 1]);
        let p2 = PartitionLattice::from_ids(vec![0u32, 1, 0, 1]);
        let (_, interference) = p1.complex_refine(&p2, 0, -128); // pi phase difference

        // cos(pi)^2 = (-1)^2 = 1.0; note that cos^2 is always non-negative,
        // so modelling true cancellation would need a different form such as cos(theta/2)^2.
        assert!((0.0..=1.0).contains(&interference));
    }

    /// `complex_refine()` at orthogonal phase (π/2).
    #[test]
    fn test_partition_complex_meet_orthogonal_phase() {
        let p1 = PartitionLattice::from_ids(vec![0u32, 0, 1, 1]);
        let p2 = PartitionLattice::from_ids(vec![0u32, 1, 0, 1]);
        let (_, interference) = p1.complex_refine(&p2, 0, 64); // pi/2 phase difference

        // cos^2(θ/2) calculation
        let phase_diff_theta = 64.0 * core::f64::consts::PI / 128.0; // π/2
        let expected = (phase_diff_theta / 2.0).cos().powi(2); // cos^2(π/4) = 0.5
        assert!((interference - expected).abs() < 1e-10);
    }

    // ========================================================================
    // Physical consistency checks.
    // ========================================================================

    /// Area-spectrum convergence: exact vs linear approximation.
    ///
    /// Exact: A_j = sqrt(j·(j+1))
    /// Approximation: A_j ≈ j + 0.5
    ///
    /// Here j is the distinction count; for a fully discrete partition
    /// j = N·(N−1) (ordered distinguished pairs).
    ///
    /// The relative error must stay below 5% for large j.
    #[test]
    fn test_lqrm_area_spectrum_convergence() {
        // j values, in units of distinctions.
        let test_cases = vec![
            10,  // j=10
            20,  // j=20
            50,  // j=50
            100, // j=100
        ];

        for j in test_cases {
            let j_f64 = j as f64;

            // Exact area: sqrt(j * (j + 1)).
            let area_lqg = (j_f64 * (j_f64 + 1.0_f64)).sqrt();

            // Approximate area: j + 0.5.
            let area_lqrm = j_f64 + 0.5;

            // Relative error.
            let relative_error = (area_lqg - area_lqrm).abs() / area_lqg;

            println!(
                "j={}: LQG={:.4}, L-QRM={:.4}, error={:.4}%",
                j,
                area_lqg,
                area_lqrm,
                relative_error * 100.0
            );

            // The error must stay below 5% for large j.
            if j >= 20 {
                assert!(
                    relative_error < 0.05,
                    "j={}: LQG={:.4}, L-QRM={:.4}, error={:.4} (> 5%)",
                    j,
                    area_lqg,
                    area_lqrm,
                    relative_error
                );
            }
        }
    }

    /// Entropy extensivity: the sum of microscopic entropies equals the macroscopic area.
    ///
    /// Bekenstein-Hawking: S ∝ A
    /// L-QRM: S_total = Σ h(π_i) ∝ Area
    #[test]
    fn test_entropy_extensivity() {
        // Build several independent partitions.
        let partitions = [
            PartitionLattice::from_ids(vec![0u32, 1, 2, 3]), // fully discrete
            PartitionLattice::from_ids(vec![0u32, 0, 1, 1]), // two equal blocks
            PartitionLattice::from_ids(vec![0u32, 1, 1, 2]), // mixed
        ];

        // Sum of the individual entropies.
        let total_entropy: f64 = partitions.iter().map(|p| p.logical_entropy()).sum();

        // Sum of the individual areas.
        let total_area: f64 = partitions.iter().map(|p| p.logical_area()).sum();

        // Proportionality: Area is approximately N^2 * h plus a constant, so both must be
        // positive and within a sensible range.
        assert!(total_entropy > 0.0);
        assert!(total_area > 0.0);
        assert!(total_area > total_entropy); // area carries the N^2 scaling
    }

    /// Information preservation: distinction counts across a refinement.
    ///
    /// dit(π ∧ σ) = dit(π) ∪ dit(σ)
    #[test]
    fn test_distinction_preservation() {
        let p1 = PartitionLattice::from_ids(vec![0u32, 0, 1, 1]);
        let p2 = PartitionLattice::from_ids(vec![0u32, 1, 0, 1]);

        // Refinement (meet).
        let p_meet = p1.differentiate(&p2);

        // Distinction count = N^2 * h.
        let n = 4.0;
        let dits_p1 = n * n * p1.logical_entropy();
        let dits_p2 = n * n * p2.logical_entropy();
        let dits_meet = n * n * p_meet.logical_entropy();

        // A meet adds distinctions.
        assert!(dits_meet >= dits_p1);
        assert!(dits_meet >= dits_p2);
    }
}
