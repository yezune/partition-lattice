//! Unified Partition: single-point abstraction over PartitionLattice and FastPartition
//!
//! Provides:
//! - `PartitionOps` trait: common interface for all partition types
//! - `Partition` enum: static-dispatch wrapper that auto-selects optimal backend
//! - `From` conversions between PartitionLattice ↔ FastPartition

use crate::join::shallow_join_double_buffer;
#[cfg(feature = "std")]
use crate::optimized::FastPartition;
use crate::partition::PartitionLattice;
#[cfg(feature = "std")]
use crate::sparse::SparsePartition;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::FloatExt;
#[allow(unused_imports)]
use alloc::borrow::ToOwned;
#[allow(unused_imports)]
use alloc::boxed::Box;
#[allow(unused_imports)]
use alloc::string::{String, ToString};
#[allow(unused_imports)]
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

// ── Phase 0: Mutual conversions ─────────────────────────────────

#[cfg(feature = "std")]
impl From<&PartitionLattice> for FastPartition {
    fn from(pl: &PartitionLattice) -> Self {
        FastPartition::from_ids(pl.get_ids())
    }
}

#[cfg(feature = "std")]
impl From<PartitionLattice> for FastPartition {
    fn from(pl: PartitionLattice) -> Self {
        FastPartition::from_ids(pl.get_ids())
    }
}

#[cfg(feature = "std")]
impl From<&FastPartition> for PartitionLattice {
    fn from(fp: &FastPartition) -> Self {
        PartitionLattice::from_ids(fp.get_ids().to_vec())
    }
}

#[cfg(feature = "std")]
impl From<FastPartition> for PartitionLattice {
    fn from(fp: FastPartition) -> Self {
        PartitionLattice::from_ids(fp.get_ids().to_vec())
    }
}

// ── Phase 1: PartitionOps trait ─────────────────────────────────

/// Common interface for all partition types.
///
/// Both `PartitionLattice` and `FastPartition` implement this trait,
/// enabling generic code that works with either backend.
pub trait PartitionOps: Clone + core::fmt::Debug + Send + Sync {
    /// Number of elements in the universe.
    fn size(&self) -> usize;

    /// Number of blocks in the partition.
    fn block_count(&self) -> usize;

    /// Block IDs as a slice: element i belongs to block `get_ids()[i]`.
    fn get_ids(&self) -> &[u32];

    /// Logical entropy: h(π) = 1 − Σ p_B²
    fn logical_entropy(&self) -> f64;

    /// Number of distinctions: |dit(π)| = N² − Σ|B|²
    fn dit_count(&self) -> usize;

    /// Logical divergence: D_L(self‖other) = h(self ∧ other) − h(other)
    fn logical_divergence(&self, other: &Self) -> f64;

    /// Ditset XOR distance: d⊕(P,Q) = |dit(P) Δ dit(Q)| / N²
    fn ditset_xor_distance(&self, other: &Self) -> f64;

    /// Meet (∧): common refinement — two elements equivalent iff equivalent in BOTH.
    fn differentiate(&self, other: &Self) -> Self;

    /// Join (∨): common coarsening — two elements equivalent if equivalent in EITHER.
    ///
    /// Default strategy: ShallowJoin d=2 (double-buffer label propagation).
    /// For exact transitive closure, use [`Partition::integrate_exact`].
    fn integrate(&self, other: &Self) -> Self;

    /// Exact Join (∨) via Union-Find with full transitive closure.
    ///
    /// Use this when mathematical exactness is required (witness verification,
    /// lattice proofs, unit tests asserting exact block structure).
    /// For learning loops, prefer [`Partition::integrate`] (ShallowJoin, 30~484x faster).
    fn integrate_exact(&self, other: &Self) -> Self {
        // Default: delegate to integrate() — backends override for exact semantics
        self.integrate(other)
    }

    /// Join with configurable strategy.
    fn integrate_with_strategy(&self, other: &Self, strategy: CoarsenStrategy) -> Self {
        match strategy {
            CoarsenStrategy::Exact => self.integrate_exact(other),
            CoarsenStrategy::Shallow(_) | CoarsenStrategy::Auto => self.integrate(other),
        }
    }

    /// Refinement check: self ⊑ other
    fn refines(&self, other: &Self) -> bool;

    /// Fused entropy of meet: h(self ∧ other) without materializing the meet.
    fn fused_entropy_differentiate(&self, other: &Self) -> f64;

    /// L-QRM logical area: A ≈ N² · h(π) + 0.5
    fn logical_area(&self) -> f64;

    /// Parallel logical entropy (default: falls back to sequential).
    fn logical_entropy_parallel(&self) -> f64 {
        self.logical_entropy()
    }

    /// Parallel meet/differentiate (default: falls back to sequential).
    fn differentiate_parallel(&self, other: &Self) -> Self {
        self.differentiate(other)
    }
}

impl PartitionOps for PartitionLattice {
    fn size(&self) -> usize {
        self.size()
    }
    fn block_count(&self) -> usize {
        self.block_count()
    }
    fn get_ids(&self) -> &[u32] {
        self.get_ids()
    }
    fn logical_entropy(&self) -> f64 {
        self.logical_entropy()
    }
    fn dit_count(&self) -> usize {
        self.dit_count()
    }
    fn logical_divergence(&self, other: &Self) -> f64 {
        self.logical_divergence(other)
    }
    fn ditset_xor_distance(&self, other: &Self) -> f64 {
        self.ditset_xor_distance(other)
    }
    fn differentiate(&self, other: &Self) -> Self {
        self.differentiate(other)
    }
    fn integrate(&self, other: &Self) -> Self {
        self.integrate(other)
    }
    fn integrate_exact(&self, other: &Self) -> Self {
        self.integrate(other)
    }
    fn refines(&self, other: &Self) -> bool {
        self.refines(other)
    }
    fn fused_entropy_differentiate(&self, other: &Self) -> f64 {
        self.fused_entropy_differentiate(other)
    }
    fn logical_area(&self) -> f64 {
        self.logical_area()
    }
    // logical_entropy_parallel / differentiate_parallel: use default (sequential fallback)
}

#[cfg(feature = "std")]
impl PartitionOps for FastPartition {
    fn size(&self) -> usize {
        self.size()
    }
    fn block_count(&self) -> usize {
        self.block_count()
    }
    fn get_ids(&self) -> &[u32] {
        self.get_ids()
    }
    fn logical_entropy(&self) -> f64 {
        self.logical_entropy()
    }
    fn dit_count(&self) -> usize {
        self.dit_count()
    }
    fn logical_divergence(&self, other: &Self) -> f64 {
        self.logical_divergence(other)
    }
    fn ditset_xor_distance(&self, other: &Self) -> f64 {
        self.ditset_xor_distance(other)
    }
    fn differentiate(&self, other: &Self) -> Self {
        self.differentiate(other)
    }
    fn integrate(&self, other: &Self) -> Self {
        self.integrate(other)
    }
    fn integrate_exact(&self, other: &Self) -> Self {
        self.integrate(other)
    }
    fn refines(&self, other: &Self) -> bool {
        self.refines(other)
    }
    fn fused_entropy_differentiate(&self, other: &Self) -> f64 {
        self.fused_entropy_differentiate(other)
    }
    fn logical_area(&self) -> f64 {
        self.logical_area()
    }
    fn logical_entropy_parallel(&self) -> f64 {
        self.logical_entropy_parallel()
    }
    fn differentiate_parallel(&self, other: &Self) -> Self {
        self.differentiate_parallel(other)
    }
}

#[cfg(feature = "std")]
impl PartitionOps for SparsePartition {
    fn size(&self) -> usize {
        self.size()
    }
    fn block_count(&self) -> usize {
        self.block_count()
    }
    fn get_ids(&self) -> &[u32] {
        self.get_ids()
    }
    fn logical_entropy(&self) -> f64 {
        self.logical_entropy()
    }
    fn dit_count(&self) -> usize {
        self.dit_count()
    }
    fn logical_divergence(&self, other: &Self) -> f64 {
        self.logical_divergence(other)
    }
    fn ditset_xor_distance(&self, other: &Self) -> f64 {
        self.ditset_xor_distance(other)
    }
    fn differentiate(&self, other: &Self) -> Self {
        self.differentiate(other)
    }
    fn integrate(&self, other: &Self) -> Self {
        self.integrate(other)
    }
    fn integrate_exact(&self, other: &Self) -> Self {
        self.integrate(other)
    }
    fn refines(&self, other: &Self) -> bool {
        self.refines(other)
    }
    fn fused_entropy_differentiate(&self, other: &Self) -> f64 {
        self.fused_entropy_differentiate(other)
    }
    fn logical_area(&self) -> f64 {
        self.logical_area()
    }
    // logical_entropy_parallel / differentiate_parallel: use default (sequential fallback)
    // SparsePartition entropy is already O(1) from precomputed sum_block_sizes_sq
}

// ── Sparse ↔ Lattice/Fast conversions ──────────────────────────

#[cfg(feature = "std")]
impl From<&SparsePartition> for PartitionLattice {
    fn from(sp: &SparsePartition) -> Self {
        PartitionLattice::from_ids(sp.get_ids().to_vec())
    }
}

#[cfg(feature = "std")]
impl From<&SparsePartition> for FastPartition {
    fn from(sp: &SparsePartition) -> Self {
        FastPartition::from_ids(sp.get_ids())
    }
}

#[cfg(feature = "std")]
impl From<&PartitionLattice> for SparsePartition {
    fn from(pl: &PartitionLattice) -> Self {
        SparsePartition::from_ids(pl.get_ids())
    }
}

#[cfg(feature = "std")]
impl From<&FastPartition> for SparsePartition {
    fn from(fp: &FastPartition) -> Self {
        SparsePartition::from_ids(fp.get_ids())
    }
}

// ── Phase 2: Partition enum ─────────────────────────────────────

/// Size threshold: partitions larger than this auto-select FastPartition.
const AUTO_SELECT_THRESHOLD: usize = 128;

/// Above this many bytes of block bitvectors, `from_ids` hands the input to
/// `PartitionLattice` instead of `FastPartition`.
///
/// `PartitionLattice` stores only `ids` (`n * 4` bytes, independent of the block
/// count), so it bounds a single operand's footprint where `FastPartition` grows as
/// `K * ceil(n/64) * 8`. On a measured 276-config grid across two architectures this
/// took the worst peak RSS from 6.79 GB to 2.52 GB on one machine and 23.52 GB to
/// 2.55 GB on the other, and took the configurations the OOM killer stopped from six
/// to none on each. Median time regret moved by about 4% — up on one architecture and
/// down on the other. It is a bound on memory, not a speed-up.
///
/// The value is not fitted to that grid: 256 MiB and 1 GiB produce identical time,
/// memory and survival numbers, so it sits mid-plateau rather than on an edge.
/// 1 GiB is chosen from that plateau because it stays strictly above the Join
/// sparse-coarsen threshold, which keeps the reasoning about their interaction simple.
#[cfg(feature = "std")]
const LATTICE_FALLBACK_BUDGET: usize = 1024 * 1024 * 1024;

// The dispatch thresholds must stay ordered: a Lattice fallback below the sparse
// guards would fire before them and change which guard owns a decision.
#[cfg(feature = "std")]
const _: () = assert!(LATTICE_FALLBACK_BUDGET > 256 * 1024 * 1024);


/// A reasonable factorization block size for a universe of `n` elements.
///
/// Rounds `sqrt(N)` down to a power of two, with a floor of `DEFAULT_FACTOR` (256):
/// `N = 65_536` gives 256 and `N = 262_144` gives 512.
///
/// `from_ids` used this when it converted to `FactorizedFastPartition`
/// automatically. That conversion was withdrawn — see the note in
/// [`Partition::from_ids`] — so the choice now belongs to whoever calls
/// [`Partition::from_ids_factorized`], and this is published so that a caller who
/// has no particular factor in mind can ask for the one the crate would have picked.
///
/// # Examples
/// ```
/// use partition_lattice::pick_factor_for_universe;
/// assert_eq!(pick_factor_for_universe(65_536), 256);
/// assert_eq!(pick_factor_for_universe(262_144), 512);
/// ```
#[cfg(feature = "std")]
pub fn pick_factor_for_universe(n: usize) -> u32 {
    let sqrt = (n as f64).sqrt() as usize;
    let mut f = 1u32;
    while (f as usize) * 2 <= sqrt {
        f *= 2;
    }
    f.max(crate::factorized::DEFAULT_FACTOR)
}

/// Join strategy for integrate operations.
///
/// Controls the trade-off between exactness and speed.
/// Default: `Shallow(2)` with double-buffer symmetric propagation.
///
/// # Evidence
/// - ShallowJoin d=2: 30~484x faster than Exact, h(π) 8% higher (prevents over-coarsening)
/// - CPU double-buffer ensures symmetric propagation (in-place causes h collapse)
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CoarsenStrategy {
    /// ShallowJoin with fixed depth d (double-buffer symmetric propagation).
    /// d=2 is the recommended default.
    Shallow(u32),
    /// Exact Join via Union-Find (full transitive closure).
    /// Required for witness verification and lattice proofs.
    Exact,
    /// Auto-select based on context (currently defaults to Shallow(2)).
    Auto,
}

impl Default for CoarsenStrategy {
    fn default() -> Self {
        CoarsenStrategy::Shallow(2)
    }
}

/// PCS-A1/A5: Unified dispatch advice for partition operations.
///
/// Computed in O(1) from (N, K₁, K₂) metadata only.
/// See the module documentation for the dispatch analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchAdvice {
    /// R₀/R₁: Normal block-cross path. K₁·K₂ is manageable.
    Fast,
    /// R₂: Route through SparsePartition to avoid memory explosion.
    /// For Meet: K₁×K₂ result blocks would exceed memory limit.
    /// For Join: max(K₁,K₂) result blocks would exceed memory limit.
    Sparse,
    /// R₃: Use streaming (non-materializing) path for metrics only.
    /// Result: exact h/D_L/d_⊕, but no materialized partition.
    Streaming,
}

/// PCS-A1: Partition operation type for dispatch diagnosis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionOp {
    /// Meet (∧) — differentiate: result has up to K₁×K₂ blocks
    Meet,
    /// Join (∨) — integrate: result has ≤ max(K₁,K₂) blocks
    Join,
    /// Metric-only: h, D_L, d_⊕, I — can use streaming path
    Metric,
}

/// Default chunk size for streaming operations.
#[allow(dead_code)]
const DEFAULT_CHUNK_SIZE: usize = 4096;

/// Memory threshold for adaptive streaming dispatch.
/// When estimated meet memory exceeds this, streaming path is used.
/// Aligned with FastPartition's BLOCKS_MEMORY_LIMIT in optimized.rs.
const STREAMING_DISPATCH_THRESHOLD: usize = 8 * 1024 * 1024;

// Ordering, continued from `LATTICE_FALLBACK_BUDGET`: the metric/meet streaming
// threshold is the lowest of the three, so it keeps first claim on the decisions
// it already owns.
#[cfg(feature = "std")]
const _: () = assert!(256 * 1024 * 1024 > STREAMING_DISPATCH_THRESHOLD);

/// Unified partition type with static dispatch.
///
/// Wraps `PartitionLattice` (small/simple) and `FastPartition` (large/optimized)
/// behind a single enum. All operations delegate to the inner type without
/// vtable overhead.
///
/// # Factory
/// ```rust
/// use partition_lattice::unified::Partition;
///
/// // Auto-selects Lattice (small)
/// let p = Partition::from_ids(vec![0, 0, 1, 1]);
///
/// // Force a specific backend
/// let fast = Partition::from_ids_fast(vec![0, 0, 1, 1]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Partition {
    /// Simple vector-based representation (optimal for N < 128).
    /// The ONLY backend in the `no_std` profile (Lattice-only) — every other
    /// variant rests on `optimized`/`BitVec<u64>` which needs 64-bit atomics
    /// unavailable on 32-bit RISC-V, so all are `std`-gated.
    Lattice(PartitionLattice),
    /// Bitvector-based representation with SIMD/parallel support (optimal for N ≥ 128)
    #[cfg(feature = "std")]
    Fast(FastPartition),
    /// Roaring Bitmap-based representation (optimal for large N with many blocks / sparse structure)
    #[cfg(feature = "std")]
    Sparse(SparsePartition),
    /// Cyclic-pattern representation (40 bytes regardless of |U|) — for
    /// ngram crystals where `ids[i] = pattern[i % cycle_len]`,
    /// `cycle_len ≤ 8`.
    /// Materializes lazily to dense for `get_ids()` and other compound ops.
    #[cfg(feature = "std")]
    Cyclic(crate::cyclic::CyclicPartition),
    /// Wide-cyclic pattern representation (~72 bytes regardless of |U|) —
    /// extends `Cyclic` to `cycle_len ∈ [9, MAX_CYCLE_WIDE = 16]`. Layer A
    /// substrate — `Cyclic`
    /// is always preferred for L ≤ 8 (narrow form is more compact); this
    /// variant only carries L ∈ [9, 16] patterns. Materializes lazily to
    /// dense for `get_ids()` and other compound ops.
    #[cfg(feature = "std")]
    WideCyclic(crate::cyclic_wide::WideCyclicPartition),
    /// 2-tier factorized representation of an arbitrary partition's
    /// element-to-block mapping (Layer B-β-true substrate,
    /// factorized substrate). Stores
    /// `coarse_block[i / F]` + per-coarse-group dominant block + sparse
    /// `fine_overrides` map for `block_of(i) ≠ dominant` exceptions.
    /// Memory ~N/F + override_count vs dense N — substrate-compatible
    /// with Fast fallback ngram crystals where block distribution within
    /// each coarse window is dominant-heavy. Materializes lazily to dense
    /// `FastPartition` for compound ops (Meet/Join/etc.).
    #[cfg(feature = "std")]
    FactorizedFast(crate::factorized::FactorizedFastPartition),
}

impl Partition {
    /// Create a discrete partition: each element in its own block.
    pub fn discrete(size: usize) -> Self {
        #[cfg(feature = "std")]
        if size >= AUTO_SELECT_THRESHOLD {
            return Partition::Fast(FastPartition::discrete(size));
        }
        // no_std profile is Lattice-only (same partition, no large-N optimization).
        Partition::Lattice(PartitionLattice::new(size))
    }

    /// Create from block IDs with automatic backend selection.
    ///
    /// Cyclic wire-up: if ids form a cyclic pattern with
    /// `cycle_len ≤ MAX_CYCLE`, route to the `Cyclic` variant (40-byte
    /// compact representation). Otherwise fall back to the size-based
    /// Lattice / Fast dispatch.
    ///
    /// Routing chain (wide-cyclic residual + bounded Fast):
    /// 1. `Cyclic` (L ≤ 8) — narrowest compact form first.
    /// 2. `WideCyclic` (L ∈ [9, 16]) — extended cycle range.
    /// 3. `Lattice` when `Fast`'s block bitvectors would exceed
    ///    [`LATTICE_FALLBACK_BUDGET`], since `PartitionLattice`'s footprint does not
    ///    grow with the block count. This bounds a single operand's memory.
    /// 4. `Fast` fallback for non-cyclic ids (small universe or block count within budget).
    ///
    /// `FactorizedFast` is **not** selected automatically; the conversion was withdrawn
    /// because it cost time without saving memory against step 3. It remains available
    /// through [`Partition::from_ids_factorized`].
    ///
    /// Narrow form is always preferred when both detectors would succeed —
    /// `CyclicPartition::try_from_ids` selects the *shortest* cycle, so
    /// any L ≤ 8 input lands in `Cyclic` and never reaches the wide step.
    pub fn from_ids(ids: Vec<u32>) -> Self {
        // Large-N backend routing (Cyclic/WideCyclic/Lattice/Fast) is
        // `std`-only. The `no_std` profile is Lattice-only — same partition,
        // just without the large-N memory optimization.
        #[cfg(feature = "std")]
        if ids.len() >= AUTO_SELECT_THRESHOLD {
            // Try cyclic detection only at large universe sizes — the
            // memory savings dominate there.
            if let Some(c) = crate::cyclic::CyclicPartition::try_from_ids(&ids) {
                return Partition::Cyclic(c);
            }
            // Layer A: extend to L ∈ [9, 16] via WideCyclicPartition.
            if let Some(w) = crate::cyclic_wide::WideCyclicPartition::try_from_ids(&ids) {
                return Partition::WideCyclic(w);
            }
            // Automatic conversion to FactorizedFast used to sit here, choosing that
            // representation whenever it looked smaller than `n * 4` bytes. It was
            // withdrawn once `from_ids` gained a Lattice fallback to compare against:
            // measured on a 276-config grid across two architectures, factorized beat
            // Lattice on both time and memory in 0 of 200 admissible configs on one
            // machine and 3 of 200 on the other, was never the fastest backend in any
            // admissible config, and where it actually fired — block-dominant inputs — cost
            // time without saving memory.
            //
            // Its comparison was also against the wrong thing: `n * 4` is
            // `PartitionLattice`'s footprint, not `FastPartition`'s, which also holds
            // one bitvector per block. The alternative it was really competing with
            // is now reachable and wins.
            //
            // `Partition::FactorizedFast` and `from_ids_factorized` remain: this
            // withdraws a default, not a capability.
            // Last resort before Fast: if Fast's block bitvectors would not fit the
            // budget, hand the input to Lattice, whose footprint does not grow with
            // the block count.
            //
            // Position matters. This sits immediately before the Fast fallback and
            // after every compressed backend, so it can only ever replace Fast.
            // Placed any earlier it would take inputs away from Cyclic, WideCyclic
            // or FactorizedFast — all of which are far more compact than Lattice —
            // and a change meant to bound memory would raise it instead.
            if Self::exceeds_bitvec_budget(&ids) {
                return Partition::Lattice(PartitionLattice::from_ids(ids));
            }
            return Partition::Fast(FastPartition::from_ids(&ids));
        }
        Partition::Lattice(PartitionLattice::from_ids(ids))
    }

    /// Bytes `FastPartition` would spend on its block bitvectors: one bitvector of
    /// `n` bits per *distinct* block (`optimized.rs` allocates `0..num_blocks` after
    /// renumbering). This is the term `from_ids` must weigh, and it is the term the
    /// earlier guard at [`Self::from_ids`] omitted — it counted only
    /// `element_to_block` (`n * 4`), understating the real footprint by three orders
    /// of magnitude at large block counts.
    #[cfg(feature = "std")]
    fn fast_bitvec_bytes(distinct_blocks: usize, n: usize) -> usize {
        distinct_blocks
            .saturating_mul(n.div_ceil(64))
            .saturating_mul(8)
    }

    /// Does the input need more bitvector memory than `LATTICE_FALLBACK_BUDGET`?
    ///
    /// Counts distinct ids, **not** `max(id) + 1`: `FastPartition` renumbers before
    /// allocating, so a vector labelled `[0, 1000, 2000]` is three blocks, not 2001.
    /// Using the maximum overstated the footprint by up to 990x on label-sparse input
    /// and would have demoted partitions that cost nothing to represent.
    ///
    /// Stops counting once the budget is already exceeded, so the set never grows
    /// past the threshold and a decision on a huge-K input does not itself allocate.
    #[cfg(feature = "std")]
    fn exceeds_bitvec_budget(ids: &[u32]) -> bool {
        let n = ids.len();
        // One block's worth of bitvector, from the same formula the budget is about.
        let per_block = Self::fast_bitvec_bytes(1, n);
        if per_block == 0 {
            return false;
        }
        let max_blocks_in_budget = LATTICE_FALLBACK_BUDGET / per_block;
        let mut seen = crate::HashSet::with_capacity(64);
        for &id in ids {
            seen.insert(id);
            if seen.len() > max_blocks_in_budget {
                return true;
            }
        }
        false
    }

    /// Create from block IDs, always using FastPartition backend.
    /// Bypasses cyclic auto-detection — use when caller specifically needs
    /// the dense representation.
    #[cfg(feature = "std")]
    pub fn from_ids_fast(ids: Vec<u32>) -> Self {
        Partition::Fast(FastPartition::from_ids(&ids))
    }

    /// Create from block IDs, always using CyclicPartition backend.
    /// Panics if `ids` is not cyclic with `cycle_len ≤ MAX_CYCLE`.
    #[cfg(feature = "std")]
    pub fn from_ids_cyclic(ids: Vec<u32>) -> Self {
        let c = crate::cyclic::CyclicPartition::try_from_ids(&ids)
            .expect("Partition::from_ids_cyclic: input is not a cyclic pattern");
        Partition::Cyclic(c)
    }

    /// Create from block IDs, always using WideCyclicPartition backend.
    /// Panics if `ids` is not cyclic with `cycle_len ≤ MAX_CYCLE_WIDE`.
    #[cfg(feature = "std")]
    pub fn from_ids_wide_cyclic(ids: Vec<u32>) -> Self {
        let c = crate::cyclic_wide::WideCyclicPartition::try_from_ids(&ids)
            .expect("Partition::from_ids_wide_cyclic: input is not a wide-cyclic pattern");
        Partition::WideCyclic(c)
    }

    /// Create from block IDs, always using PartitionLattice backend.
    pub fn from_ids_lattice(ids: Vec<u32>) -> Self {
        Partition::Lattice(PartitionLattice::from_ids(ids))
    }

    /// Create from block IDs, using `FactorizedFastPartition` backend
    /// (factorized substrate).
    ///
    /// Bit-exact arithmetic equivalent of `from_ids_fast` — every
    /// `partition.get_block(i)` matches `FastPartition::from_ids(&ids).get_block(i)`.
    /// Use when the `element_to_block` vector dominates memory cost
    /// (e.g., wp2 Fast fallback ngram crystals at universe ≥ 65 536).
    #[cfg(feature = "std")]
    pub fn from_ids_factorized(ids: Vec<u32>, factor: u32) -> Self {
        Partition::FactorizedFast(
            crate::factorized::FactorizedFastPartition::from_ids_factorized(&ids, factor),
        )
    }

    /// Create from block IDs, always using SparsePartition backend.
    #[cfg(feature = "std")]
    pub fn from_ids_sparse(ids: Vec<u32>) -> Self {
        Partition::Sparse(SparsePartition::from_ids(&ids))
    }

    /// Indiscrete partition (top ⊤): all elements in one block.
    pub fn indiscrete(size: usize) -> Self {
        Partition::from_ids(vec![0u32; size])
    }

    /// Create from explicit blocks (each inner Vec is one block).
    pub fn from_blocks(size: usize, blocks: Vec<Vec<usize>>) -> Self {
        let mut ids = vec![0u32; size];
        for (block_id, block) in blocks.iter().enumerate() {
            for &elem in block {
                if elem < size {
                    ids[elem] = block_id as u32;
                }
            }
        }
        Partition::from_ids(ids)
    }

    /// Create from bitmask representation (for FFI/low-level interop).
    #[cfg(feature = "std")]
    pub fn from_bitmasks(size: usize, num_blocks: usize, bitmasks: &[u64]) -> Self {
        Partition::Fast(FastPartition::from_bitmasks(size, num_blocks, bitmasks))
    }

    /// Create from block_id assignments as a slice reference.
    pub fn from_block_ids(ids: &[u32]) -> Self {
        Partition::from_ids(ids.to_vec())
    }

    /// Extract the inner PartitionLattice (converting if needed).
    pub fn to_lattice(&self) -> PartitionLattice {
        match self {
            Partition::Lattice(pl) => pl.clone(),
            #[cfg(feature = "std")]
            Partition::Fast(fp) => PartitionLattice::from(fp),
            #[cfg(feature = "std")]
            Partition::Sparse(sp) => PartitionLattice::from(sp),
            #[cfg(feature = "std")]
            Partition::Cyclic(c) => PartitionLattice::from_ids(c.get_ids_owned()),
            #[cfg(feature = "std")]
            Partition::WideCyclic(c) => PartitionLattice::from_ids(c.get_ids_owned()),
            #[cfg(feature = "std")]
            Partition::FactorizedFast(f) => PartitionLattice::from_ids(f.get_ids_owned()),
        }
    }

    /// Extract the inner FastPartition (converting if needed).
    #[cfg(feature = "std")]
    pub fn to_fast(&self) -> FastPartition {
        match self {
            Partition::Lattice(pl) => FastPartition::from(pl),
            #[cfg(feature = "std")]
            Partition::Fast(fp) => fp.clone(),
            #[cfg(feature = "std")]
            Partition::Sparse(sp) => FastPartition::from(sp),
            #[cfg(feature = "std")]
            Partition::Cyclic(c) => c.to_fast_partition(),
            #[cfg(feature = "std")]
            Partition::WideCyclic(c) => c.to_fast_partition(),
            #[cfg(feature = "std")]
            Partition::FactorizedFast(f) => f.to_fast_partition(),
        }
    }

    /// Extract the inner SparsePartition (converting if needed).
    #[cfg(feature = "std")]
    pub fn to_sparse(&self) -> SparsePartition {
        match self {
            #[cfg(feature = "std")]
            Partition::Sparse(sp) => sp.clone(),
            Partition::Lattice(pl) => SparsePartition::from(pl),
            #[cfg(feature = "std")]
            Partition::Fast(fp) => SparsePartition::from(fp),
            #[cfg(feature = "std")]
            Partition::Cyclic(c) => SparsePartition::from_ids(&c.get_ids_owned()),
            #[cfg(feature = "std")]
            Partition::WideCyclic(c) => SparsePartition::from_ids(&c.get_ids_owned()),
            #[cfg(feature = "std")]
            Partition::FactorizedFast(f) => SparsePartition::from_ids(&f.get_ids_owned()),
        }
    }

    /// Returns true if using the Fast backend (always false in the no_std
    /// Lattice-only profile).
    pub fn is_fast(&self) -> bool {
        #[cfg(feature = "std")]
        {
            matches!(self, Partition::Fast(_))
        }
        #[cfg(not(feature = "std"))]
        {
            false
        }
    }

    /// Returns true if using the Sparse backend.
    pub fn is_sparse(&self) -> bool {
        #[cfg(feature = "std")]
        {
            matches!(self, Partition::Sparse(_))
        }
        #[cfg(not(feature = "std"))]
        {
            false
        }
    }

    /// Returns the partition as a list of blocks (each block = Vec of element indices).
    pub fn to_blocks(&self) -> Vec<Vec<usize>> {
        let ids = self.get_ids();
        if ids.is_empty() {
            return vec![];
        }
        let mut blocks_map: crate::HashMap<u32, Vec<usize>> = crate::HashMap::new();
        for (element, &block_id) in ids.iter().enumerate() {
            blocks_map.entry(block_id).or_default().push(element);
        }
        let mut blocks: Vec<Vec<usize>> = blocks_map.into_values().collect();
        blocks.sort_by(|a, b| a[0].cmp(&b[0]));
        blocks
    }

    /// Continuous lattice interpolation between two partitions.
    ///
    /// Moves continuously from `meet` (finest) to `join` (coarsest) by interpolating the block count.
    ///
    /// # Parameters
    /// - `meet`: π ∧ σ, the finest partition
    /// - `join`: π ∨ σ, the coarsest partition
    /// - `alpha`: interpolation parameter in `[0, 1]`; 0.0 gives `meet`, 1.0 gives `join`
    ///
    /// # Properties
    /// - **Endpoints:** `interpolate(m, j, 0.0) = m`, `interpolate(m, j, 1.0) = j`
    /// - **Refinement order:** meet ⪯ π_α ⪯ join
    /// - **Monotonicity:** α₁ < α₂ implies h(π_α₁) ≥ h(π_α₂)
    /// - **Idempotence:** `interpolate(π, π, α) = π`
    /// - **Determinism:** equal inputs give equal outputs
    pub fn interpolate_with(meet: &Self, join: &Self, alpha: f64) -> Self {
        let alpha = alpha.clamp(0.0, 1.0);
        let n = meet.size();

        if n == 0 {
            return meet.clone();
        }

        let meet_ids = meet.get_ids();
        let join_ids = join.get_ids();

        // Normalise meet ids into the contiguous range 0..b_meet.
        let mut meet_norm: crate::HashMap<u32, usize> = crate::HashMap::new();
        let mut norm_meet = vec![0usize; n];
        let mut b_meet = 0usize;
        for i in 0..n {
            let raw = meet_ids[i];
            let nb = meet_norm.len();
            let id = *meet_norm.entry(raw).or_insert(nb);
            if id == nb {
                b_meet += 1;
            }
            norm_meet[i] = id;
        }

        // Normalise join ids into the contiguous range 0..b_join.
        let mut join_norm: crate::HashMap<u32, usize> = crate::HashMap::new();
        let mut norm_join = vec![0usize; n];
        let mut b_join = 0usize;
        for i in 0..n {
            let raw = join_ids[i];
            let nb = join_norm.len();
            let id = *join_norm.entry(raw).or_insert(nb);
            if id == nb {
                b_join += 1;
            }
            norm_join[i] = id;
        }

        // Target block count: ceil((1 - alpha) * b_meet + alpha * b_join).
        //
        // The bounds are ordered rather than assumed. `meet` is documented as the finer
        // side, but a caller can pass the two the other way round, and `clamp(lo, hi)`
        // panics when `lo > hi` -- a library that dies on a caller's mistake takes the
        // prover embedding it down with it, from one line of input. Ordering them keeps
        // this identical whenever the precondition holds, and gives a defined result
        // when it does not: this function only ever coarsens away from `meet`, so if
        // `join` is the finer of the two there is nothing to coarsen and `meet` is
        // returned unchanged by the guard below.
        let target_f = (1.0 - alpha) * b_meet as f64 + alpha * b_join as f64;
        let (lo, hi) = (b_join.min(b_meet), b_join.max(b_meet));
        let target = (target_f.ceil() as usize).clamp(lo, hi);

        if target >= b_meet {
            return meet.clone();
        }
        if target <= b_join {
            return join.clone();
        }

        // Initialise the disjoint-set union.
        let mut parent: Vec<usize> = (0..b_meet).collect();
        let mut sz = vec![0usize; b_meet];
        let mut min_e = vec![n; b_meet]; // n is a sentinel above every valid index

        for i in 0..n {
            let mb = norm_meet[i];
            sz[mb] += 1;
            if i < min_e[mb] {
                min_e[mb] = i;
            }
        }

        // Path-compressed find
        fn find(parent: &mut [usize], x: usize) -> usize {
            let mut root = x;
            while parent[root] != root {
                root = parent[root];
            }
            // Path compression
            let mut cur = x;
            while cur != root {
                let next = parent[cur];
                parent[cur] = root;
                cur = next;
            }
            root
        }

        // join block -> list of meet blocks
        let mut jb_members: Vec<Vec<usize>> = vec![vec![]; b_join];
        for i in 0..n {
            jb_members[norm_join[i]].push(norm_meet[i]);
        }
        for jb in 0..b_join {
            jb_members[jb].sort_unstable();
            jb_members[jb].dedup();
        }

        // Reverse mapping: meet-block → set of join-blocks it belongs to
        let mut mb_to_jbs: Vec<Vec<usize>> = vec![vec![]; b_meet];
        for (jb, members) in jb_members.iter().enumerate() {
            for &mb in members {
                mb_to_jbs[mb].push(jb);
            }
        }

        // Priority queue: min-heap of (combined_size, min_element, ra, rb)
        use alloc::collections::BinaryHeap;
        use core::cmp::Reverse;
        let mut pq: BinaryHeap<Reverse<(usize, usize, usize, usize)>> = BinaryHeap::new();

        // Phase 1: initial pair collection
        for jb in 0..b_join {
            let mut roots: Vec<usize> = jb_members[jb]
                .iter()
                .map(|&mb| find(&mut parent, mb))
                .collect();
            roots.sort_unstable();
            roots.dedup();

            for i in 0..roots.len() {
                for j in (i + 1)..roots.len() {
                    let ra = roots[i];
                    let rb = roots[j];
                    let csz = sz[ra] + sz[rb];
                    let cmin = min_e[ra].min(min_e[rb]);
                    pq.push(Reverse((csz, cmin, ra, rb)));
                }
            }
        }

        let mut current = b_meet;

        // Phase 2: incremental merge
        while current > target {
            // Find valid pair from priority queue
            let mut found = false;
            while let Some(Reverse((_csz, _cmin, ra, rb))) = pq.pop() {
                // Validate: both must still be roots
                if find(&mut parent, ra) == ra && find(&mut parent, rb) == rb && ra != rb {
                    // Merge rb → ra
                    let new_sz = sz[ra] + sz[rb];
                    let new_min = min_e[ra].min(min_e[rb]);
                    parent[rb] = ra;
                    sz[ra] = new_sz;
                    min_e[ra] = new_min;
                    current -= 1;

                    // Re-insert affected pairs: join-blocks containing ra or rb
                    let mut affected_jbs = mb_to_jbs[ra].clone();
                    affected_jbs.extend_from_slice(&mb_to_jbs[rb]);
                    affected_jbs.sort_unstable();
                    affected_jbs.dedup();

                    // Update reverse mapping: ra now covers rb's join-blocks too
                    mb_to_jbs[ra] = affected_jbs.clone();

                    for &jb in &affected_jbs {
                        let mut roots: Vec<usize> = jb_members[jb]
                            .iter()
                            .map(|&mb| find(&mut parent, mb))
                            .collect();
                        roots.sort_unstable();
                        roots.dedup();

                        for &other in &roots {
                            if other != ra {
                                let csz = sz[ra] + sz[other];
                                let cmin = min_e[ra].min(min_e[other]);
                                pq.push(Reverse((csz, cmin, ra, other)));
                            }
                        }
                    }

                    found = true;
                    break;
                }
            }
            if !found {
                break; // no more valid pairs
            }
        }

        // Phase 3: build result partition
        let mut root_to_id: crate::HashMap<usize, u32> = crate::HashMap::new();
        let mut next_id = 0u32;
        let mut result_ids = vec![0u32; n];

        for i in 0..n {
            let root = find(&mut parent, norm_meet[i]);
            let nb = root_to_id.len() as u32;
            let id = *root_to_id.entry(root).or_insert(nb);
            if id == nb {
                next_id += 1;
            }
            result_ids[i] = id;
        }
        let _ = next_id;

        Partition::from_ids(result_ids)
    }

    /// Returns all distinction pairs (i, j) where i and j are in different blocks.
    /// O(n²) — use for small partitions or debugging only.
    pub fn distinctions(&self) -> Vec<(usize, usize)> {
        let ids = self.get_ids();
        let n = ids.len();
        let mut dits = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if ids[i] != ids[j] {
                    dits.push((i, j));
                }
            }
        }
        dits
    }
}

/// Expands to a match over **every** `Partition` variant, binding the inner
/// backend value to `$inner` and evaluating `$body` for each. This centralizes
/// the variant enumeration for *uniform-delegation* methods (those whose body
/// is identical across backends): adding a new variant means updating only this
/// macro plus the handful of methods with genuinely variant-specific logic
/// (whose exhaustive `match` the compiler will flag).
macro_rules! dispatch {
    ($self:expr, |$inner:ident| $body:expr) => {
        match $self {
            Partition::Lattice($inner) => $body,
            #[cfg(feature = "std")]
            Partition::Fast($inner) => $body,
            #[cfg(feature = "std")]
            Partition::Sparse($inner) => $body,
            #[cfg(feature = "std")]
            Partition::Cyclic($inner) => $body,
            #[cfg(feature = "std")]
            Partition::WideCyclic($inner) => $body,
            #[cfg(feature = "std")]
            Partition::FactorizedFast($inner) => $body,
        }
    };
}

// Delegate all PartitionOps methods via match
impl Partition {
    pub fn size(&self) -> usize {
        dispatch!(self, |p| p.size())
    }

    pub fn block_count(&self) -> usize {
        dispatch!(self, |p| p.block_count())
    }

    /// Alias for `block_count()` for backward compatibility.
    pub fn num_blocks(&self) -> usize {
        self.block_count()
    }

    /// Get the block ID for a given element.
    pub fn get_block(&self, element: usize) -> u32 {
        match self {
            Partition::Lattice(p) => p.get_ids()[element],
            #[cfg(feature = "std")]
            Partition::Fast(p) => p.get_block(element),
            #[cfg(feature = "std")]
            Partition::Sparse(p) => p.block_of(element),
            #[cfg(feature = "std")]
            Partition::Cyclic(c) => c.block_of(element),
            #[cfg(feature = "std")]
            Partition::WideCyclic(c) => c.block_of(element),
            #[cfg(feature = "std")]
            Partition::FactorizedFast(f) => f.block_of(element),
        }
    }

    pub fn get_ids(&self) -> &[u32] {
        dispatch!(self, |p| p.get_ids())
    }

    /// Pull this partition back along a permutation of its finite universe.
    ///
    /// `new_to_old[new] = old` specifies which old universe element occupies
    /// each new coordinate. The map must be a bijection over `0..self.size()`;
    /// invalid lengths, duplicates, or out-of-range indices return `None`.
    /// Block IDs are canonicalized by [`Partition::from_ids`], so the result
    /// depends only on the transported equivalence relation, not block labels.
    ///
    /// Universe permutations are lattice automorphisms: applying the same
    /// permutation to both operands preserves Meet, exact Join, and `d_⊕`.
    pub fn try_pullback_permutation(&self, new_to_old: &[usize]) -> Option<Self> {
        let n = self.size();
        if new_to_old.len() != n {
            return None;
        }
        let mut seen = vec![false; n];
        let mut ids = Vec::with_capacity(n);
        for &old in new_to_old {
            if old >= n || seen[old] {
                return None;
            }
            seen[old] = true;
            ids.push(self.get_block(old));
        }
        Some(Self::from_ids(ids))
    }

    pub fn logical_entropy(&self) -> f64 {
        if let Some(h) = crate::dispatch::try_distributed_entropy(self) {
            return h;
        }
        dispatch!(self, |p| p.logical_entropy())
    }

    pub fn dit_count(&self) -> usize {
        dispatch!(self, |p| p.dit_count())
    }

    /// Calculate mutual information between two partitions.
    ///
    /// Adaptively uses streaming path when meet materialization would
    /// exceed BLOCKS_MEMORY_LIMIT, avoiding 8MB cap lossy reduction.
    pub fn mutual_information(&self, other: &Self) -> f64 {
        let h_meet = self.fused_entropy_differentiate(other);
        self.logical_entropy() + other.logical_entropy() - h_meet
    }

    /// Logical divergence: D_L(self‖other) = h(self ∧ other) − h(other).
    ///
    /// Adaptively uses streaming path when meet materialization would
    /// exceed BLOCKS_MEMORY_LIMIT, avoiding 8MB cap lossy reduction.
    pub fn logical_divergence(&self, other: &Self) -> f64 {
        let h_meet = self.fused_entropy_differentiate(other);
        h_meet - other.logical_entropy()
    }

    /// Ditset XOR distance: d_⊕(P,Q) = 2·h(P∧Q) − h(P) − h(Q).
    ///
    /// Adaptively uses streaming path when meet materialization would
    /// exceed BLOCKS_MEMORY_LIMIT, avoiding 8MB cap lossy reduction.
    pub fn ditset_xor_distance(&self, other: &Self) -> f64 {
        #[cfg(feature = "std")]
        if self.should_use_streaming(other) {
            return crate::streaming::streaming_ditset_xor_distance(
                self,
                other,
                DEFAULT_CHUNK_SIZE,
            );
        }
        match (self, other) {
            (Partition::Lattice(a), Partition::Lattice(b)) => a.ditset_xor_distance(b),
            #[cfg(feature = "std")]
            (Partition::Fast(a), Partition::Fast(b)) => a.ditset_xor_distance(b),
            #[cfg(feature = "std")]
            (Partition::Sparse(a), Partition::Sparse(b)) => a.ditset_xor_distance(b),
            // Mixed-variant pairs — including every Cyclic/WideCyclic/FactorizedFast
            // comparison (no same-variant arm exists for them). Path A
            // (generate-vptree-optimize): compute directly via `get_block` + sort instead
            // of rebuilding two `SparsePartition`s on every call.
            _ => self.ditset_xor_distance_blocks(other),
        }
    }

    /// Minimum symmetric d_⊕ from `self` to any partition in `pool` — the nearest-neighbour
    /// distinction distance, i.e. the reduction underlying *novelty / surprise* ("how close is
    /// the nearest stored partition?"). Empty pool ⟹ `f64::INFINITY` (nothing to be near).
    ///
    /// The **single home of the `min d_⊕ over a pool` reduction** — the single source of truth: the
    /// recall-layer novelty force and the surprise / MIU /
    /// aligner sites all delegate here instead of re-folding the min, applying their own
    /// empty-default (1.0 / 2.0) and size-compat filter at the *callsite* (those are domain
    /// boundaries, not the force). Items are `&Partition`, so store-backed callers can pass
    /// `store.iter().map(|c| &c.partition)` with no `Vec<Partition>` clone.
    pub fn min_ditset_xor_distance_over<'a, I>(&self, pool: I) -> f64
    where
        I: IntoIterator<Item = &'a Self>,
    {
        pool.into_iter()
            .map(|p| self.ditset_xor_distance(p))
            .fold(f64::INFINITY, f64::min)
    }

    /// **INT-only novelty membership** — `true` iff no partition in `pool` is
    /// bit-identical to `self` (empty pool ⟹ `true`). This is the boolean
    /// specialization of `min_ditset_xor_distance_over(pool) > 0.0`: because
    /// `d_⊕` is a metric, `d_⊕(P,Q) = 0 ⟺ dit(P) = dit(Q) ⟺ P == Q` (YD-1),
    /// so "nearest stored distance > 0" is exactly "distinct from every stored
    /// item". Delegating to the `==` equality SSOT (`get_ids` canonical compare,
    /// with the Cyclic/WideCyclic compact fast paths) drops the per-comparison
    /// Meet allocation and the three f64 entropy divisions of the `d_⊕` path —
    /// the threshold-free WMF novelty gate (`tlw::is_novel`) needs only this
    /// boolean, so on FPU-less targets it runs as an INT equality scan (and
    /// short-circuits on the first match). Bit-exact with the f64 predicate
    /// (`unified_tests::test_is_distinct_from_all_matches_min_dxor_positive`).
    pub fn is_distinct_from_all<'a, I>(&self, pool: I) -> bool
    where
        I: IntoIterator<Item = &'a Self>,
    {
        !pool.into_iter().any(|p| p == self)
    }

    /// **Rand agreement index** — fraction of unordered element pairs `(i,j)` on which `self` and
    /// `other` *agree* about same-block / different-block classification (∈ `[0,1]`, `< 2` elements
    /// ⟹ `1.0`). The single home of the "Class A" agreement force — `pdlb::metrics::
    /// dit_jaccard` (a misnomer: it is the Rand index, **not** set-Jaccard) and `lpg::bridge`'s
    /// concordance metrics delegate here. ⚠ Distinct from [`dit_set_jaccard`](Self::dit_set_jaccard):
    /// Rand counts *agree-on-different* pairs too, so the two give different numbers (e.g. singletons
    /// vs one-block: Rand `< 1`, set-Jaccard `0`). Delegates to the slice core [`rand_agreement_ids`].
    pub fn rand_agreement(&self, other: &Self) -> f64 {
        rand_agreement_ids(self.get_ids(), other.get_ids())
    }

    /// **Set-Jaccard on dit-sets** — `|dit(self) ∩ dit(other)| / |dit(self) ∪ dit(other)|`, where
    /// `dit(p) = { (i,j), i<j : p[i] ≠ p[j] }` (the *differing* pairs). `None` ⟺ neither partition
    /// makes any distinction (`union = 0`) — callers apply their own empty-default (eigenform: jaccard
    /// `1.0`; pdn correlation: `0.0`). The single home of the "Class B" set-Jaccard force —
    /// `eigenform::dc_metric::dit_jaccard_distance` and `pdn::neuron::dit_correlation` delegate
    /// here. ⚠ Distinct from [`rand_agreement`](Self::rand_agreement) (Rand) and from
    /// `crystal::DitProfile::jaccard` (an h-ratio `h(P∨Q)/h(P∧Q)`); same name, three different forces.
    /// Same-universe precondition (equal element count); compared over `min(len)` defensively.
    pub fn dit_set_jaccard(&self, other: &Self) -> Option<f64> {
        let a = self.get_ids();
        let b = other.get_ids();
        let n = a.len().min(b.len());
        let (mut inter, mut union) = (0usize, 0usize);
        for i in 0..n {
            for j in (i + 1)..n {
                let da = a[i] != a[j];
                let db = b[i] != b[j];
                if da && db {
                    inter += 1;
                }
                if da || db {
                    union += 1;
                }
            }
        }
        if union == 0 {
            None
        } else {
            Some(inter as f64 / union as f64)
        }
    }

    /// `1 − `[`dit_set_jaccard`](Self::dit_set_jaccard) (empty ⟹ `1.0` jaccard ⟹ `0.0` distance) —
    /// the label-free dit-set XOR distance (eigenform `dit_jaccard_distance`'s exact behavior).
    pub fn dit_set_jaccard_distance(&self, other: &Self) -> f64 {
        1.0 - self.dit_set_jaccard(other).unwrap_or(1.0)
    }

    /// d_⊕ for mixed-variant pairs via per-element [`get_block`](Self::get_block) (no
    /// materialization) + sort-based block-size accounting. Bit-identical to the leaf
    /// impls' integer formula (`same_self + same_other − 2·same_both`) / N² — pinned by
    /// `unified_tests::test_ditset_xor_mixed_variant_matches_reference`.
    ///
    /// Path A (generate-vptree-optimize): replaces the prior `to_sparse()` rebuild that ran
    /// on *every* call. The `generate_retrieve_nearest` hot path compares a `Fast` query
    /// against `Cyclic` ngram crystals (always the mixed arm), so this was the bottleneck.
    /// `get_block` is O(1) per element for every backend (`Cyclic` = `pattern[i % L]`), so
    /// no compact crystal is ever densified — avoids the OnceLock dense-materialization
    /// regression. Thread-local scratch buffers
    /// avoid per-call allocation in the inner generation loop.
    /// Borrowed dense block-id slice when the backend stores one natively
    /// (Lattice/Fast/Sparse: `&self.element_to_block` / `&self.ids` — zero cost).
    /// Returns `None` for compact backends (Cyclic/WideCyclic/FactorizedFast) whose
    /// `get_ids()` would materialize *and OnceLock-cache* a universe-sized dense Vec
    /// (the gate-6.2 regression). Callers fall back to `get_block(i)` for those.
    fn dense_block_slice(&self) -> Option<&[u32]> {
        match self {
            Partition::Lattice(p) => Some(p.get_ids()),
            #[cfg(feature = "std")]
            Partition::Fast(p) => Some(p.get_ids()),
            #[cfg(feature = "std")]
            Partition::Sparse(p) => Some(p.get_ids()),
            _ => None,
        }
    }

    fn ditset_xor_distance_blocks(&self, other: &Self) -> f64 {
        let n = self.size();
        assert_eq!(n, other.size(), "Partition sizes must match");
        if n == 0 {
            return 0.0;
        }
        // Scratch buffers. (Was a thread_local reused buffer — a std-only micro-opt;
        // local allocation is value-identical and no_std-compatible.)
        // key = (a_block << 32 | b_block): sorting is a-major, so one pass yields both
        // same_both (full-key runs) and same_self (high-32 prefix runs).
        let mut comb: Vec<u64> = Vec::with_capacity(n);
        // b_block values only, for same_other (b is not contiguous in the a-major sort).
        let mut bvals: Vec<u32> = Vec::with_capacity(n);
        {
            // Dense backends (Lattice/Fast/Sparse) expose a borrowed `&[u32]` with no
            // materialization; iterate it directly to skip per-element enum dispatch.
            // Compact backends (Cyclic/WideCyclic/Factorized) fall back to `get_block`
            // (still O(1), no densification). The Option is loop-invariant ⇒ the inner
            // branch is perfectly predicted.
            let a_dense = self.dense_block_slice();
            let b_dense = other.dense_block_slice();
            for i in 0..n {
                let ai = match a_dense {
                    Some(s) => s[i],
                    None => self.get_block(i),
                };
                let bi = match b_dense {
                    Some(s) => s[i],
                    None => other.get_block(i),
                };
                comb.push(((ai as u64) << 32) | bi as u64);
                bvals.push(bi);
            }
            comb.sort_unstable();
            bvals.sort_unstable();

            // same_both = Σ |B_self ∩ B_other|² (full-key runs);
            // same_self = Σ |B_self|²        (high-32 prefix runs of the a-major sort).
            let mut same_both: u64 = 0;
            let mut same_self: u64 = 0;
            let (mut run_full, mut prev_full) = (0u64, u64::MAX);
            let (mut run_a, mut prev_a) = (0u64, u64::MAX);
            for &k in comb.iter() {
                if k == prev_full {
                    run_full += 1;
                } else {
                    same_both += run_full * run_full;
                    run_full = 1;
                    prev_full = k;
                }
                let a_hi = k >> 32;
                if a_hi == prev_a {
                    run_a += 1;
                } else {
                    same_self += run_a * run_a;
                    run_a = 1;
                    prev_a = a_hi;
                }
            }
            same_both += run_full * run_full;
            same_self += run_a * run_a;

            // same_other = Σ |B_other|²
            let mut same_other: u64 = 0;
            let (mut run_b, mut prev_b) = (0u64, u32::MAX);
            for &v in bvals.iter() {
                if v == prev_b {
                    run_b += 1;
                } else {
                    same_other += run_b * run_b;
                    run_b = 1;
                    prev_b = v;
                }
            }
            same_other += run_b * run_b;

            (same_self as i64 + same_other as i64 - 2 * same_both as i64) as f64 / (n * n) as f64
        }
    }

    // ── AccelerationBackend dispatch methods ─────────────────────

    /// Logical entropy via pluggable acceleration backend.
    pub fn logical_entropy_with(
        &self,
        backend: &dyn crate::acceleration::AccelerationBackend,
    ) -> f64 {
        backend.logical_entropy(self.get_ids())
    }

    /// Fused meet entropy via pluggable acceleration backend.
    pub fn fused_entropy_differentiate_with(
        &self,
        other: &Partition,
        backend: &dyn crate::acceleration::AccelerationBackend,
    ) -> f64 {
        backend.fused_entropy_differentiate(self.get_ids(), other.get_ids())
    }

    /// Logical divergence via pluggable acceleration backend.
    pub fn logical_divergence_with(
        &self,
        other: &Partition,
        backend: &dyn crate::acceleration::AccelerationBackend,
    ) -> f64 {
        backend.logical_divergence(self.get_ids(), other.get_ids())
    }

    /// Ditset XOR distance via pluggable acceleration backend.
    pub fn ditset_xor_distance_with(
        &self,
        other: &Partition,
        backend: &dyn crate::acceleration::AccelerationBackend,
    ) -> f64 {
        backend.ditset_xor_distance(self.get_ids(), other.get_ids())
    }

    /// Meet (differentiate) via pluggable acceleration backend.
    pub fn differentiate_with(
        &self,
        other: &Partition,
        backend: &dyn crate::acceleration::AccelerationBackend,
    ) -> Partition {
        Partition::from_ids(backend.differentiate(self.get_ids(), other.get_ids()))
    }

    /// Join (integrate) via pluggable acceleration backend (ShallowJoin d=2).
    ///
    /// GPU backends use SIMT symmetric propagation; CPU uses double-buffer.
    pub fn integrate_with(
        &self,
        other: &Partition,
        backend: &dyn crate::acceleration::AccelerationBackend,
    ) -> Partition {
        Partition::from_ids(backend.integrate_shallow(self.get_ids(), other.get_ids(), 2))
    }

    /// Exact Join via pluggable acceleration backend.
    pub fn integrate_exact_with(
        &self,
        other: &Partition,
        backend: &dyn crate::acceleration::AccelerationBackend,
    ) -> Partition {
        Partition::from_ids(backend.integrate_exact(self.get_ids(), other.get_ids()))
    }

    /// Weighted logical entropy $h_w(\pi)$.
    ///
    /// Ellerman(IGPL 2017 §5 p.812): $h_p(\pi) = \sum_{(u_j,u_k)\in\mathrm{dit}(\pi)} p_j p_k
    /// = 1 - \sum_i p(B_i)^2$ where $p(B_i) = \sum_{u_j \in B_i} p_j$.
    ///
    /// # Weights are element multiplicities, not probabilities
    ///
    /// Reading `w: U -> N` as a multiplicity makes `h_w(π)` exactly the uniform logical entropy
    /// of the multiset that repeats each element `u` exactly `w(u)` times (checked exhaustively for $N \le 5$).
    /// This is not a new measure: it is the same counting measure applied to a multiset, so it
    /// still presupposes no probability distribution.
    ///
    /// # Returns integers `(numerator, denominator)`
    ///
    /// $\left(W^2 - \sum_B m(B)^2,\; W^2\right)$, $m(B) = \sum_{u\in B} w(u)$, $W = \sum_u w(u)$.
    /// The computation is entirely integral, so no floating point is involved. For `w ≡ 1` the
    /// result is bit-exact with [`Self::logical_entropy`].
    ///
    /// A total mass of zero returns `(0, 1)`: with no elements there are no distinctions.
    ///
    /// # Reduced symmetry
    ///
    /// A non-uniform $w$ breaks the $S_N$ symmetry: the invariance group shrinks to $\mathrm{Stab}(w)$.
    /// Discrete-Noether DNT-1 (invariant under every $\sigma \in S_N$;
    /// Results assuming full relabelling invariance ($h(\sigma\pi)=h(\pi)$) therefore apply to the uniform case only.
    ///
    pub fn logical_entropy_weighted(&self, w: &[u64]) -> (u64, u64) {
        assert_eq!(
            w.len(),
            self.size(),
            "logical_entropy_weighted: weight length {} does not match universe size {}",
            w.len(),
            self.size()
        );
        let total: u64 = w.iter().sum();
        if total == 0 {
            return (0, 1); // zero mass: no distinctions
        }
        // Per-block mass m(B) = sum of w(u) over u in B; block ids are normalised to 0..k-1.
        let ids = self.get_ids();
        let nblocks = ids.iter().copied().max().map_or(0, |m| m as usize + 1);
        let mut mass = vec![0u64; nblocks];
        for (i, &id) in ids.iter().enumerate() {
            mass[id as usize] += w[i];
        }
        let sum_sq: u64 = mass.iter().map(|&m| m * m).sum();
        let w2 = total * total;
        (w2 - sum_sq, w2)
    }

    /// Logical cross-entropy: the same dit set measured with two different product measures.
    ///
    /// Ellerman(IGPL 2017 §12 p.822, verbatim): `μ_pq(S) = Σ{p_i q_i' : (i,i') ∈ S}`,
    /// `h(p‖q) = μ_pq(S_{i≠i'})`. Cross-entropy is exactly a re-parameterisation of the measure:
    /// the dit set is unchanged and only the measure becomes `(p, q)`.
    ///
    /// $$\mathrm{cross}_w(\pi; p, q) = \frac{PQ - \sum_B p(B)\,q(B)}{PQ},
    ///   \quad P=\textstyle\sum p,\; Q=\textstyle\sum q$$
    ///
    /// # One definition subsumes two levels (checked exhaustively for $N \le 4$)
    ///
    /// - `p == q` agrees with [`Self::logical_entropy_weighted`]: self-cross is self-entropy.
    /// - `π == ⊥` (discrete) agrees with the distribution-level cross-entropy `1 − Σ p̂ᵢ q̂ᵢ`.
    ///
    /// The distribution-level formula is thus the `π = ⊥` instance of this function, not a separate object.
    ///
    /// # Not the partition-level cross-entropy
    ///
    /// A cross-entropy defined *between two partitions* as
    /// `h(π‖σ) = 1 − tr[ρ(π)ρ(σ)]` degenerates into the join entropy, which in this crate's
    /// order convention is `h(Meet(π, σ))` and is already available. This function is the other
    /// thing: two measures over a single partition.
    ///
    /// # Symmetry and integrality
    ///
    /// `cross(π;p,q) == cross(π;q,p)`. Fully integral, with denominator $PQ$ and no floating point.
    /// A total mass of zero returns `(0, 1)`.
    ///
    /// # Divergence
    ///
    /// `d = cross − ½(h_p + h_q) ≥ 0`, and `= 0` exactly when `p == q` (exhaustive for $N \le 4$).
    /// `d` itself is not a metric (the triangle inequality fails); the metric is `sqrt(2d)`.
    /// The counterpart of KL divergence is not this symmetric quantity but the directed
    /// divergence `d*(p‖q) = Σpᵢ²/qᵢ − 1`. This function provides the symmetric form only.
    ///
    pub fn cross_entropy_weighted(&self, p: &[u64], q: &[u64]) -> (u64, u64) {
        assert_eq!(p.len(), self.size(), "cross_entropy_weighted: p length does not match universe size");
        assert_eq!(q.len(), self.size(), "cross_entropy_weighted: q length does not match universe size");
        let (pt, qt): (u64, u64) = (p.iter().sum(), q.iter().sum());
        if pt == 0 || qt == 0 {
            return (0, 1);
        }
        let ids = self.get_ids();
        let nblocks = ids.iter().copied().max().map_or(0, |m| m as usize + 1);
        let (mut pm, mut qm) = (vec![0u64; nblocks], vec![0u64; nblocks]);
        for (i, &b) in ids.iter().enumerate() {
            pm[b as usize] += p[i];
            qm[b as usize] += q[i];
        }
        // A dit is a pair from different blocks, so sum_{B!=C} p(B)q(C) = PQ - sum_B p(B)q(B).
        let same: u64 = pm.iter().zip(&qm).map(|(a, b)| a * b).sum();
        (pt * qt - same, pt * qt)
    }

    /// Partition implication, `self => consequent`.
    ///
    /// Builds a partition by replacing each block `B` of the antecedent as follows:
    /// - if some block `C` of `consequent` satisfies `B ⊆ C`, discretise `B` into singletons;
    /// - otherwise keep `B` intact.
    ///
    /// # Internalising the `implies` judgement
    ///
    /// **`a.implication(b) == ⊥  ⟺  a.refines(b)`**  (= `a ≤ b`)
    ///
    /// This turns the meta-level judgement `a implies b` (that is, `a ≤ b`) into an object-level
    /// operation with the same argument order, so `Imp(a, b)` reads as "a implies b".
    ///
    /// Note that the argument order is reversed with respect to Ellerman's original, which writes
    /// `σ ⇒ π = 1_U` iff `σ ⪯ π`. This crate uses the *order-dual* of that convention, so
    /// transcribing the formula literally would give `Imp(a, b) = ⊥` iff `b ≤ a` — pointing the
    /// opposite way from the `implies` judgement. To stay consistent with the Meet/Join convention
    /// used throughout the crate, `Imp(a, b)` here equals Ellerman's `(b ⇒ a)`.
    ///
    /// # Why this operation
    ///
    /// `Meet`, `Join` and `≤` give the lattice structure; this implication is what makes the
    /// partition lattice a logic.
    ///
    /// # Laws (exhaustive for $N \le 5$)
    ///
    /// `Imp(a,b) ≤ a` (always a refinement of `a`), `Imp(⊥,b) = ⊥`, `Imp(a,⊤) = ⊥`, `Imp(a,⊥) = a`,
    /// `Imp(a,a) = ⊥` · `Imp(Meet(a,b), b) = ⊥` · `Meet(b, Imp(a,b)) ≤ a` · `a ≤ Join(b, Imp(a,b))`
    ///
    /// # Not a residuum
    ///
    /// This is not a Heyting relative pseudo-complement. In the Galois adjunction
    /// `Meet(b,c) ≤ a ⟺ c ≤ Imp(a,b)` the `⟸` direction holds, but `⟹` fails 1464 times at $N = 4$.
    /// So `Imp(a,b)` is not the greatest element of `{c : Meet(b,c) ≤ a}`; residuals need not exist in a
    /// non-distributive lattice. This operation is a constructive indicator rather than a residual,
    /// which is why it is well defined here regardless.
    ///
    pub fn implication(&self, consequent: &Self) -> Self {
        assert_eq!(
            self.size(), consequent.size(),
            "implication: the two partitions have different universe sizes"
        );
        let n = self.size();
        if n == 0 {
            return consequent.clone();
        }

        // Look up consequent membership per element; comparing ids keeps this O(N).
        let cons = consequent.get_ids();
        let mut ids = vec![0u32; n];
        let mut next: u32 = 0;

        for block in self.to_blocks() {
            // `B ⊆ C` for some consequent block C iff every element of B shares a consequent block
            // (the consequent's blocks are disjoint, so the two conditions coincide).
            let first = cons[block[0]];
            let contained = block.iter().all(|&e| cons[e] == first);

            if contained {
                for &e in &block {
                    ids[e] = next; // discretise: a fresh block per element
                    next += 1;
                }
            } else {
                for &e in &block {
                    ids[e] = next; // keep the block intact
                }
                next += 1;
            }
        }
        Self::from_ids(ids)
    }

    pub fn differentiate(&self, other: &Self) -> Self {
        // PCS-A5 / PCS-A3: PCD-guided meet dispatch.
        // When the meet result would exceed BLOCKS_MEMORY_LIMIT in FastPartition,
        // route through SparsePartition to avoid lossy *id % max_blocks capping.
        #[cfg(feature = "std")]
        if self.should_use_sparse_refine(other) {
            let a = self.to_sparse();
            let b = other.to_sparse();
            return Partition::Sparse(a.differentiate(&b));
        }
        match (self, other) {
            (Partition::Lattice(a), Partition::Lattice(b)) => {
                Partition::Lattice(a.differentiate(b))
            }
            #[cfg(feature = "std")]
            (Partition::Fast(a), Partition::Fast(b)) => Partition::Fast(a.differentiate(b)),
            #[cfg(feature = "std")]
            (Partition::Sparse(a), Partition::Sparse(b)) => Partition::Sparse(a.differentiate(b)),
            _ => {
                // Mixed: convert to the faster of the two based on size.
                // The Fast path is std-only; the no_std profile always folds
                // through Lattice (same partition result).
                #[cfg(feature = "std")]
                if self.size() >= AUTO_SELECT_THRESHOLD {
                    let a = self.to_fast();
                    let b = other.to_fast();
                    return Partition::Fast(a.differentiate(&b));
                }
                let a = self.to_lattice();
                let b = other.to_lattice();
                Partition::Lattice(a.differentiate(&b))
            }
        }
    }

    /// Common refinement, named for what it does rather than for its position in the
    /// lattice. Identical to [`Self::differentiate`].
    ///
    /// The literature is split on which of meet and join is the common refinement:
    /// this crate calls it `Meet`, Ellerman and others call it `join`. `refine` reads
    /// the same under either convention, so it is the safer name when porting a
    /// formula. See the module documentation for the full correspondence.
    ///
    /// ```
    /// use partition_lattice::Partition;
    /// let a = Partition::from_ids(vec![0, 0, 1, 1]);
    /// let b = Partition::from_ids(vec![0, 1, 0, 1]);
    ///
    /// assert_eq!(a.refine(&b).get_ids(), a.differentiate(&b).get_ids());
    /// assert_eq!(a.refine(&b).block_count(), 4); // every element separated
    /// ```
    pub fn refine(&self, other: &Self) -> Self {
        self.differentiate(other)
    }

    /// Common coarsening, named for what it does rather than for its position in the
    /// lattice. Identical to [`Self::integrate`].
    ///
    /// The counterpart of [`Self::refine`]; see that method for why the neutral name
    /// exists.
    ///
    /// ⚠ Like `integrate`, this is the exact join. Learning loops that want the
    /// depth-bounded variant must ask for it explicitly via
    /// [`Self::integrate_with_strategy`].
    ///
    /// ```
    /// use partition_lattice::Partition;
    /// let a = Partition::from_ids(vec![0, 0, 1, 1]);
    /// let b = Partition::from_ids(vec![0, 1, 0, 1]);
    ///
    /// assert_eq!(a.coarsen(&b).get_ids(), a.integrate(&b).get_ids());
    /// assert_eq!(a.coarsen(&b).block_count(), 1); // everything merged
    /// ```
    pub fn coarsen(&self, other: &Self) -> Self {
        self.integrate(other)
    }

    /// Join (∨): ShallowJoin d=2 by default.
    ///
    /// Uses double-buffer label propagation for symmetric propagation.
    /// For N < 128 (Lattice regime), falls back to exact join.
    /// For exact transitive closure, use [`Partition::integrate_exact`].
    ///
    /// # Performance
    /// - 30~484x faster than ExactJoin
    /// - h(π) 8% higher (prevents over-coarsening)
    /// Join (∨): ExactJoin via Union-Find (default).
    ///
    /// `ShallowJoin` is meant for label propagation inside a PCC cycle and is reachable via
    /// `integrate_with_strategy(Shallow(d))`. The general-purpose `integrate()` uses
    /// `ExactJoin`, which is exact.
    /// Consensus (approximate Fréchet median under d_⊕) of a set of equal-size partitions.
    ///
    /// For each element pair (i, j) a *strict majority* of inputs agreeing that i ~ j (same
    /// block) becomes a same-block constraint; the constraints are closed to a valid partition
    /// by Union-Find (transitive repair). This is the **majority / median** consensus — the
    /// substrate identified as missing for store-time *learning* (abstracting many observed
    /// targets sharing a context into one generalizing prototype), distinct from Meet
    /// (∩ distinctions, finer) and Join (∪ indistinctions, coarser).
    ///
    /// d_⊕ relation: the relaxed median dit-set is the per-pair L1-median (= majority) of the
    /// inputs' indistinction bits; Union-Find repairs it to the nearest valid partition
    /// (correlation-clustering closure). Exact Fréchet median is NP-hard — this is the
    /// standard majority-closure approximation.
    ///
    /// Complexity O(N²·k) (pairwise majority over k inputs) — an R₂-class op intended for
    /// consensus over a *small* group (k targets per context-key), not large fan-ins.
    /// Returns `None` for empty input. Panics on size mismatch.
    pub fn consensus(parts: &[Partition]) -> Option<Partition> {
        let first = parts.first()?;
        let n = first.size();
        let k = parts.len();
        for p in parts {
            assert_eq!(
                p.size(),
                n,
                "consensus: all partitions must have equal size"
            );
        }
        if n == 0 {
            return Some(first.clone());
        }
        let mut dsu = crate::algorithms::dsu::DisjointSet::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                let same = parts
                    .iter()
                    .filter(|p| p.get_block(i) == p.get_block(j))
                    .count();
                if same * 2 > k {
                    dsu.union(i, j);
                }
            }
        }
        let ids: Vec<u32> = (0..n).map(|i| dsu.find(i) as u32).collect();
        Some(Partition::from_ids(ids))
    }

    /// **Mode-aware compression selector** — the *most frequent* partition (by dit identity) in
    /// `parts` (the statistical mode). Single-mode (all identical) → that value; multi-modal →
    /// the **dominant mode** (NOT a blend). Deterministic: count-ties broken by *first occurrence*
    /// (no sampling). `None` if `parts` is empty.
    ///
    /// Sibling of [`consensus`](Self::consensus) (Fréchet-median *blend* of all): on multi-modal
    /// groups `consensus` over-merges incompatible modes (blends), whereas `mode` keeps the
    /// dominant one. Validated on multi-modal ctx→next data —
    /// COMPRESS C2 (mode-aware Δ p@1 +0.062→+0.099 vs naive consensus, 93–95% multi-modal).
    /// Use `mode` for mode-aware relational compression, `consensus` for single-mode denoising.
    pub fn mode(parts: &[Partition]) -> Option<Partition> {
        // (ids, count, first_idx) — group by exact dit identity, count occurrences.
        let mut groups: Vec<(Vec<u32>, usize, usize)> = Vec::new();
        for (i, p) in parts.iter().enumerate() {
            let ids = p.get_ids().to_vec();
            match groups.iter_mut().find(|(k, _, _)| *k == ids) {
                Some(e) => e.1 += 1,
                None => groups.push((ids, 1, i)),
            }
        }
        // most frequent; count-tie → smaller first_idx (earliest, deterministic).
        let mut best: Option<(usize, usize, Vec<u32>)> = None; // (count, first_idx, ids)
        for (ids, count, first) in groups {
            let better = match &best {
                None => true,
                Some((bc, bf, _)) => count > *bc || (count == *bc && first < *bf),
            };
            if better {
                best = Some((count, first, ids));
            }
        }
        best.map(|(_, _, ids)| Partition::from_ids(ids))
    }

    pub fn integrate(&self, other: &Self) -> Self {
        self.integrate_exact(other)
    }

    /// Exact Join (∨) via Union-Find with full transitive closure.
    ///
    /// Preserves the original integrate() dispatch logic (PCS-A5).
    /// Use when mathematical exactness is required (witness verification, proofs).
    pub fn integrate_exact(&self, other: &Self) -> Self {
        // PCS-A5 dispatch for ExactJoin (Sparse-join fast path is std-only).
        //
        // The Sparse detour exists to stop a *result* `FastPartition` from allocating
        // more than 256 MB of block bitvectors — see `should_use_sparse_coarsen`.
        // Two operand pairs never build such a result and so are exempt:
        //
        //   (Fast, Fast)       `FastPartition::integrate` has its own elementwise
        //                      fallback for the high-K case (the second layer named
        //                      in that predicate's doc comment).
        //   (Lattice, Lattice) `PartitionLattice::integrate` returns a
        //                      `PartitionLattice`, which has no bitvectors at all,
        //                      so the event being guarded against cannot occur.
        //
        // Routing a Lattice pair here is not merely slower than its own path, it is
        // slower than using Sparse directly, since `to_sparse()` is added on top:
        // at n=1e6, K=1e5 the measured Join was Fast 20.2ms, Lattice-exempt 21.3ms,
        // Lattice-via-detour 327.7ms, Sparse-direct 94.6ms.
        #[cfg(feature = "std")]
        if !matches!(
            (self, other),
            (Partition::Fast(_), Partition::Fast(_))
                | (Partition::Lattice(_), Partition::Lattice(_))
        ) && self.should_use_sparse_coarsen(other)
        {
            let a = self.to_sparse();
            let b = other.to_sparse();
            return Partition::Sparse(a.integrate(&b));
        }
        match (self, other) {
            (Partition::Lattice(a), Partition::Lattice(b)) => Partition::Lattice(a.integrate(b)),
            #[cfg(feature = "std")]
            (Partition::Fast(a), Partition::Fast(b)) => Partition::Fast(a.integrate(b)),
            #[cfg(feature = "std")]
            (Partition::Sparse(a), Partition::Sparse(b)) => Partition::Sparse(a.integrate(b)),
            _ => {
                // Fast path std-only; no_std folds through Lattice (same result).
                #[cfg(feature = "std")]
                if self.size() >= AUTO_SELECT_THRESHOLD {
                    let a = self.to_fast();
                    let b = other.to_fast();
                    return Partition::Fast(a.integrate(&b));
                }
                let a = self.to_lattice();
                let b = other.to_lattice();
                Partition::Lattice(a.integrate(&b))
            }
        }
    }

    /// Join with configurable strategy.
    pub fn integrate_with_strategy(&self, other: &Self, strategy: CoarsenStrategy) -> Self {
        match strategy {
            CoarsenStrategy::Exact => self.integrate_exact(other),
            CoarsenStrategy::Shallow(d) => {
                if self.size() < AUTO_SELECT_THRESHOLD {
                    return self.integrate_exact(other);
                }
                Partition::from_ids(shallow_join_double_buffer(
                    self.get_ids(),
                    other.get_ids(),
                    d,
                ))
            }
            CoarsenStrategy::Auto => self.integrate(other),
        }
    }

    /// Sift: ternary conservative operation.
    ///
    /// sift(a; b, c) = a ∧ (b ∨ c) = Meet(a, Join(b, c))
    ///
    /// Retains only those distinctions of `self` that the community of
    /// `b` and `c` also recognises. Removes phantom dits.
    pub fn sift(&self, b: &Self, c: &Self) -> Self {
        self.differentiate(&b.integrate(c))
    }

    /// Span: ternary aggressive operation.
    ///
    /// span(a; b, c) = (a ∧ b) ∨ (a ∧ c) = Join(Meet(a, b), Meet(a, c))
    ///
    /// Preserves all distinctions that `self` shares with `b` or `c`
    /// individually, including phantom dits.
    pub fn span(&self, b: &Self, c: &Self) -> Self {
        self.differentiate(b).integrate(&self.differentiate(c))
    }

    /// Phantom budget: non-distributive gap measure.
    ///
    /// Φ(a; b, c) = h(span(a; b, c)) − h(sift(a; b, c)) ≥ 0
    ///
    /// Zero iff the triple (a, b, c) is distributive.
    pub fn phantom_budget(&self, b: &Self, c: &Self) -> f64 {
        self.span(b, c).logical_entropy() - self.sift(b, c).logical_entropy()
    }

    /// Three-way Hu logical mutual information.
    ///
    /// `I₃(a,b,c) = h(a) + h(b) + h(c) − h(a∧b) − h(a∧c) − h(b∧c) + h(a∧b∧c)`
    ///
    /// The cardinality projection of dit-set inclusion-exclusion. The sign may be negative:
    /// positive ⟹ redundant info (overlap), negative ⟹ synergy (combined > separate sum).
    /// the modular property of logical entropy (`h(P∧Q) + h(P∨Q) = h(P) + h(Q)`) is the lattice
    /// ground for multi-partition inclusion-exclusion.
    ///
    pub fn triple_mutual_information(&self, b: &Self, c: &Self) -> f64 {
        let h_a = self.logical_entropy();
        let h_b = b.logical_entropy();
        let h_c = c.logical_entropy();
        let h_ab = self.fused_entropy_differentiate(b);
        let h_ac = self.fused_entropy_differentiate(c);
        let h_bc = b.fused_entropy_differentiate(c);
        // a∧b∧c = (a∧b)∧c via materialization
        let h_abc = self.differentiate(b).fused_entropy_differentiate(c);
        h_a + h_b + h_c - h_ab - h_ac - h_bc + h_abc
    }

    /// Four-way Hu logical mutual information, with alternating inclusion-exclusion signs.
    ///
    /// `I₄(a,b,c,d) = Σh(πᵢ) − Σh(πᵢ∧πⱼ) + Σh(πᵢ∧πⱼ∧πₖ) − h(π₁∧π₂∧π₃∧π₄)`
    ///
    /// Costs 11 entropy reads and 5 Meet materialisations, which is expensive for large `N`;
    /// intended for measurement and diagnostics rather than hot loops.
    pub fn quad_mutual_information(&self, b: &Self, c: &Self, d: &Self) -> f64 {
        let h_a = self.logical_entropy();
        let h_b = b.logical_entropy();
        let h_c = c.logical_entropy();
        let h_d = d.logical_entropy();
        // pairs (6)
        let h_ab = self.fused_entropy_differentiate(b);
        let h_ac = self.fused_entropy_differentiate(c);
        let h_ad = self.fused_entropy_differentiate(d);
        let h_bc = b.fused_entropy_differentiate(c);
        let h_bd = b.fused_entropy_differentiate(d);
        let h_cd = c.fused_entropy_differentiate(d);
        // The four triples, reusing the materialised Meets.
        let ab = self.differentiate(b);
        let h_abc = ab.fused_entropy_differentiate(c);
        let h_abd = ab.fused_entropy_differentiate(d);
        let ac = self.differentiate(c);
        let h_acd = ac.fused_entropy_differentiate(d);
        let bc = b.differentiate(c);
        let h_bcd = bc.fused_entropy_differentiate(d);
        // quad
        let h_abcd = ab.differentiate(c).fused_entropy_differentiate(d);
        h_a + h_b + h_c + h_d - h_ab - h_ac - h_ad - h_bc - h_bd - h_cd
            + h_abc
            + h_abd
            + h_acd
            + h_bcd
            - h_abcd
    }

    /// Complex meet with phase interference (delegates to PartitionLattice).
    pub fn complex_refine(&self, other: &Self, phase_self: i8, phase_other: i8) -> (Self, f64) {
        let a = self.to_lattice();
        let b = other.to_lattice();
        let (result, interference) = a.complex_refine(&b, phase_self, phase_other);
        (Partition::Lattice(result), interference)
    }

    pub fn refines(&self, other: &Self) -> bool {
        match (self, other) {
            (Partition::Lattice(a), Partition::Lattice(b)) => a.refines(b),
            #[cfg(feature = "std")]
            (Partition::Fast(a), Partition::Fast(b)) => a.refines(b),
            #[cfg(feature = "std")]
            (Partition::Sparse(a), Partition::Sparse(b)) => a.refines(b),
            _ => {
                // Mixed: use Sparse for efficient comparison (std); to_fast on no_std
                // (value-identical — refines is backend-independent).
                #[cfg(feature = "std")]
                {
                    let a = self.to_sparse();
                    let b = other.to_sparse();
                    a.refines(&b)
                }
                #[cfg(not(feature = "std"))]
                {
                    let a = self.to_lattice();
                    let b = other.to_lattice();
                    a.refines(&b)
                }
            }
        }
    }

    /// Fused meet + refines: computes meet = self ∧ other and checks if result refines old_model.
    pub fn fused_refine_and_check(&self, other: &Self, old_model: &Self) -> (Self, bool) {
        let meet = self.differentiate(other);
        let does_refine = meet.refines(old_model);
        (meet, does_refine)
    }

    /// Fused h(π ∧ σ) without materializing the meet partition.
    ///
    /// Adaptively selects streaming path when the estimated meet memory
    /// would exceed BLOCKS_MEMORY_LIMIT (8MB), avoiding lossy block capping.
    /// For small partitions, delegates to backend-specific SIMD/bitwise path.
    pub fn fused_entropy_differentiate(&self, other: &Self) -> f64 {
        if let Some(h) = crate::dispatch::try_distributed_meet_entropy(self, other) {
            return h;
        }
        #[cfg(feature = "std")]
        if self.should_use_streaming(other) {
            return crate::streaming::fused_refine_entropy(self, other, DEFAULT_CHUNK_SIZE);
        }
        match (self, other) {
            (Partition::Lattice(a), Partition::Lattice(b)) => a.fused_entropy_differentiate(b),
            #[cfg(feature = "std")]
            (Partition::Fast(a), Partition::Fast(b)) => a.fused_entropy_differentiate(b),
            #[cfg(feature = "std")]
            (Partition::Sparse(a), Partition::Sparse(b)) => a.fused_entropy_differentiate(b),
            _ => {
                #[cfg(feature = "std")]
                {
                    let a = self.to_sparse();
                    let b = other.to_sparse();
                    a.fused_entropy_differentiate(&b)
                }
                #[cfg(not(feature = "std"))]
                {
                    let a = self.to_lattice();
                    let b = other.to_lattice();
                    a.fused_entropy_differentiate(&b)
                }
            }
        }
    }

    /// Fused h(π ∨ σ) without materializing the join partition.
    ///
    /// Uses the block co-occurrence graph approach: Union-Find on K₁+K₂
    /// block nodes instead of N elements. O(K₁×K₂) memory, independent of |U|.
    ///
    /// This is faster than `integrate().logical_entropy()` at all scales
    /// because it avoids materializing the join partition entirely.
    #[cfg(feature = "std")]
    pub fn fused_entropy_integrate(&self, other: &Self) -> f64 {
        crate::streaming::block_graph_coarsen_entropy(self, other, DEFAULT_CHUNK_SIZE)
    }

    pub fn logical_area(&self) -> f64 {
        match self {
            Partition::Lattice(p) => p.logical_area(),
            #[cfg(feature = "std")]
            Partition::Fast(p) => p.logical_area(),
            #[cfg(feature = "std")]
            Partition::Sparse(p) => p.logical_area(),
            #[cfg(feature = "std")]
            Partition::Cyclic(_) => self.to_fast().logical_area(),
            #[cfg(feature = "std")]
            Partition::WideCyclic(_) => self.to_fast().logical_area(),
            #[cfg(feature = "std")]
            Partition::FactorizedFast(_) => self.to_fast().logical_area(),
        }
    }

    // ── Adaptive streaming dispatch ─────────────────────────────

    /// PCS-A5: Unified O(1) dispatch diagnosis for partition operations.
    ///
    /// Combines `should_use_sparse_refine`, `should_use_sparse_coarsen`, and
    /// `should_use_streaming` into a single entry point based on PCS regime theory.
    ///
    /// Three-layer application:
    /// - Layer 3: synthesis level, i.e. problem complexity
    /// - Layer 2: Operation level (PCS, this function) — backend selection
    /// - Layer 1: Hardware level (distributed/GPU) — resource allocation
    pub fn diagnose(&self, op: PartitionOp, other: &Self) -> DispatchAdvice {
        let n = self.size();
        if n < AUTO_SELECT_THRESHOLD {
            return DispatchAdvice::Fast;
        }
        let k1 = self.block_count() as u64;
        let k2 = other.block_count() as u64;
        let words_per_block = ((n + 63) / 64) as u64;

        match op {
            PartitionOp::Meet => {
                // Meet result: up to K₁×K₂ blocks
                let meet_mem = k1
                    .saturating_mul(k2)
                    .saturating_mul(words_per_block)
                    .saturating_mul(8);
                if meet_mem > STREAMING_DISPATCH_THRESHOLD as u64 {
                    DispatchAdvice::Sparse
                } else {
                    DispatchAdvice::Fast
                }
            }
            PartitionOp::Join => {
                // Join result: ≤ max(K₁,K₂) blocks
                let k_max = k1.max(k2);
                let join_mem = k_max.saturating_mul(words_per_block).saturating_mul(8);
                const JOIN_RESULT_MEMORY_LIMIT: u64 = 256 * 1024 * 1024;
                if join_mem > JOIN_RESULT_MEMORY_LIMIT {
                    DispatchAdvice::Sparse
                } else {
                    DispatchAdvice::Fast
                }
            }
            PartitionOp::Metric => {
                // Metrics (h, D_L, d_⊕): streaming avoids materialization
                let meet_mem = k1
                    .saturating_mul(k2)
                    .saturating_mul(words_per_block)
                    .saturating_mul(8);
                if meet_mem > STREAMING_DISPATCH_THRESHOLD as u64 {
                    DispatchAdvice::Streaming
                } else {
                    DispatchAdvice::Fast
                }
            }
        }
    }

    /// Returns true when meet materialization would likely exceed
    /// BLOCKS_MEMORY_LIMIT, causing lossy block capping in FastPartition.
    ///
    /// Estimated meet memory = K₁ × K₂ × ⌈N/64⌉ × 8 bytes.
    pub fn should_use_streaming(&self, other: &Self) -> bool {
        let n = self.size();
        if n == 0 {
            return false;
        }
        let k1 = self.block_count();
        let k2 = other.block_count();
        let words_per_block = (n + 63) / 64;
        k1.saturating_mul(k2)
            .saturating_mul(words_per_block)
            .saturating_mul(8)
            > STREAMING_DISPATCH_THRESHOLD
    }

    /// PCS-A5 / PCD-guided Meet dispatch: returns true when the meet result's
    /// FastPartition BitVec memory would exceed STREAMING_DISPATCH_THRESHOLD (8MB).
    ///
    /// Meet result has up to K₁×K₂ blocks. Memory = K₁×K₂ × ⌈N/64⌉ × 8 bytes.
    /// When this exceeds 8MB, FastPartition's differentiate_elementwise() would
    /// apply lossy `*id % max_blocks` reduction. Instead, route through
    /// SparsePartition which handles arbitrary K without loss.
    ///
    /// Diagnosis cost: O(1) — only reads size() and block_count().
    pub fn should_use_sparse_refine(&self, other: &Self) -> bool {
        let n = self.size();
        if n < AUTO_SELECT_THRESHOLD {
            return false;
        }
        let k1 = self.block_count() as u64;
        let k2 = other.block_count() as u64;
        let words_per_block = ((n + 63) / 64) as u64;
        k1.saturating_mul(k2)
            .saturating_mul(words_per_block)
            .saturating_mul(8)
            > STREAMING_DISPATCH_THRESHOLD as u64
    }

    /// PCD-guided Join dispatch: returns true when the result FastPartition's
    /// BitVec memory would exceed 256MB.
    ///
    /// Key difference from `should_use_streaming()` (meet):
    /// - Meet result has K₁×K₂ blocks → memory = K₁×K₂ × ⌈N/64⌉ × 8
    /// - Join result has ≤ max(K₁,K₂) blocks → memory = max(K) × ⌈N/64⌉ × 8
    ///
    /// Two-layer defense:
    /// - Layer 1 (here): OOM prevention — Sparse route for high-memory results
    /// - Layer 2 (optimized.rs): CPU prevention — elementwise DSU for O(K·N/64) explosion
    ///
    /// Examples:
    /// - K=10, N=100M: 10×12.5MB = 125MB < 256MB → Fast ✅
    /// - K=7071, N=50M: 7071×6.25MB = 44GB > 256MB → Sparse ✅
    pub fn should_use_sparse_coarsen(&self, other: &Self) -> bool {
        let n = self.size();
        if n < AUTO_SELECT_THRESHOLD {
            return false;
        }
        let k_max = self.block_count().max(other.block_count());
        let words_per_block = (n + 63) / 64;
        // 256MB: generous enough for K=10 at N=100M (125MB),
        // catches K=√N at N=50M (44GB) and K=1000 at N=10M (1.25GB).
        const JOIN_RESULT_MEMORY_LIMIT: usize = 256 * 1024 * 1024;
        k_max * words_per_block * 8 > JOIN_RESULT_MEMORY_LIMIT
    }

    // ── Parallel / Batch API (transparent delegation) ───────────

    /// Parallel logical entropy — auto-converts Lattice/Sparse→Fast for SIMD acceleration.
    pub fn logical_entropy_parallel(&self) -> f64 {
        match self {
            #[cfg(feature = "std")]
            Partition::Fast(fp) => fp.logical_entropy_parallel(),
            #[cfg(feature = "std")]
            Partition::Sparse(sp) => sp.logical_entropy(), // Already O(1) from precomputed value
            #[cfg(feature = "std")]
            // `FastPartition::logical_entropy_parallel` just reads its cached
            // `sum_block_sizes_sq`, so converting here allocated one bitvector per
            // block and then never read them. `PartitionLattice` now caches the same
            // sum, so ask it directly.
            Partition::Lattice(pl) => pl.logical_entropy(),
            #[cfg(not(feature = "std"))]
            Partition::Lattice(pl) => pl.logical_entropy(), // no SIMD backend — sequential
            #[cfg(feature = "std")]
            Partition::Cyclic(c) => c.logical_entropy(), // O(cycle_len), analytical
            #[cfg(feature = "std")]
            Partition::WideCyclic(c) => c.logical_entropy(), // O(cycle_len), analytical
            #[cfg(feature = "std")]
            Partition::FactorizedFast(f) => f.logical_entropy(), // O(1), cached Σ|B|²
        }
    }

    /// Parallel meet (differentiate) — auto-converts Lattice→Fast (std). The
    /// no_std profile has no Fast/SIMD backend, so it falls back to the exact
    /// sequential `differentiate` (same partition result).
    pub fn differentiate_parallel(&self, other: &Self) -> Self {
        #[cfg(feature = "std")]
        {
            let a = self.to_fast();
            let b = other.to_fast();
            Partition::Fast(a.differentiate_parallel(&b))
        }
        #[cfg(not(feature = "std"))]
        {
            self.differentiate(other)
        }
    }

    /// Batch entropy for multiple partitions in parallel (std-only SIMD batch API).
    #[cfg(feature = "std")]
    pub fn entropy_batch(partitions: &[Partition]) -> Vec<f64> {
        let fps: Vec<FastPartition> = partitions.iter().map(|p| p.to_fast()).collect();
        crate::optimized::BatchPartitionOps::entropy_batch(&fps)
    }

    /// Pairwise mutual information matrix in parallel (std-only SIMD batch API).
    /// Returns Vec where `result[i * n + j] = I(partitions[i]; partitions[j])`.
    #[cfg(feature = "std")]
    pub fn mutual_info_matrix(partitions: &[Partition]) -> Vec<f64> {
        let fps: Vec<FastPartition> = partitions.iter().map(|p| p.to_fast()).collect();
        crate::optimized::BatchPartitionOps::mutual_info_matrix(&fps)
    }

    /// Restrict partition to a subset of the universe.
    ///
    /// Given π ∈ Π(U) and S ⊆ U, returns π|\_S ∈ Π(S).
    /// Elements are renumbered 0..|S|-1 in the order they appear in `subset`.
    /// Block IDs are renumbered to be contiguous starting from 0.
    ///
    /// # Mathematical Property
    /// For meet: (π ∧ σ)|\_S = π|\_S ∧ σ|\_S (commutativity with restriction).
    ///
    /// # Examples
    /// ```
    /// use partition_lattice::Partition;
    /// let p = Partition::from_ids(vec![0, 0, 1, 2, 2, 2]);
    /// let r = p.restrict(&[0, 1, 4, 5]);
    /// assert_eq!(r.size(), 4);
    /// assert_eq!(r.block_count(), 2);
    /// ```
    pub fn restrict(&self, subset: &[usize]) -> Partition {
        if subset.is_empty() {
            return Partition::from_ids(vec![]);
        }
        let ids = self.get_ids();
        let raw: Vec<u32> = subset.iter().map(|&i| ids[i]).collect();
        let mut remap = crate::HashMap::new();
        let mut next_id = 0u32;
        let renumbered: Vec<u32> = raw
            .iter()
            .map(|&bid| {
                *remap.entry(bid).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect();
        Partition::from_ids(renumbered)
    }

    /// Returns per-block element counts as (block_id, count) pairs.
    ///
    /// Useful for streaming entropy computation: h(π) = 1 - Σ(|Bⱼ|/N)².
    /// The sum of all counts equals `self.size()`.
    ///
    /// # Examples
    /// ```
    /// use partition_lattice::Partition;
    /// let p = Partition::from_ids(vec![0, 0, 1, 2, 2, 2]);
    /// let sizes = p.block_sizes();
    /// assert_eq!(sizes.len(), 3);
    /// ```
    pub fn block_sizes(&self) -> Vec<(u32, usize)> {
        let ids = self.get_ids();
        let mut counts: crate::HashMap<u32, usize> = crate::HashMap::new();
        for &bid in ids {
            *counts.entry(bid).or_default() += 1;
        }
        let mut result: Vec<(u32, usize)> = counts.into_iter().collect();
        result.sort_by_key(|(bid, _)| *bid);
        result
    }
}

// Convenient From impls
impl From<PartitionLattice> for Partition {
    fn from(pl: PartitionLattice) -> Self {
        #[cfg(feature = "std")]
        if pl.size() >= AUTO_SELECT_THRESHOLD {
            return Partition::Fast(FastPartition::from(&pl));
        }
        Partition::Lattice(pl)
    }
}

#[cfg(feature = "std")]
impl From<FastPartition> for Partition {
    fn from(fp: FastPartition) -> Self {
        Partition::Fast(fp)
    }
}

#[cfg(feature = "std")]
impl From<SparsePartition> for Partition {
    fn from(sp: SparsePartition) -> Self {
        Partition::Sparse(sp)
    }
}

impl PartialEq for Partition {
    fn eq(&self, other: &Self) -> bool {
        // Fast path: Cyclic ↔ Cyclic compares canonicalized patterns
        // (O(MAX_CYCLE)) WITHOUT materializing get_ids. Critical for
        // H_SFM_FUSE_MEET_AMORTIZATION skip check on cyclic crystals —
        // a full get_ids materialization would defeat the substrate's
        // compact-at-rest property. (Cyclic/WideCyclic are std-only backends.)
        #[cfg(feature = "std")]
        if let (Partition::Cyclic(a), Partition::Cyclic(b)) = (self, other) {
            return a == b;
        }
        // Layer A: WideCyclic ↔ WideCyclic gets the same compact-equality
        // fast path. Mixed Cyclic ↔ WideCyclic intentionally falls through
        // to id materialization (different L ranges).
        #[cfg(feature = "std")]
        if let (Partition::WideCyclic(a), Partition::WideCyclic(b)) = (self, other) {
            return a == b;
        }
        self.get_ids() == other.get_ids()
    }
}

impl Eq for Partition {}

/// **Rand agreement index over two id-sequences** — fraction of unordered pairs `(i,j)` where
/// `a` and `b` agree on same-id / different-id classification (∈ `[0,1]`, `< 2` elements ⟹ `1.0`).
/// The slice-level core of the Class-A agreement force ([`Partition::rand_agreement`] delegates
/// here). Exposed for callers that hold raw label/id slices rather than a `Partition` — e.g.
/// `lpg::bridge::{structural_concordance, information_preservation_rate}` (argmax-label vs
/// partition-block concordance). ⚠ This is the **Rand index**, not set-Jaccard (it counts
/// agree-on-different pairs); see [`Partition::dit_set_jaccard`] for the set-Jaccard force.
/// Same-length precondition; compared over `min(len)` defensively.
pub fn rand_agreement_ids(a: &[u32], b: &[u32]) -> f64 {
    let n = a.len().min(b.len());
    let total = (n * (n - 1)) / 2;
    if total == 0 {
        return 1.0;
    }
    let mut agree = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if (a[i] == a[j]) == (b[i] == b[j]) {
                agree += 1;
            }
        }
    }
    agree as f64 / total as f64
}

/// **Label-blind identity key of an id-sequence** — renumber block ids to `0..k-1` in
/// order of first appearance.
///
/// Two partitions are the *same partition* iff their canonical forms are equal; block
/// labels carry no information. This is the standard key for membership,
/// de-dup, cycle detection, and equivalence-class numbering over partition traces.
///
/// SSOT. Four byte-identical private copies previously lived in downstream crates; one of
/// them carried a note about avoiding a cross-crate dependency cycle, which this placement
/// resolves: every one of them already depends on this crate, so no cycle is possible. Semantics are
/// pinned against the legacy copies by
/// `unified_tests::test_canonicalize_ids_matches_legacy_copy_semantics`.
///
/// Slice-level like [`rand_agreement_ids`], for callers holding raw ids rather than a
/// `Partition` (the usual shape is `canonicalize_ids(p.get_ids())`).
pub fn canonicalize_ids(ids: &[u32]) -> Vec<u32> {
    let mut map: crate::HashMap<u32, u32> = crate::HashMap::new();
    let mut next_id = 0u32;
    let mut out = Vec::with_capacity(ids.len());
    for &id in ids {
        let canon = *map.entry(id).or_insert_with(|| {
            let v = next_id;
            next_id += 1;
            v
        });
        out.push(canon);
    }
    out
}

// Unit tests live in `unified_tests.rs` (kept as a child module of `unified`
// via #[path] so `use super::*` still reaches this module's private items).
#[cfg(test)]
#[path = "unified_tests.rs"]
mod tests;
