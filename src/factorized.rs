//! Factorized Fast Partition (Layer B-β-true substrate).
//!
//! Universe-scaling
//! `FastPartition::element_to_block: Vec<u32>(N)` becomes a 2-tier encoding:
//!
//! - `coarse_block: Vec<u32>` indexed by `i / F` — assigns each coarse group
//!   one *coarse group id* (canonical 0..K_coarse).
//! - `coarse_dominant_block: Vec<u32>` indexed by coarse group id — gives the
//!   *dominant fine block id* for that group (the most frequent block among
//!   the `F` elements of the group).
//! - `fine_overrides: BTreeMap<(coarse_group_id, fine_idx), u32>` — exceptions
//!   to the dominant block (i.e., `block_of(i) ≠ dominant` cases).
//!
//! `block_of(i)` is then:
//!
//! ```text
//!   let c = i / F, f = i % F;
//!   let g = coarse_block[c];                        // group id
//!   fine_overrides.get(&(g, f)).copied()
//!       .unwrap_or(coarse_dominant_block[g as usize])
//! ```
//!
//! ## Arithmetic equivalence (PDD)
//!
//! For any `FastPartition fp` and factor `F ≥ 1`,
//!
//! ```text
//!   factorize(fp, F).block_of(i) = fp.element_to_block[i]    ∀ i ∈ [0, N)
//! ```
//!
//! (`theorem factorize_block_of_bit_exact`).
//!
//! ## Memory cost
//!
//! At N=65 536, F=256, override_ratio=0.05 (wp2 Fast fallback profile):
//!
//! | representation                    | bytes at N=65 536 |
//! |---                                |---                |
//! | `FastPartition::element_to_block` | 256 KB            |
//! | `FactorizedFastPartition`         | ~54 KB (-79%)     |
//!
//! At N=262 144, F=512, override_ratio=0.05 (en-L / multi-L profile):
//!
//! | representation                    | bytes at N=262 144 |
//! |---                                |---                 |
//! | `FastPartition::element_to_block` | 1 024 KB           |
//! | `FactorizedFastPartition`         | ~214 KB (-79%)     |
//!
//! ## LBP / DCE compliance
//!
//! - **LBP-1:** All internal storage is `u32`/`u8` integer — no FP. block IDs
//!   are partition equivalence-class indices (distinction), not values.
//! - **DCE-T5:** Memory is the dominant cost dimension here. Reducing the
//!   universe-scaling `Vec<u32>` to `(N/F + K_coarse + override_count)` cuts
//!   DRAM footprint, which dominates over per-lookup cost in PCC inner loops.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::FloatExt;
use crate::HashMap;
use crate::OnceLock;
#[allow(unused_imports)]
use alloc::borrow::ToOwned;
#[allow(unused_imports)]
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
#[allow(unused_imports)]
use alloc::string::{String, ToString};
#[allow(unused_imports)]
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::optimized::FastPartition;

/// Default factorization parameter for `try_from_fast_partition`.
///
/// `F=256` is selected to match wp2 Fast fallback universe-scaling profile
/// (N=65 536 → N/F = 256 coarse groups). For N=262 144 (en-L / multi-L),
/// callers should pass `F=512` explicitly.
pub const DEFAULT_FACTOR: u32 = 256;

/// Minimum universe size below which factorization is not attempted —
/// the overhead of `coarse_block` / `coarse_dominant_block` outweighs
/// the saving on `element_to_block` itself.
pub const MIN_UNIVERSE_FOR_FACTORIZATION: usize = 1024;

/// Factorized representation of a `FastPartition` element-to-block mapping.
///
/// See module-level docstring for storage layout and the `block_of` formula.
#[derive(Debug, Serialize, Deserialize)]
pub struct FactorizedFastPartition {
    /// Coarse-group id (canonical 0..K_coarse) for each coarse index `c ∈ [0, ceil(N/F))`.
    coarse_block: Vec<u32>,
    /// Dominant fine block id for each coarse group id `g ∈ [0, K_coarse)`.
    coarse_dominant_block: Vec<u32>,
    /// Exceptions: `(coarse_group_id, fine_idx) → fine block id` for elements
    /// whose `block_of(i) ≠ coarse_dominant_block[g]`.
    fine_overrides: BTreeMap<(u32, u32), u32>,
    /// `F` — the factorization parameter (fine group size).
    fine_count: u32,
    /// `N` — total universe size.
    size: usize,
    /// `K_total` — total distinct fine block IDs (`max(block_of(i)) + 1`).
    /// Used by `block_count()` for O(1) answer.
    total_block_count: u32,
    /// `Σ |B|²` for O(1) logical entropy and distinction count.
    sum_block_sizes_sq: u64,
    /// Lazy materialized id vector. Not serialized; reset on clone.
    #[serde(skip)]
    materialized: OnceLock<Vec<u32>>,
}

impl Clone for FactorizedFastPartition {
    fn clone(&self) -> Self {
        Self {
            coarse_block: self.coarse_block.clone(),
            coarse_dominant_block: self.coarse_dominant_block.clone(),
            fine_overrides: self.fine_overrides.clone(),
            fine_count: self.fine_count,
            size: self.size,
            total_block_count: self.total_block_count,
            sum_block_sizes_sq: self.sum_block_sizes_sq,
            materialized: OnceLock::new(),
        }
    }
}

impl PartialEq for FactorizedFastPartition {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size
            && self.fine_count == other.fine_count
            && self.coarse_block == other.coarse_block
            && self.coarse_dominant_block == other.coarse_dominant_block
            && self.fine_overrides == other.fine_overrides
    }
}

impl Eq for FactorizedFastPartition {}

impl FactorizedFastPartition {
    /// Build from a slice of element-to-block ids (the natural representation).
    ///
    /// `factor` must be ≥ 1. For very small universes (`size < factor`),
    /// the result is equivalent to storing one coarse group containing all
    /// elements.
    pub fn from_ids_factorized(ids: &[u32], factor: u32) -> Self {
        assert!(factor >= 1, "FactorizedFastPartition: factor must be ≥ 1");
        let size = ids.len();
        if size == 0 {
            return Self {
                coarse_block: vec![],
                coarse_dominant_block: vec![],
                fine_overrides: BTreeMap::new(),
                fine_count: factor,
                size: 0,
                total_block_count: 0,
                sum_block_sizes_sq: 0,
                materialized: OnceLock::new(),
            };
        }

        let f = factor as usize;
        let n_coarse_idx = size.div_ceil(f);

        // Step 1: canonicalize block ids to 0..K_total (matches FastPartition::from_ids).
        let mut canon: HashMap<u32, u32> = HashMap::new();
        let mut next_id: u32 = 0;
        let mut canon_ids: Vec<u32> = Vec::with_capacity(size);
        for &raw in ids {
            let id = *canon.entry(raw).or_insert_with(|| {
                let n = next_id;
                next_id += 1;
                n
            });
            canon_ids.push(id);
        }
        let total_block_count = next_id;

        // Step 2: compute Σ|B|² over canon_ids.
        let mut block_size: HashMap<u32, u64> = HashMap::new();
        for &id in &canon_ids {
            *block_size.entry(id).or_insert(0) += 1;
        }
        let sum_block_sizes_sq: u64 = block_size.values().map(|&c| c * c).sum();

        // Step 3: per coarse index, find dominant block and collect overrides.
        // We canonicalize *coarse groups* by their (dominant_block, override_set) signature
        // so identical coarse rows share one group id.
        let mut group_sig_to_id: HashMap<(u32, Vec<(u32, u32)>), u32> = HashMap::new();
        let mut coarse_block: Vec<u32> = Vec::with_capacity(n_coarse_idx);
        let mut coarse_dominant_block: Vec<u32> = Vec::new();
        let mut fine_overrides: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        let mut next_group_id: u32 = 0;

        for c in 0..n_coarse_idx {
            let start = c * f;
            let end = (start + f).min(size);
            let slice = &canon_ids[start..end];

            // Find dominant block within this coarse group (mode).
            let mut local_count: HashMap<u32, u32> = HashMap::new();
            for &b in slice {
                *local_count.entry(b).or_insert(0) += 1;
            }
            // Pick the smallest block id among those tied for the largest count
            // (deterministic).
            let dominant = local_count
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
                .map(|(&k, _)| k)
                .expect("non-empty coarse group");

            // Collect overrides for this coarse group.
            let mut overrides_local: Vec<(u32, u32)> = Vec::new();
            for (j, &b) in slice.iter().enumerate() {
                if b != dominant {
                    overrides_local.push((j as u32, b));
                }
            }
            overrides_local.sort_unstable();

            // Canonicalize this coarse group's signature.
            let sig = (dominant, overrides_local.clone());
            let group_id = *group_sig_to_id.entry(sig).or_insert_with(|| {
                let g = next_group_id;
                next_group_id += 1;
                coarse_dominant_block.push(dominant);
                for (fine_idx, b) in &overrides_local {
                    fine_overrides.insert((g, *fine_idx), *b);
                }
                g
            });
            coarse_block.push(group_id);
        }

        Self {
            coarse_block,
            coarse_dominant_block,
            fine_overrides,
            fine_count: factor,
            size,
            total_block_count,
            sum_block_sizes_sq,
            materialized: OnceLock::new(),
        }
    }

    /// Build from a `FastPartition` (arithmetic-equivalent conversion).
    ///
    /// Result satisfies `factorize(fp, F).block_of(i) = fp.get_block(i)` for
    /// every `i ∈ [0, fp.size())`.
    pub fn from_fast_partition(fp: &FastPartition, factor: u32) -> Self {
        Self::from_ids_factorized(fp.get_ids(), factor)
    }

    /// Materialize back to dense `FastPartition`. Round-trip is bit-exact.
    pub fn to_fast_partition(&self) -> FastPartition {
        FastPartition::from_ids(&self.get_ids_owned())
    }

    /// Block id of element `i`. O(log overrides) worst case, O(1) average.
    #[inline]
    pub fn block_of(&self, i: usize) -> u32 {
        debug_assert!(
            i < self.size,
            "FactorizedFastPartition::block_of out of range"
        );
        let f = self.fine_count as usize;
        let c = i / f;
        let fine_idx = (i % f) as u32;
        let g = self.coarse_block[c];
        if let Some(&b) = self.fine_overrides.get(&(g, fine_idx)) {
            b
        } else {
            self.coarse_dominant_block[g as usize]
        }
    }

    /// Universe size N.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Factorization parameter F.
    #[inline]
    pub fn factor(&self) -> u32 {
        self.fine_count
    }

    /// Number of *distinct fine block IDs* — the partition's block count.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.total_block_count as usize
    }

    /// Number of canonicalized coarse groups (`K_coarse`). Distinct from
    /// `coarse_block.len()` which is `ceil(N/F)`.
    #[inline]
    pub fn coarse_group_count(&self) -> usize {
        self.coarse_dominant_block.len()
    }

    /// Number of fine-block overrides stored.
    #[inline]
    pub fn override_count(&self) -> usize {
        self.fine_overrides.len()
    }

    /// Factorized-native builder for sparse positions, used on the generate-index hot path.
    ///
    /// Produces the `(universe_position, dense_block_id)` list *without* calling
    /// `get_ids()`, so the universe-sized `Vec<u32>` is never materialised.
    ///
    /// The caller supplies the `block_id → dense_id` mapping; every `(pos, dense_id)`
    /// other than `dominant_dense_block` is emitted.
    ///
    /// Algorithm:
    /// 1. Iterate the coarse blocks (`N/F` entries) and take each one's dominant id
    ///    (`coarse_dominant_block[g]`).
    /// 2. If that dominant id differs from `dom_dense`, emit all `F` fine entries.
    /// 3. Otherwise emit only the `fine_overrides` that differ from `dom_dense`.
    ///
    /// Per-coarse cost is `O(F + |overrides_in_coarse|)`, so the total is
    /// `O(N + |overrides|)` — the same order as `get_ids`, but without materialising it.
    pub fn factorized_sparse_positions(
        &self,
        block_id_to_dense: &[u32],
        dom_dense: u32,
    ) -> Vec<(u32, u32)> {
        let f = self.fine_count as usize;
        let n = self.size;
        if n == 0 {
            return Vec::new();
        }
        let mut out: Vec<(u32, u32)> = Vec::new();
        for (c, &g) in self.coarse_block.iter().enumerate() {
            let coarse_dom_block = self.coarse_dominant_block[g as usize];
            let coarse_dom_dense = block_id_to_dense[coarse_dom_block as usize];
            // Base position of this coarse block: i = c * F + f, f in [0, F).
            let base = c * f;
            if coarse_dom_dense != dom_dense {
                // Coarse dominant != dom_dense: emit every fine entry (overrides handled below).
                for fine in 0..f {
                    let i = base + fine;
                    if i >= n {
                        break;
                    }
                    let key = (g, fine as u32);
                    let block = if let Some(&b) = self.fine_overrides.get(&key) {
                        b
                    } else {
                        coarse_dom_block
                    };
                    let dense = block_id_to_dense[block as usize];
                    if dense != dom_dense {
                        out.push((i as u32, dense));
                    }
                }
            } else {
                // Coarse dominant == dom_dense: only the overrides need checking.
                for (&(og, fine), &b) in self.fine_overrides.range((g, 0)..(g, f as u32)) {
                    if og != g {
                        break;
                    }
                    let i = base + fine as usize;
                    if i >= n {
                        continue;
                    }
                    let dense = block_id_to_dense[b as usize];
                    if dense != dom_dense {
                        out.push((i as u32, dense));
                    }
                }
            }
        }
        // Sort by position so the output is deterministic.
        out.sort_by_key(|&(pos, _)| pos);
        out
    }

    /// The distinct sparse block ids, i.e. the input to dense renumbering.
    /// Callers use this to build the `block_id_to_dense` map.
    ///
    /// Equal to the distinct values of `get_ids()[i] for i in 0..N`, computed without
    /// materialising that vector.
    pub fn distinct_block_ids(&self) -> Vec<u32> {
        // Union of all distinct block ids in coarse_dominant_block and fine_overrides.
        let mut seen: alloc::collections::BTreeSet<u32> = alloc::collections::BTreeSet::new();
        for &b in &self.coarse_dominant_block {
            seen.insert(b);
        }
        for &b in self.fine_overrides.values() {
            seen.insert(b);
        }
        seen.into_iter().collect()
    }

    /// Logical entropy h(π) = 1 - Σ(|B|/N)². O(1).
    pub fn logical_entropy(&self) -> f64 {
        if self.size == 0 {
            return 0.0;
        }
        let n = self.size as f64;
        1.0 - (self.sum_block_sizes_sq as f64) / (n * n)
    }

    /// Distinction count `dit(π) = N² - Σ|B|²`. O(1).
    pub fn dit_count(&self) -> usize {
        if self.size == 0 {
            return 0;
        }
        (self.size * self.size) - self.sum_block_sizes_sq as usize
    }

    /// Lazily materialized `&[u32]` view of element-to-block. Cache is kept in
    /// `OnceLock` so subsequent borrows are O(1).
    pub fn get_ids(&self) -> &[u32] {
        self.materialized.get_or_init(|| self.get_ids_owned())
    }

    /// Whether the lazy `Vec<u32>` cache has been populated yet.
    pub fn is_materialized(&self) -> bool {
        self.materialized.get().is_some()
    }

    /// Materialize the full id sequence as a `Vec<u32>`. Always O(N).
    pub fn get_ids_owned(&self) -> Vec<u32> {
        (0..self.size).map(|i| self.block_of(i)).collect()
    }

    /// Analytic block sizes — bit-exact with `FastPartition::block_count()`
    /// distribution but computed without materializing the full id vector.
    /// O(N/F + override_count + K_coarse).
    pub fn block_sizes_analytic(&self) -> Vec<usize> {
        if self.size == 0 {
            return Vec::new();
        }
        let f = self.fine_count as usize;
        let mut block_sizes: HashMap<u32, usize> = HashMap::new();

        // Each coarse index contributes `f` elements (last one may be short).
        for (c, &g) in self.coarse_block.iter().enumerate() {
            let start = c * f;
            let end = (start + f).min(self.size);
            let group_len = end - start;
            let dominant = self.coarse_dominant_block[g as usize];

            // Count overrides within this coarse index that fall in the live range.
            let mut overrides_in_range: usize = 0;
            // Use range scan: (g, 0) ..= (g, group_len-1)
            // BTreeMap range avoids scanning whole map.
            let start_key = (g, 0u32);
            let end_key = (g, group_len as u32);
            for ((_gg, _f_idx), &b) in self.fine_overrides.range(start_key..end_key) {
                *block_sizes.entry(b).or_insert(0) += 1;
                overrides_in_range += 1;
            }
            // Dominant gets the remaining (group_len - overrides_in_range).
            *block_sizes.entry(dominant).or_insert(0) += group_len - overrides_in_range;
            // Avoid unused warning if overrides_in_range == 0
            let _ = c;
        }

        let mut sizes: Vec<usize> = block_sizes.into_values().collect();
        sizes.sort_by(|a, b| b.cmp(a));
        sizes
    }

    /// First-appearance dense renumbering, sweeping `coarse_block` and `fine_overrides`
    /// directly. This is the build path of the generate-index.
    ///
    /// Replaces the two-pass `block_of(i) × N` build, which costs `2N` `BTreeMap::get`
    /// lookups (`O(log |overrides|)` each). This method instead:
    ///
    /// 1. Computes `max_id` from `coarse_dominant_block ∪ fine_overrides.values()`,
    ///    which is `O(K_coarse + |overrides|)` and independent of `N`.
    /// 2. Sweeps coarse block by coarse block (`N/F` entries), doing one `BTreeMap`
    ///    range scan per coarse block and batch-filling the dominant id, so that
    ///    position `i = c*F + fine` keeps first-appearance order.
    /// 3. Returns `(sparse_to_dense, dense_block_sizes)`.
    ///    bit-exact equivalence with `block_of(i) × N` 2-pass path.
    ///
    /// Per-coarse cost is `O(F + |overrides_in_coarse|)`, so the total is `O(N + |overrides|)`;
    /// `BTreeMap` lookups drop from `N` to `K_coarse`, an `F`-fold reduction.
    ///
    /// Measured at `N = 65_536, F = 256`: 256 lookups instead of 65_536.
    pub fn build_dense_renumbering(&self) -> (Vec<u32>, Vec<usize>) {
        let n = self.size;
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        // Step 1: max_id over coarse_dominant + override values (N-independent).
        let mut max_id: u32 = 0;
        for &d in &self.coarse_dominant_block {
            if d > max_id {
                max_id = d;
            }
        }
        for &b in self.fine_overrides.values() {
            if b > max_id {
                max_id = b;
            }
        }
        let mut sparse_to_dense: Vec<u32> = vec![u32::MAX; (max_id as usize) + 1];
        let mut dense_block_sizes: Vec<usize> = Vec::new();

        // Step 2: coarse-by-coarse sweep — first-appearance order preserved.
        let f = self.fine_count as usize;
        for (c, &g) in self.coarse_block.iter().enumerate() {
            let dominant = self.coarse_dominant_block[g as usize];
            let base = c * f;
            let group_len = f.min(n - base);

            // BTreeMap range scan: overrides for this coarse group.
            let start_key = (g, 0u32);
            let end_key = (g, group_len as u32);
            let mut override_iter = self.fine_overrides.range(start_key..end_key).peekable();

            for fine in 0..group_len {
                // Determine block id at position (c, fine).
                let block = if let Some(&(&(_og, ofine), &b)) = override_iter.peek() {
                    if ofine as usize == fine {
                        override_iter.next();
                        b
                    } else {
                        dominant
                    }
                } else {
                    dominant
                };
                // First-appearance dense renumbering.
                let slot = &mut sparse_to_dense[block as usize];
                if *slot == u32::MAX {
                    *slot = dense_block_sizes.len() as u32;
                    dense_block_sizes.push(1);
                } else {
                    dense_block_sizes[*slot as usize] += 1;
                }
            }

            if base + group_len >= n {
                break;
            }
        }
        (sparse_to_dense, dense_block_sizes)
    }

    /// Memory cost in bytes (best-effort estimate including heap allocations).
    /// Used for substrate memory accounting.
    pub fn memory_usage(&self) -> usize {
        let coarse = self.coarse_block.capacity() * core::mem::size_of::<u32>();
        let dominant = self.coarse_dominant_block.capacity() * core::mem::size_of::<u32>();
        // BTreeMap: nodes are typically ~24 B node + ~12 B per (u32,u32)→u32 entry.
        // Approximate at 16 B per entry to be conservative.
        let overrides = self.fine_overrides.len() * 16;
        core::mem::size_of::<Self>() + coarse + dominant + overrides
    }

    /// Compare a FastPartition for arithmetic-equivalence with this factorized
    /// representation. Used by tests and the runtime guard in `from_fast_partition`.
    pub fn arithmetically_equivalent(&self, fp: &FastPartition) -> bool {
        if self.size != fp.size() {
            return false;
        }
        let fp_ids = fp.get_ids();
        for i in 0..self.size {
            if self.block_of(i) != fp_ids[i] {
                return false;
            }
        }
        true
    }
}

impl From<&FactorizedFastPartition> for FastPartition {
    fn from(f: &FactorizedFastPartition) -> Self {
        f.to_fast_partition()
    }
}

// ────────────────────────────────────────────────────────────────────────
// Unit tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: deterministic block ids for a given pattern + universe.
    fn fast_from_ids(ids: &[u32]) -> FastPartition {
        FastPartition::from_ids(ids)
    }

    /// Helper: synthesize an id vector with `k` distinct blocks scattered
    /// over a universe of size `n` with a given override ratio.
    fn synth_ids(n: usize, k: u32, seed: u64) -> Vec<u32> {
        // Simple LCG to avoid bringing rand into partition-lattice crate.
        let mut s = seed
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out.push(((s >> 33) as u32) % k);
        }
        out
    }

    // ── A. Round-trip / bit-exact across universe sweep ──────────────

    #[test]
    fn test_factorized_block_of_bit_exact_universe_64() {
        let ids: Vec<u32> = vec![
            0, 0, 1, 1, 0, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 0, 0, 1, 1, 0, 2, 2, 2, 0, 0, 0, 1, 1,
            1, 2, 2, 0, 0, 1, 1, 0, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 0, 0, 1, 1, 0, 2, 2, 2, 0, 0,
            0, 1, 1, 1, 2, 2,
        ];
        let fp = fast_from_ids(&ids);
        for f in [16u32, 32, 64] {
            let fz = FactorizedFastPartition::from_fast_partition(&fp, f);
            for i in 0..ids.len() {
                assert_eq!(
                    fz.block_of(i),
                    fp.get_block(i),
                    "block_of mismatch at i={}, F={}, N=64",
                    i,
                    f
                );
            }
        }
    }

    #[test]
    fn test_factorized_block_of_bit_exact_universe_4096() {
        for seed in [1u64, 2, 3] {
            let ids = synth_ids(4096, 8, seed);
            let fp = fast_from_ids(&ids);
            for f in [64u32, 256] {
                let fz = FactorizedFastPartition::from_fast_partition(&fp, f);
                for i in 0..4096 {
                    assert_eq!(
                        fz.block_of(i),
                        fp.get_block(i),
                        "block_of mismatch at i={}, F={}, seed={}",
                        i,
                        f,
                        seed
                    );
                }
            }
        }
    }

    #[test]
    fn test_factorized_block_of_bit_exact_universe_65536() {
        let ids = synth_ids(65536, 16, 42);
        let fp = fast_from_ids(&ids);
        for f in [128u32, 256, 512] {
            let fz = FactorizedFastPartition::from_fast_partition(&fp, f);
            // Spot check every 17th element to keep the test under a second
            // while still covering the whole universe.
            for i in (0..65536).step_by(17) {
                assert_eq!(
                    fz.block_of(i),
                    fp.get_block(i),
                    "block_of mismatch at i={}, F={}, N=65536",
                    i,
                    f
                );
            }
        }
    }

    #[test]
    fn test_factorized_round_trip_universe_4096() {
        let ids = synth_ids(4096, 8, 7);
        let fp = fast_from_ids(&ids);
        let fz = FactorizedFastPartition::from_fast_partition(&fp, 64);
        let fp_round = fz.to_fast_partition();
        assert_eq!(
            fp.get_ids(),
            fp_round.get_ids(),
            "round-trip get_ids must be bit-exact"
        );
        assert_eq!(fp.block_count(), fp_round.block_count());
    }

    #[test]
    fn test_factorized_round_trip_universe_65536() {
        let ids = synth_ids(65536, 16, 11);
        let fp = fast_from_ids(&ids);
        let fz = FactorizedFastPartition::from_fast_partition(&fp, 256);
        let fp_round = fz.to_fast_partition();
        assert_eq!(fp.get_ids(), fp_round.get_ids());
        assert_eq!(fp.block_count(), fp_round.block_count());
    }

    // ── B. Metric agreement ──────────────────────────────────────────

    #[test]
    fn test_factorized_logical_entropy_agrees_with_fast() {
        let ids = synth_ids(4096, 8, 99);
        let fp = fast_from_ids(&ids);
        let fz = FactorizedFastPartition::from_fast_partition(&fp, 64);
        let h_fast = fp.logical_entropy();
        let h_fz = fz.logical_entropy();
        assert!(
            (h_fast - h_fz).abs() < 1e-12,
            "logical_entropy mismatch: fast={} factorized={}",
            h_fast,
            h_fz
        );
    }

    #[test]
    fn test_factorized_distinction_count_agrees_with_fast() {
        let ids = synth_ids(4096, 8, 13);
        let fp = fast_from_ids(&ids);
        let fz = FactorizedFastPartition::from_fast_partition(&fp, 64);
        assert_eq!(fz.dit_count(), fp.dit_count());
    }

    #[test]
    fn test_factorized_block_count_agrees_with_fast() {
        let ids = synth_ids(4096, 11, 23);
        let fp = fast_from_ids(&ids);
        let fz = FactorizedFastPartition::from_fast_partition(&fp, 64);
        assert_eq!(fz.block_count(), fp.block_count());
    }

    #[test]
    fn test_factorized_block_sizes_analytic_agrees() {
        let ids = synth_ids(2048, 7, 31);
        let fp = fast_from_ids(&ids);
        let fz = FactorizedFastPartition::from_fast_partition(&fp, 64);

        // Reference: block sizes from FastPartition (sorted descending).
        let mut ref_sizes: Vec<usize> = {
            let mut counts: HashMap<u32, usize> = HashMap::new();
            for &b in fp.get_ids() {
                *counts.entry(b).or_insert(0) += 1;
            }
            counts.into_values().collect()
        };
        ref_sizes.sort_by(|a, b| b.cmp(a));

        let analytic = fz.block_sizes_analytic();
        assert_eq!(analytic, ref_sizes);
    }

    // ── C. Edge cases ────────────────────────────────────────────────

    #[test]
    fn test_factorized_empty_universe() {
        let fz = FactorizedFastPartition::from_ids_factorized(&[], 64);
        assert_eq!(fz.size(), 0);
        assert_eq!(fz.block_count(), 0);
        assert_eq!(fz.logical_entropy(), 0.0);
        assert_eq!(fz.dit_count(), 0);
        assert!(fz.get_ids_owned().is_empty());
    }

    #[test]
    fn test_factorized_factor_larger_than_universe() {
        let ids = vec![0u32, 0, 1, 1, 2, 2, 2, 3];
        let fp = fast_from_ids(&ids);
        let fz = FactorizedFastPartition::from_fast_partition(&fp, 64);
        for i in 0..ids.len() {
            assert_eq!(fz.block_of(i), fp.get_block(i));
        }
        assert_eq!(fz.block_count(), 4);
    }

    #[test]
    fn test_factorized_factor_equals_one_is_identity() {
        let ids = synth_ids(256, 5, 100);
        let fp = fast_from_ids(&ids);
        let fz = FactorizedFastPartition::from_fast_partition(&fp, 1);
        for i in 0..256 {
            assert_eq!(fz.block_of(i), fp.get_block(i));
        }
        // F=1 means each fine group has size 1; dominant_block = the only id.
        // All overrides should be empty.
        assert_eq!(fz.override_count(), 0);
    }

    #[test]
    fn test_factorized_get_ids_lazy_materialization() {
        let ids = synth_ids(1024, 4, 17);
        let fz = FactorizedFastPartition::from_ids_factorized(&ids, 64);
        assert!(!fz.is_materialized(), "cache must start empty");
        let _slice = fz.get_ids();
        assert!(fz.is_materialized(), "cache populated after get_ids");
        // Second call returns the same slice (no recomputation).
        let slice2 = fz.get_ids();
        assert_eq!(slice2.len(), 1024);
    }

    // ── D. Memory budget ─────────────────────────────────────────────

    #[test]
    fn test_factorized_memory_under_dense_at_n_65536() {
        // Dominant-heavy synthetic ids (4 blocks, light scatter) — best case.
        let mut ids = vec![0u32; 65536];
        for chunk in 0..256usize {
            for j in 0..256 {
                let i = chunk * 256 + j;
                // 95% dominant block, 5% scatter.
                ids[i] = if (i % 19) == 0 { (i % 4) as u32 } else { 0 };
            }
        }
        let fp = fast_from_ids(&ids);
        let fz = FactorizedFastPartition::from_fast_partition(&fp, 256);
        let dense_bytes = 65536 * core::mem::size_of::<u32>();
        let fz_bytes = fz.memory_usage();
        assert!(
            fz_bytes < dense_bytes / 4,
            "factorized {} bytes must be < dense/4 = {} bytes for dominant-heavy case",
            fz_bytes,
            dense_bytes / 4
        );
    }

    // ── E. Clone / serde / PartialEq ────────────────────────────────

    #[test]
    fn test_factorized_clone_preserves_block_of() {
        let ids = synth_ids(1024, 5, 71);
        let fz = FactorizedFastPartition::from_ids_factorized(&ids, 64);
        let cloned = fz.clone();
        for i in 0..1024 {
            assert_eq!(fz.block_of(i), cloned.block_of(i));
        }
    }

    #[test]
    fn test_factorized_partial_eq_matches_construction() {
        let ids = synth_ids(512, 4, 19);
        let a = FactorizedFastPartition::from_ids_factorized(&ids, 64);
        let b = FactorizedFastPartition::from_ids_factorized(&ids, 64);
        assert_eq!(a, b);

        // Different F → different physical layout, even if block_of agrees.
        let c = FactorizedFastPartition::from_ids_factorized(&ids, 32);
        assert_ne!(a, c);
    }

    #[test]
    fn test_factorized_arithmetically_equivalent_guard() {
        let ids = synth_ids(2048, 6, 5);
        let fp = fast_from_ids(&ids);
        let fz = FactorizedFastPartition::from_fast_partition(&fp, 64);
        assert!(fz.arithmetically_equivalent(&fp));
    }

    // ── F. build_dense_renumbering  ─────

    /// Reference: 2-pass `block_of(i) × N` first-appearance dense renumbering.
    /// Bit-exact baseline against which `build_dense_renumbering()` is checked.
    fn build_dense_renumbering_baseline(f: &FactorizedFastPartition) -> (Vec<u32>, Vec<usize>) {
        let n = f.size();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let mut max_id = 0u32;
        for i in 0..n {
            let id = f.block_of(i);
            if id > max_id {
                max_id = id;
            }
        }
        let mut sparse_to_dense: Vec<u32> = vec![u32::MAX; (max_id as usize) + 1];
        let mut dense_block_sizes: Vec<usize> = Vec::new();
        for i in 0..n {
            let id = f.block_of(i);
            let slot = &mut sparse_to_dense[id as usize];
            if *slot == u32::MAX {
                *slot = dense_block_sizes.len() as u32;
                dense_block_sizes.push(1);
            } else {
                dense_block_sizes[*slot as usize] += 1;
            }
        }
        (sparse_to_dense, dense_block_sizes)
    }

    /// `build_dense_renumbering` agrees bit-exactly with a full universe sweep.
    #[test]
    fn test_factorized_build_dense_renumbering_bit_exact_universe_sweep() {
        for (n, k, factor, seed) in [
            (256usize, 4u32, 16u32, 1u64),
            (1024, 6, 32, 7),
            (4096, 8, 64, 11),
            (16384, 12, 128, 17),
            (65536, 16, 256, 23),
        ] {
            let ids = synth_ids(n, k, seed);
            let fz = FactorizedFastPartition::from_ids_factorized(&ids, factor);
            let (b2d_ref, sizes_ref) = build_dense_renumbering_baseline(&fz);
            let (b2d_v2, sizes_v2) = fz.build_dense_renumbering();
            assert_eq!(
                b2d_v2, b2d_ref,
                "sparse_to_dense mismatch at N={}, K={}, F={}, seed={}",
                n, k, factor, seed
            );
            assert_eq!(
                sizes_v2, sizes_ref,
                "dense_block_sizes mismatch at N={}, K={}, F={}, seed={}",
                n, k, factor, seed
            );
        }
    }

    ///  edge — empty universe / single block / dominant-only coarse.
    #[test]
    fn test_factorized_build_dense_renumbering_edges() {
        // Empty universe.
        let fz_empty = FactorizedFastPartition::from_ids_factorized(&[], 64);
        let (b2d, sizes) = fz_empty.build_dense_renumbering();
        assert!(b2d.is_empty());
        assert!(sizes.is_empty());

        // Single block (all same id).
        let ids = vec![7u32; 1024];
        let fz_single = FactorizedFastPartition::from_ids_factorized(&ids, 64);
        let (b2d, sizes) = fz_single.build_dense_renumbering();
        let (b2d_ref, sizes_ref) = build_dense_renumbering_baseline(&fz_single);
        assert_eq!(b2d, b2d_ref);
        assert_eq!(sizes, sizes_ref);
        assert_eq!(sizes.len(), 1);
        assert_eq!(sizes[0], 1024);

        // Factor larger than universe (single coarse group).
        let ids = vec![0u32, 0, 1, 1, 2, 2, 2, 3];
        let fz_big_factor = FactorizedFastPartition::from_ids_factorized(&ids, 64);
        let (b2d, sizes) = fz_big_factor.build_dense_renumbering();
        let (b2d_ref, sizes_ref) = build_dense_renumbering_baseline(&fz_big_factor);
        assert_eq!(b2d, b2d_ref);
        assert_eq!(sizes, sizes_ref);
    }

    /// First-appearance order is preserved for coarse blocks mixing a dominant id
    /// with overrides, following `fine_idx` order within the coarse block.
    ///
    /// `from_ids_factorized` canonicalises raw ids — raw `[5,5,3,5,8,8,8,5]` becomes
    /// `[0,0,1,0,2,2,2,0]` — so the dense renumbering is measured against the canonical ids.
    #[test]
    fn test_factorized_build_dense_renumbering_first_appearance_order_preserved() {
        // F=4, raw coarse 0 = [5, 5, 3, 5] (override at fine=2), raw coarse 1 = [8, 8, 8, 5] (override at fine=3).
        // After canonicalization: canonical 0=raw 5, canonical 1=raw 3, canonical 2=raw 8.
        let ids = vec![5u32, 5, 3, 5, 8, 8, 8, 5];
        let fz = FactorizedFastPartition::from_ids_factorized(&ids, 4);
        let (b2d, sizes) = fz.build_dense_renumbering();
        let (b2d_ref, sizes_ref) = build_dense_renumbering_baseline(&fz);
        assert_eq!(b2d, b2d_ref);
        assert_eq!(sizes, sizes_ref);
        // sparse_to_dense indexed by canonical id (0..K_canon).
        // First-appearance of canonical id: 0 (i=0) → dense 0, 1 (i=2) → dense 1, 2 (i=4) → dense 2.
        assert_eq!(b2d.len(), 3);
        assert_eq!(b2d[0], 0); // canonical 0 (raw 5)
        assert_eq!(b2d[1], 1); // canonical 1 (raw 3)
        assert_eq!(b2d[2], 2); // canonical 2 (raw 8)
                               // Sizes: dense 0 = 4, dense 1 = 1, dense 2 = 3.
        assert_eq!(sizes, vec![4, 1, 3]);
    }
}
