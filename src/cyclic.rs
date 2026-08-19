//! Cyclic Partition representation for ngram-style implicit patterns.
//!
//! Cyclic crystals: n-gram-style patterns follow the
//! pattern `ids[i] = chars[i % cycle_len]` over a universe of size N. The
//! dense `FastPartition` materializes a `Vec<u32>` of length N (≈ 256 KB
//! at N=65536). For ngram crystals where `cycle_len ≤ 8`, the entire
//! partition is fully determined by the pattern itself — a 40-byte
//! struct, **independent of N**.
//!
//! ## Shared substrate (macro-generated)
//!
//! A single declarative macro (`cyclic_partition_impl`) generates both
//! production types from one method body:
//!
//! - [`CyclicPartition`] (pattern `[u32; `[`MAX_CYCLE`]` = 8]`) — narrow form
//!   (~40 B), SFM ngram hot path (`cycle_len ≤ 8`).
//! - [`WideCyclicPartition`] (pattern `[u32; `[`MAX_CYCLE_WIDE`]` = 16]`) —
//!   extended form (~72 B), Layer A substrate for `cycle_len ∈ [9, 16]`
//!   (wide-cyclic residual routing).
//!
//! Prior to 2026-05, these were two hand-duplicated structs (`cyclic.rs` +
//! `cyclic_wide.rs`, ~87% identical). They now share one macro body, keeping
//! each struct's concrete `[u32; N]` array so the `serde` byte format is
//! unchanged. [`crate::cyclic_wide`] is a thin compatibility re-export that
//! retains the L ∈ [9, 16] verification sweep.
//!
//! ## Memory cost
//!
//! | representation                | bytes at N=4096 | bytes at N=65536 |
//! |---|---|---|
//! | `FastPartition`               | ~16 KB  | ~275 KB |
//! | `SparsePartition`             | ~16 KB  | ~290 KB |
//! | `CyclicPartition` (L=8)       | **40 B** | **40 B** |
//! | `WideCyclicPartition` (L=16)  | **~72 B** | **~72 B** |
//!
//! ## When to use
//!
//! - **YES** — ngram crystals with `cycle_len ≤ 8` (SFM build hot path)
//! - **NO** — arbitrary partitions, large cycle_len, or after a Meet that
//!   produces a non-cyclic pattern (convert to `FastPartition` via
//!   `to_fast_partition()` in those cases)
//!
//! ## Correctness — bit-exact with FastPartition
//!
//! For any pattern `p[]` of length `L' ≤ L` and any universe size `N`:
//!
//! ```text
//!   cyclic = CyclicPartitionN::<L>::from_pattern(p, N)
//!   fast   = FastPartition::from_ids(&(0..N).map(|i| p[i % L']).collect())
//!
//!   ⟹  cyclic.get_ids_owned()    == fast.get_ids()
//!   ⟹  cyclic.logical_entropy()  == fast.logical_entropy()
//!   ⟹  cyclic.distinction_count() == fast.distinction_count()
//!   ⟹  cyclic.block_count()      == fast.block_count()
//! ```
//!
//! Verified in `tests::test_cyclic_bit_exact_with_fast_partition` over a
//! 48-case sweep (12 patterns × 4 universe sizes) for the narrow form, and
//! in [`crate::cyclic_wide`] for the extended L ∈ [9, 16] range.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::FloatExt;
use crate::OnceLock;
#[allow(unused_imports)]
use alloc::borrow::ToOwned;
#[allow(unused_imports)]
use alloc::boxed::Box;
#[allow(unused_imports)]
use alloc::string::{String, ToString};
#[allow(unused_imports)]
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::optimized::FastPartition;

/// Maximum cycle_len for the narrow [`CyclicPartition`]. SFM ngram_partition
/// uses `max_ngram_len ≤ 8`, so patterns of length up to 8 cover the full
/// hot path.
pub const MAX_CYCLE: usize = 8;

/// Wide upper bound for [`WideCyclicPartition::cycle_len`]. Doubles the
/// narrow [`MAX_CYCLE`] = 8.
pub const MAX_CYCLE_WIDE: usize = 16;

/// Generates a fixed-capacity cyclic-partition type whose pattern array is
/// `[u32; $cap]`. Both [`CyclicPartition`] (`$cap = MAX_CYCLE`) and
/// [`WideCyclicPartition`] (`$cap = MAX_CYCLE_WIDE`) are emitted from this one
/// body. Keeping the array size concrete (rather than a const generic) means
/// the `serde` derive applies its standard fixed-array impl, so the
/// serialized byte format is identical to the pre-unification structs.
macro_rules! cyclic_partition_impl {
    ($(#[$meta:meta])* $name:ident, $cap:expr) => {
        $(#[$meta])*
        #[derive(Debug, Serialize, Deserialize)]
        pub struct $name {
            /// Cyclic block-id pattern. Only the first `cycle_len` entries are valid.
            pattern: [u32; $cap],
            /// Active length of `pattern`, in [1, `$cap`].
            cycle_len: u8,
            /// Universe size N.
            size: u32,
            /// Lazy materialized id vector for `get_ids() -> &[u32]` compatibility.
            /// Populated on first read; reset on Clone. Not serialized.
            #[serde(skip)]
            materialized: OnceLock<Vec<u32>>,
        }

        impl Clone for $name {
            fn clone(&self) -> Self {
                Self {
                    pattern: self.pattern,
                    cycle_len: self.cycle_len,
                    size: self.size,
                    materialized: OnceLock::new(), // reset cache on clone
                }
            }
        }

        impl PartialEq for $name {
            /// Two cyclic partitions are equal iff they have the same
            /// canonicalized pattern. O(cap) — no materialization.
            fn eq(&self, other: &Self) -> bool {
                self.size == other.size
                    && self.cycle_len == other.cycle_len
                    && self.pattern[..self.cycle_len as usize]
                        == other.pattern[..other.cycle_len as usize]
            }
        }

        impl Eq for $name {}

        impl $name {
            /// Maximum cycle length this type can hold (the pattern capacity).
            pub const MAX_CYCLE_LEN: usize = $cap;

            /// Construct from a cyclic pattern. Pattern values are canonicalized
            /// to first-appearance block IDs (0, 1, 2, ...) to match
            /// `FastPartition::from_ids` semantics — partition labels are
            /// arbitrary, only the equivalence classes are meaningful.
            ///
            /// Example: `from_pattern(&[97, 98, 99], N)` stores `[0, 1, 2]`.
            /// `from_pattern(&[1, 0], N)` stores `[0, 1]`.
            ///
            /// # Panics
            /// - If `pattern.len() == 0` and `size > 0` (cannot index empty pattern)
            /// - If `pattern.len() > MAX_CYCLE_LEN`
            pub fn from_pattern(pattern: &[u32], size: usize) -> Self {
                assert!(
                    pattern.len() <= $cap,
                    concat!(stringify!($name), ": cycle_len {} exceeds capacity {}"),
                    pattern.len(), $cap,
                );
                if size > 0 {
                    assert!(!pattern.is_empty(),
                        concat!(stringify!($name), ": pattern empty but size > 0"));
                }

                // Canonicalize: map raw values → 0, 1, 2, ... in first-appearance
                // order so partition labels match FastPartition.
                let mut arr = [0u32; $cap];
                let mut seen_raw = [0u32; $cap];
                let mut seen_canon: u32 = 0;
                for (i, &raw) in pattern.iter().enumerate() {
                    // Lookup raw → canonical (linear scan over already-seen, at
                    // most `cap` entries → O(cap) per insert).
                    let mut canon = None;
                    for k in 0..(seen_canon as usize) {
                        if seen_raw[k] == raw {
                            canon = Some(k as u32);
                            break;
                        }
                    }
                    arr[i] = match canon {
                        Some(c) => c,
                        None => {
                            let c = seen_canon;
                            seen_raw[seen_canon as usize] = raw;
                            seen_canon += 1;
                            c
                        }
                    };
                }
                Self {
                    pattern: arr,
                    cycle_len: pattern.len() as u8,
                    size: size as u32,
                    materialized: OnceLock::new(),
                }
            }

            /// Indiscrete partition (all elements in block 0).
            pub fn indiscrete(size: usize) -> Self {
                Self::from_pattern(&[0u32], size)
            }

            /// Lazy materialized id slice. First call computes + caches;
            /// subsequent calls return the cached slice. Tied to `&self` lifetime.
            pub fn get_ids(&self) -> &[u32] {
                self.materialized.get_or_init(|| self.get_ids_owned())
            }

            /// Whether the lazy `Vec<u32>` cache has been populated yet. Used by the
            /// wide-cyclic routing regression tests to confirm that the analytic path is
            /// taken when it should be.
            pub fn is_materialized(&self) -> bool {
                self.materialized.get().is_some()
            }

            /// Try to detect a cyclic pattern in an arbitrary id slice. Returns
            /// `Some(Self)` if `ids` is cyclic with `cycle_len ≤ MAX_CYCLE_LEN`,
            /// else `None`. Selects the *shortest* valid cycle.
            pub fn try_from_ids(ids: &[u32]) -> Option<Self> {
                let n = ids.len();
                if n == 0 {
                    return Some(Self::from_pattern(&[], 0));
                }
                // Find shortest cycle_len in [1, min(cap, n)] such that
                // ids[i] == ids[i % cycle_len] for all i.
                for cycle_len in 1..=$cap.min(n) {
                    let mut ok = true;
                    for i in cycle_len..n {
                        if ids[i] != ids[i % cycle_len] {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        return Some(Self::from_pattern(&ids[..cycle_len], n));
                    }
                }
                None
            }

            /// Universe size N.
            pub fn size(&self) -> usize {
                self.size as usize
            }

            /// Cycle length.
            pub fn cycle_len(&self) -> usize {
                self.cycle_len as usize
            }

            /// Raw pattern slice (only first `cycle_len` valid).
            pub fn pattern(&self) -> &[u32] {
                &self.pattern[..self.cycle_len as usize]
            }

            /// Block ID of element `i`. O(1).
            pub fn block_of(&self, element: usize) -> u32 {
                debug_assert!(element < self.size as usize);
                let cl = self.cycle_len as usize;
                self.pattern[element % cl]
            }

            /// Materialize the full id sequence as a `Vec<u32>`.
            pub fn get_ids_owned(&self) -> Vec<u32> {
                let n = self.size as usize;
                let cl = self.cycle_len as usize;
                (0..n).map(|i| self.pattern[i % cl]).collect()
            }

            /// Analytical block-size distribution (each block's element count).
            ///
            /// O(cycle_len). Used by `SpectralSignature::from_partition` to
            /// compute the signature without materializing the full id vector —
            /// keeps the substrate compact at Crystal::new time.
            pub fn block_sizes_analytic(&self) -> Vec<usize> {
                let n = self.size as usize;
                let cl = self.cycle_len as usize;
                if n == 0 || cl == 0 {
                    return Vec::new();
                }
                let pat = &self.pattern[..cl];
                let mut uniq_val = [0u32; $cap];
                let mut uniq_cnt = [0usize; $cap];
                let mut nuniq = 0;
                for (j, &v) in pat.iter().enumerate() {
                    let cnt = if j < n { (n - j + cl - 1) / cl } else { 0 };
                    let mut idx = None;
                    for k in 0..nuniq {
                        if uniq_val[k] == v {
                            idx = Some(k);
                            break;
                        }
                    }
                    match idx {
                        Some(k) => uniq_cnt[k] += cnt,
                        None => {
                            uniq_val[nuniq] = v;
                            uniq_cnt[nuniq] = cnt;
                            nuniq += 1;
                        }
                    }
                }
                uniq_cnt[..nuniq].to_vec()
            }

            /// Number of distinct blocks. O(cycle_len).
            ///
            /// Counts the distinct values appearing in `pattern[..cycle_len]`.
            /// For size 0, returns 0.
            pub fn block_count(&self) -> usize {
                if self.size == 0 {
                    return 0;
                }
                // Count unique values in pattern[..cycle_len].
                // pattern values are arbitrary u32 (could be high codepoints).
                let pat = self.pattern();
                let mut uniq = [0u32; $cap];
                let mut count = 0;
                for &v in pat {
                    let mut seen = false;
                    for u in &uniq[..count] {
                        if *u == v {
                            seen = true;
                            break;
                        }
                    }
                    if !seen {
                        uniq[count] = v;
                        count += 1;
                    }
                }
                count
            }

            /// Logical entropy `h(π) = 1 - Σ(|B|/N)²`.
            ///
            /// For cyclic patterns, each pattern position contributes
            /// ⌈N/cycle_len⌉ or ⌊N/cycle_len⌋ elements depending on
            /// `N mod cycle_len`. We compute exact block sizes and sum |B|².
            pub fn logical_entropy(&self) -> f64 {
                if self.size == 0 {
                    return 0.0;
                }
                let n = self.size as usize;
                let n_sq = (n as f64) * (n as f64);
                let sum_sq = self.sum_block_sizes_sq();
                1.0 - (sum_sq as f64 / n_sq)
            }

            /// `dit(π) = N² - Σ|B|²`. O(cycle_len) given `sum_block_sizes_sq`.
            pub fn distinction_count(&self) -> usize {
                if self.size == 0 {
                    return 0;
                }
                let n = self.size as usize;
                n * n - self.sum_block_sizes_sq()
            }

            /// Σ|B|² where B ranges over partition blocks.
            ///
            /// For cyclic pattern of length L over universe N: each distinct
            /// value `v` in the pattern is paired with positions
            /// `{i : pattern[i] = v}`. The block size for value v is:
            ///   |B_v| = Σ_{j ∈ positions(v)} ⌈(N - j) / L⌉
            /// (sum over positions `j` in [0, L) where pattern[j] = v)
            fn sum_block_sizes_sq(&self) -> usize {
                let n = self.size as usize;
                let cl = self.cycle_len as usize;
                if n == 0 || cl == 0 {
                    return 0;
                }
                // For each unique value in the pattern, sum its positions'
                // counts. count_at_position(j) = ⌈(n - j) / cl⌉ =
                // (n - j + cl - 1) / cl, for j in [0, cl).
                let pat = &self.pattern[..cl];

                // Build map: value -> total count (small inline scratch since
                // cl ≤ cap → at most `cap` distinct values).
                let mut uniq_val = [0u32; $cap];
                let mut uniq_cnt = [0usize; $cap];
                let mut nuniq = 0;

                for (j, &v) in pat.iter().enumerate() {
                    let cnt = if j < n { (n - j + cl - 1) / cl } else { 0 };
                    let mut idx = None;
                    for k in 0..nuniq {
                        if uniq_val[k] == v {
                            idx = Some(k);
                            break;
                        }
                    }
                    match idx {
                        Some(k) => uniq_cnt[k] += cnt,
                        None => {
                            uniq_val[nuniq] = v;
                            uniq_cnt[nuniq] = cnt;
                            nuniq += 1;
                        }
                    }
                }

                let mut sum: usize = 0;
                for k in 0..nuniq {
                    sum += uniq_cnt[k] * uniq_cnt[k];
                }
                sum
            }

            /// Memory cost in bytes (sizeof Self). Constant regardless of N.
            pub fn memory_usage(&self) -> usize {
                core::mem::size_of::<Self>()
            }

            /// Convert to dense `FastPartition` by materializing ids and feeding
            /// `FastPartition::from_ids`. Used when subsequent ops (Meet/Join
            /// with non-cyclic input) require the dense representation.
            pub fn to_fast_partition(&self) -> FastPartition {
                let ids = self.get_ids_owned();
                FastPartition::from_ids(&ids)
            }
        }

        impl From<&$name> for FastPartition {
            fn from(c: &$name) -> Self {
                c.to_fast_partition()
            }
        }
    };
}

cyclic_partition_impl!(
    /// A partition whose element-to-block assignment follows a cyclic pattern:
    /// `block_of(i) = pattern[i % cycle_len]`, with `cycle_len ≤ MAX_CYCLE = 8`.
    ///
    /// Memory: ~40 bytes (compact form), regardless of `|U|`. On first call to
    /// [`Self::get_ids`] (the `&[u32]` reference API), the partition lazily
    /// materializes a `Vec<u32>` of length `|U|` and caches it in [`OnceLock`]
    /// for subsequent borrows. Clone resets the cache.
    CyclicPartition,
    MAX_CYCLE
);

cyclic_partition_impl!(
    /// Cyclic partition with extended pattern length (`cycle_len ≤
    /// MAX_CYCLE_WIDE = 16`). Layer A substrate; ~72 bytes regardless of `|U|`.
    /// See [`crate::cyclic_wide`] for the extended-range verification sweep.
    WideCyclicPartition,
    MAX_CYCLE_WIDE
);

// ─── lcm Meet closure ──────────────────────────────────────────────────────
//
// Lattice identity: Cyclic(L1) ∧ Cyclic(L2) == Cyclic(lcm(L1, L2)).
// This keeps the Cyclic representation while lcm <= MAX_CYCLE and returns None otherwise,
// so the caller can fall back to FastPartition::differentiate.

#[inline]
fn gcd_u(a: usize, b: usize) -> usize {
    let mut a = a;
    let mut b = b;
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[inline]
fn lcm_u(a: usize, b: usize) -> usize {
    if a == 0 || b == 0 {
        0
    } else {
        a / gcd_u(a, b) * b
    }
}

/// Meet (`differentiate`, ∧) of two narrow cyclic partitions.
///
/// Returns `Some(CyclicPartition)` of cycle length `lcm(c1.cycle_len, c2.cycle_len)`
/// **when that lcm ≤ MAX_CYCLE**, preserving the compact substrate. Returns `None`
/// when the lcm exceeds the capacity (caller may fall back to
/// `c1.to_fast_partition().differentiate(&c2.to_fast_partition())`).
///
/// The Meet of two cyclic partitions is cyclic with the lcm period — a sub-closure of the
/// lattice. Bit-exact with `FastPartition::differentiate` on the same input (verified by
/// `tests::test_cyclic_meet_lcm_bit_exact`).
///
/// # Panics
/// - If `c1.size() != c2.size()`.
pub fn cyclic_refine(c1: &CyclicPartition, c2: &CyclicPartition) -> Option<CyclicPartition> {
    assert_eq!(
        c1.size(),
        c2.size(),
        "cyclic_refine: partition sizes must match ({} vs {})",
        c1.size(),
        c2.size()
    );
    let l1 = c1.cycle_len();
    let l2 = c2.cycle_len();
    let lcm = lcm_u(l1, l2);
    if lcm == 0 || lcm > MAX_CYCLE {
        return None;
    }
    // Encode (a, b) pair as a single u32. Both a and b are canonicalized into
    // [0, MAX_CYCLE), so `a * MAX_CYCLE + b` gives a unique key per pair.
    let mut combined: Vec<u32> = Vec::with_capacity(lcm);
    for i in 0..lcm {
        let a = c1.pattern[i % l1];
        let b = c2.pattern[i % l2];
        combined.push(a * (MAX_CYCLE as u32) + b);
    }
    // from_pattern re-canonicalizes via first-appearance, so labels stay irrelevant.
    Some(CyclicPartition::from_pattern(&combined, c1.size()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fast_from_pattern(pattern: &[u32], size: usize) -> FastPartition {
        let cl = pattern.len();
        let ids: Vec<u32> = (0..size).map(|i| pattern[i % cl]).collect();
        FastPartition::from_ids(&ids)
    }

    /// C1 — Memory cost is constant and ≤ 128 bytes.
    ///
    /// Compact representation (~40B) + OnceLock<Vec<u32>> slot (~32B) for
    /// lazy `get_ids()` materialization. Total ~72B. At M scale (N=65536)
    /// dense FastPartition is ~256 KB → cyclic substrate saves ≥ 3600×.
    /// After first get_ids() call, the materialized Vec<u32> adds N·4 bytes,
    /// but the substrate's compact-at-rest property holds before that.
    #[test]
    fn test_cyclic_memory_under_128_bytes() {
        let small = CyclicPartition::from_pattern(&[0, 1, 2], 4096);
        let large = CyclicPartition::from_pattern(&[0, 1, 2], 65536);
        assert_eq!(
            small.memory_usage(),
            large.memory_usage(),
            "CyclicPartition compact-form memory MUST be independent of |U|"
        );
        assert!(
            small.memory_usage() <= 128,
            "CyclicPartition compact memory ({}) must be ≤ 128 bytes",
            small.memory_usage()
        );
    }

    /// C2 — get_ids_owned is bit-exact with FastPartition::from_ids over a
    /// 12 patterns × 4 universe sizes = 48-case sweep.
    #[test]
    fn test_cyclic_bit_exact_get_ids_with_fast_partition() {
        let patterns: Vec<&[u32]> = vec![
            &[0],
            &[0, 1],
            &[1, 0],
            &[0, 1, 2],
            &[97, 98, 99], // ngram "abc" raw codepoints
            &[0, 0, 1],
            &[5, 5, 5, 5],
            &[0, 1, 2, 3],
            &[10, 20, 10, 20],
            &[0, 1, 2, 0, 1],
            &[7, 7, 7, 8],
            &[1, 2, 3, 4, 5, 6, 7, 8], // max cycle_len = 8
        ];
        let universes = [4096usize, 8192, 16384, 65536];

        for pattern in &patterns {
            for &n in &universes {
                let cyclic = CyclicPartition::from_pattern(pattern, n);
                let fast = fast_from_pattern(pattern, n);

                let cy_ids = cyclic.get_ids_owned();
                let fast_ids = fast.get_ids();

                assert_eq!(
                    cy_ids.len(),
                    fast_ids.len(),
                    "ids length mismatch for pattern {:?} at N={}",
                    pattern,
                    n
                );
                assert_eq!(
                    cy_ids.as_slice(),
                    fast_ids,
                    "get_ids byte-exact MISMATCH for pattern {:?} at N={}",
                    pattern,
                    n
                );
            }
        }
    }

    /// C2 — logical_entropy is bit-exact with FastPartition.
    #[test]
    fn test_cyclic_bit_exact_logical_entropy_with_fast_partition() {
        let patterns: Vec<&[u32]> = vec![
            &[0],
            &[0, 1],
            &[1, 0],
            &[0, 1, 2],
            &[97, 98, 99],
            &[0, 0, 1],
            &[5, 5, 5, 5],
            &[0, 1, 2, 3],
            &[10, 20, 10, 20],
            &[0, 1, 2, 0, 1],
            &[7, 7, 7, 8],
            &[1, 2, 3, 4, 5, 6, 7, 8],
        ];
        let universes = [4096usize, 8192, 16384, 65536];

        for pattern in &patterns {
            for &n in &universes {
                let cyclic = CyclicPartition::from_pattern(pattern, n);
                let fast = fast_from_pattern(pattern, n);

                let h_cyclic = cyclic.logical_entropy();
                let h_fast = fast.logical_entropy();

                assert!(
                    (h_cyclic - h_fast).abs() < 1e-12,
                    "entropy MISMATCH for pattern {:?} at N={}: cyclic={} fast={}",
                    pattern,
                    n,
                    h_cyclic,
                    h_fast
                );
            }
        }
    }

    /// C2 — distinction_count is bit-exact with FastPartition.
    #[test]
    fn test_cyclic_bit_exact_distinction_count_with_fast_partition() {
        let patterns: Vec<&[u32]> = vec![
            &[0],
            &[0, 1],
            &[0, 1, 2],
            &[97, 98, 99],
            &[0, 0, 1],
            &[5, 5, 5, 5],
            &[10, 20, 10, 20],
            &[0, 1, 2, 0, 1],
        ];
        let universes = [4096usize, 8192, 16384, 65536];

        for pattern in &patterns {
            for &n in &universes {
                let cyclic = CyclicPartition::from_pattern(pattern, n);
                let fast = fast_from_pattern(pattern, n);

                assert_eq!(
                    cyclic.distinction_count(),
                    fast.distinction_count(),
                    "distinction_count MISMATCH for pattern {:?} at N={}",
                    pattern,
                    n
                );
            }
        }
    }

    /// C2 — block_count is bit-exact with FastPartition.
    #[test]
    fn test_cyclic_bit_exact_block_count_with_fast_partition() {
        let patterns: Vec<&[u32]> = vec![
            &[0],
            &[0, 1],
            &[0, 1, 2],
            &[97, 98, 99],
            &[5, 5, 5, 5],
            &[0, 1, 2, 3],
            &[10, 20, 10, 20],
            &[0, 1, 2, 0, 1],
        ];
        let universes = [4096usize, 8192, 16384, 65536];

        for pattern in &patterns {
            for &n in &universes {
                let cyclic = CyclicPartition::from_pattern(pattern, n);
                let fast = fast_from_pattern(pattern, n);

                assert_eq!(
                    cyclic.block_count(),
                    fast.block_count(),
                    "block_count MISMATCH for pattern {:?} at N={}: cyclic={} fast={}",
                    pattern,
                    n,
                    cyclic.block_count(),
                    fast.block_count()
                );
            }
        }
    }

    /// C3 — Determinism: same input → same output (100 trials).
    #[test]
    fn test_cyclic_deterministic_100_trials() {
        let baseline_ids = CyclicPartition::from_pattern(&[1, 2, 3], 1000).get_ids_owned();
        let baseline_h = CyclicPartition::from_pattern(&[1, 2, 3], 1000).logical_entropy();
        for _ in 0..100 {
            let c = CyclicPartition::from_pattern(&[1, 2, 3], 1000);
            assert_eq!(c.get_ids_owned(), baseline_ids);
            assert!((c.logical_entropy() - baseline_h).abs() < 1e-12);
        }
    }

    /// block_of: O(1) virtual access matches the materialized id.
    #[test]
    fn test_cyclic_block_of_matches_get_ids() {
        let pattern = vec![10, 20, 30, 40];
        let n = 100;
        let c = CyclicPartition::from_pattern(&pattern, n);
        let ids = c.get_ids_owned();
        for i in 0..n {
            assert_eq!(c.block_of(i), ids[i]);
        }
    }

    /// try_from_ids: detects cyclic patterns; returns None for non-cyclic.
    #[test]
    fn test_cyclic_try_from_ids_detection() {
        // [0, 1, 2, 0, 1, 2, ...] — shortest cycle_len is 3.
        let ids: Vec<u32> = (0..30).map(|i| (i % 3) as u32).collect();
        let c = CyclicPartition::try_from_ids(&ids).unwrap();
        assert_eq!(c.cycle_len(), 3);
        assert_eq!(c.pattern(), &[0, 1, 2]);

        // Non-cyclic: 99 breaks the periodicity. With MAX_CYCLE=8, no cycle
        // of length ≤ 8 can absorb a single outlier at index 18.
        let mut non_cyclic: Vec<u32> = (0..20).map(|i| (i % 5) as u32).collect();
        non_cyclic[18] = 99;
        let r = CyclicPartition::try_from_ids(&non_cyclic);
        assert!(
            r.is_none(),
            "single-outlier sequence must not be detected as cyclic"
        );
    }

    /// try_from_ids prefers the SHORTEST valid cycle_len.
    /// Pattern is canonicalized to first-appearance block IDs.
    #[test]
    fn test_cyclic_try_from_ids_prefers_shortest() {
        // Pattern [0, 0, 0, 0] is cyclic with cycle_len=1 (all same).
        let ids = vec![0, 0, 0, 0, 0];
        let c = CyclicPartition::try_from_ids(&ids).unwrap();
        assert_eq!(c.cycle_len(), 1);
        assert_eq!(c.pattern(), &[0]);

        // Pattern [1, 2, 1, 2, 1, 2] — shortest cycle is 2.
        // Stored pattern is canonicalized to [0, 1].
        let ids = vec![1, 2, 1, 2, 1, 2];
        let c = CyclicPartition::try_from_ids(&ids).unwrap();
        assert_eq!(c.cycle_len(), 2);
        assert_eq!(c.pattern(), &[0, 1]);
    }

    /// FastPartition conversion is bit-exact.
    #[test]
    fn test_cyclic_to_fast_partition_bit_exact() {
        let c = CyclicPartition::from_pattern(&[97, 98, 99], 4096);
        let fast = c.to_fast_partition();

        let cy_ids = c.get_ids_owned();
        let fast_ids = fast.get_ids();
        assert_eq!(cy_ids.as_slice(), fast_ids);

        let expected_fast = fast_from_pattern(&[97, 98, 99], 4096);
        assert_eq!(fast.get_ids(), expected_fast.get_ids());
    }

    // ─── lcm Meet closure tests ────────────────────────────────────────────

    /// B-C1 — lcm closure: `cyclic_refine` returns Some iff lcm(L₁, L₂) ≤ MAX_CYCLE.
    #[test]
    fn test_cyclic_meet_lcm_closure_some_none() {
        let pairs: &[((usize, usize), bool)] = &[
            // (L₁, L₂, expect_some)
            ((1, 5), true),  // lcm=5  ≤ 8
            ((2, 2), true),  // lcm=2
            ((2, 3), true),  // lcm=6
            ((2, 4), true),  // lcm=4
            ((3, 3), true),  // lcm=3
            ((4, 2), true),  // lcm=4
            ((4, 4), true),  // lcm=4
            ((2, 8), true),  // lcm=8
            ((4, 8), true),  // lcm=8
            ((3, 6), true),  // lcm=6
            ((3, 5), false), // lcm=15 > 8
            ((4, 5), false), // lcm=20 > 8
            ((3, 8), false), // lcm=24 > 8
            ((5, 6), false), // lcm=30 > 8
            ((7, 8), false), // lcm=56 > 8
        ];
        let universes = [1024usize, 4096, 16384];

        for &((l1, l2), expect_some) in pairs {
            for &n in &universes {
                let p1: Vec<u32> = (0..l1 as u32).collect();
                let p2: Vec<u32> = (l1 as u32..(l1 + l2) as u32).collect();
                let c1 = CyclicPartition::from_pattern(&p1, n);
                let c2 = CyclicPartition::from_pattern(&p2, n);
                let result = cyclic_refine(&c1, &c2);
                assert_eq!(
                    result.is_some(),
                    expect_some,
                    "cyclic_refine({}, {}) at N={}: expected Some={}, got Some={}",
                    l1,
                    l2,
                    n,
                    expect_some,
                    result.is_some()
                );
                if let Some(merged) = result {
                    let lcm = super::lcm_u(l1, l2);
                    assert_eq!(
                        merged.cycle_len(),
                        lcm,
                        "merged cycle_len mismatch: got {}, expected lcm({}, {}) = {}",
                        merged.cycle_len(),
                        l1,
                        l2,
                        lcm
                    );
                }
            }
        }
    }

    /// Bit-exact on the Some branch: the cyclic_refine result matches
    /// FastPartition::differentiate over the same input.
    #[test]
    fn test_cyclic_meet_lcm_bit_exact() {
        // Pairs in the lcm ≤ MAX_CYCLE region.
        let pairs: &[(Vec<u32>, Vec<u32>)] = &[
            (vec![0, 1], vec![0, 1, 2]),                // lcm=6
            (vec![0, 1, 2], vec![0, 1]),                // lcm=6
            (vec![0, 0, 1, 1], vec![0, 1]),             // lcm=4
            (vec![0, 1, 2, 3], vec![0, 1]),             // lcm=4
            (vec![0], vec![0, 1, 2, 3, 4]),             // lcm=5
            (vec![0, 1, 2, 3, 4, 5, 6, 7], vec![0, 1]), // lcm=8
            (vec![0, 1, 2], vec![0, 1, 2]),             // lcm=3
            (vec![0, 0, 0, 1], vec![0, 1, 0, 1]),       // lcm=4
        ];
        let universes = [1024usize, 4096, 16384];

        let mut passed = 0;
        let mut total = 0;
        for (p1, p2) in pairs {
            for &n in &universes {
                total += 1;
                let c1 = CyclicPartition::from_pattern(p1, n);
                let c2 = CyclicPartition::from_pattern(p2, n);
                let merged = cyclic_refine(&c1, &c2).expect("lcm should be in-range");
                // Reference: FastPartition::differentiate on the materialized inputs.
                let f1 = c1.to_fast_partition();
                let f2 = c2.to_fast_partition();
                let f_ref = f1.differentiate(&f2);
                let merged_ids = merged.get_ids_owned();
                let ref_ids = f_ref.get_ids();
                // Both are partitions, so the block IDs may differ in labeling,
                // but the *equivalence classes* must match. Compare via
                // pairwise-equivalence: for all (i, j), merged_eq(i,j) iff ref_eq(i,j).
                // Cheaper proxy: re-canonicalize both via FastPartition::from_ids.
                let merged_canon = FastPartition::from_ids(&merged_ids);
                let ref_canon = FastPartition::from_ids(ref_ids);
                assert_eq!(
                    merged_canon.get_ids(),
                    ref_canon.get_ids(),
                    "cyclic_refine vs FastPartition::differentiate MISMATCH for ({:?}, {:?}) at N={}",
                    p1,
                    p2,
                    n
                );
                passed += 1;
            }
        }
        assert_eq!(passed, total, "B-C2 bit-exact: {}/{} PASS", passed, total);
    }

    /// B-C2 — entropy + distinction_count bit-exact for lcm-closed Meet.
    #[test]
    fn test_cyclic_meet_metrics_bit_exact() {
        let pairs: &[(Vec<u32>, Vec<u32>)] = &[
            (vec![0, 1], vec![0, 1, 2]),
            (vec![0, 1, 2, 3], vec![0, 1]),
            (vec![0, 1, 2, 3, 4, 5, 6, 7], vec![0, 1]),
            (vec![0, 1, 2], vec![0, 1, 2]),
        ];
        let universes = [1024usize, 4096, 16384];

        for (p1, p2) in pairs {
            for &n in &universes {
                let c1 = CyclicPartition::from_pattern(p1, n);
                let c2 = CyclicPartition::from_pattern(p2, n);
                let merged = cyclic_refine(&c1, &c2).unwrap();
                let f_ref = c1
                    .to_fast_partition()
                    .differentiate(&c2.to_fast_partition());

                let h_merged = merged.logical_entropy();
                let h_ref = f_ref.logical_entropy();
                assert!(
                    (h_merged - h_ref).abs() < 1e-12,
                    "entropy MISMATCH for ({:?}, {:?}) at N={}: merged={} ref={}",
                    p1,
                    p2,
                    n,
                    h_merged,
                    h_ref
                );
                assert_eq!(
                    merged.distinction_count(),
                    f_ref.distinction_count(),
                    "distinction_count MISMATCH for ({:?}, {:?}) at N={}",
                    p1,
                    p2,
                    n
                );
                assert_eq!(
                    merged.block_count(),
                    f_ref.block_count(),
                    "block_count MISMATCH for ({:?}, {:?}) at N={}",
                    p1,
                    p2,
                    n
                );
            }
        }
    }

    /// B-C3 — Memory of lcm-closed Meet result remains ≤ 128 bytes (substrate
    /// preserved vs FastPartition's |U|·4 + K·|U|/8).
    #[test]
    fn test_cyclic_meet_memory_bounded() {
        let p1 = vec![0u32, 1, 2, 3, 4, 5, 6, 7];
        let p2 = vec![0u32, 1];
        let c1 = CyclicPartition::from_pattern(&p1, 65536);
        let c2 = CyclicPartition::from_pattern(&p2, 65536);
        let merged = cyclic_refine(&c1, &c2).unwrap();
        assert!(
            merged.memory_usage() <= 128,
            "cyclic_refine result memory ({}) must remain ≤ 128 bytes",
            merged.memory_usage()
        );
        // And independent of |U|: same lcm pair at N=1024 should give the
        // same compact size.
        let small = cyclic_refine(
            &CyclicPartition::from_pattern(&p1, 1024),
            &CyclicPartition::from_pattern(&p2, 1024),
        )
        .unwrap();
        assert_eq!(merged.memory_usage(), small.memory_usage());
    }

    /// B-C1 — Overflow case: lcm > MAX_CYCLE returns None.
    #[test]
    fn test_cyclic_meet_overflow_returns_none() {
        // lcm(3, 5) = 15 > 8
        let c1 = CyclicPartition::from_pattern(&[0u32, 1, 2], 1024);
        let c2 = CyclicPartition::from_pattern(&[0u32, 1, 2, 3, 4], 1024);
        assert!(cyclic_refine(&c1, &c2).is_none());

        // lcm(4, 6) = 12 > 8
        let c3 = CyclicPartition::from_pattern(&[0u32, 1, 2, 3], 1024);
        let c4 = CyclicPartition::from_pattern(&[0u32, 1, 2, 3, 4, 5], 1024);
        assert!(cyclic_refine(&c3, &c4).is_none());
    }

    /// Sanity: gcd / lcm helpers.
    #[test]
    fn test_gcd_lcm_helpers() {
        assert_eq!(super::gcd_u(12, 18), 6);
        assert_eq!(super::gcd_u(7, 5), 1);
        assert_eq!(super::lcm_u(4, 6), 12);
        assert_eq!(super::lcm_u(2, 3), 6);
        assert_eq!(super::lcm_u(8, 8), 8);
        assert_eq!(super::lcm_u(0, 5), 0);
    }
}
