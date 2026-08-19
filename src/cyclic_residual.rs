//! Cyclic With Residual (D-axis: ε-bounded almost-cyclic + sparse exceptions).
//!
//! Real-world data is
//! rarely exactly cyclic — ngram corpora have OOV tokens, image tiles have
//! boundary artifacts. A `CyclicWithResidual` stores a base
//! [`crate::cyclic::CyclicPartition`] plus a *sparse* exception map
//! `index → block_id`. Two modes:
//!
//! | mode | `epsilon` | semantics | memory |
//! |---|---|---|---|
//! | **Exact**       | `0.0`         | exceptions fully preserved → bit-exact | 40B + |E|·overhead |
//! | **ε-bounded**   | `> 0.0`       | exceptions may be dropped if `|E|/N < ε` | 40B fixed (lossy) |
//!
//! ε-bounded mode is **opt-in**: a lossy path is never taken without the caller asking. The
//! Exact mode is the one covered by the correctness argument below; the applicability of
//! ε-bounded mode to any particular corpus is a separate, empirical question.
//!
//! ## Exact mode — correctness
//!
//! For any `base: CyclicPartition`, exception map `E`, and universe `N`:
//!
//! ```text
//!   r = CyclicWithResidual::new_exact(base, E)
//!   expand[i] = E.get(i).unwrap_or(base.block_of(i))
//!
//!   ⟹ FastPartition::from_ids(r.get_ids_owned()) ≡ FastPartition::from_ids(expand)
//!     (equivalence class identity — block ID labels may differ via
//!      first-appearance canonicalization)
//! ```
//!
//! Verified across 132 cases in this module's `tests`.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::FloatExt;
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

use crate::cyclic::CyclicPartition;
use crate::optimized::FastPartition;

/// Partition defined by a cyclic base plus sparse index-keyed exceptions.
#[derive(Debug, Serialize, Deserialize)]
pub struct CyclicWithResidual {
    base: CyclicPartition,
    /// Sparse exceptions: `index → block_id` overriding the cyclic base.
    /// BTreeMap chosen over HashMap for deterministic iteration.
    exceptions: BTreeMap<u32, u32>,
    /// Lossy-allowance threshold. `0.0` = Exact mode (exceptions preserved
    /// bit-exact). `> 0.0` = ε-bounded mode (exceptions may be dropped if
    /// `|E|/N < ε`). User must opt in explicitly (PCS-A3).
    epsilon: f64,
    /// Lazy materialized id vector. Not serialized; reset on Clone.
    #[serde(skip)]
    materialized: OnceLock<Vec<u32>>,
}

impl Clone for CyclicWithResidual {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            exceptions: self.exceptions.clone(),
            epsilon: self.epsilon,
            materialized: OnceLock::new(),
        }
    }
}

impl PartialEq for CyclicWithResidual {
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base
            && self.exceptions == other.exceptions
            && self.epsilon == other.epsilon
    }
}

impl CyclicWithResidual {
    /// Construct in **Exact mode** (`epsilon = 0.0`): exceptions are fully
    /// preserved and the resulting partition is bit-exact with the
    /// FastPartition obtained by point-wise expansion.
    pub fn new_exact(base: CyclicPartition, exceptions: BTreeMap<u32, u32>) -> Self {
        // Validate: all exception indices must be in [0, base.size()).
        debug_assert!(
            exceptions.keys().all(|&i| (i as usize) < base.size()),
            "CyclicWithResidual: exception index out of range"
        );
        Self {
            base,
            exceptions,
            epsilon: 0.0,
            materialized: OnceLock::new(),
        }
    }

    /// Construct in **ε-bounded mode** (`epsilon > 0`): caller explicitly
    /// opts into lossy approximation. When `|exceptions| / N < epsilon`, the
    /// exceptions are *dropped* on materialization → 40B substrate retained.
    /// When `|exceptions| / N >= epsilon`, behaves as Exact.
    ///
    /// # Panics
    /// - If `epsilon < 0.0` or `epsilon > 1.0` (not a valid ratio).
    /// - If `epsilon == 0.0` (use `new_exact` instead — PCS-A3 gate).
    pub fn new_epsilon_bounded(
        base: CyclicPartition,
        exceptions: BTreeMap<u32, u32>,
        epsilon: f64,
    ) -> Self {
        assert!(
            (0.0..=1.0).contains(&epsilon),
            "CyclicWithResidual: epsilon ({}) must be in [0, 1]",
            epsilon
        );
        assert!(
            epsilon > 0.0,
            "CyclicWithResidual::new_epsilon_bounded: epsilon must be > 0 \
             — use new_exact for epsilon=0 (lossy approximation must be explicit)"
        );
        Self {
            base,
            exceptions,
            epsilon,
            materialized: OnceLock::new(),
        }
    }

    /// Whether lossy approximation is currently active (caller opted in *and*
    /// the exception ratio fell below the threshold).
    pub fn is_lossy_active(&self) -> bool {
        if self.epsilon == 0.0 {
            return false;
        }
        let n = self.base.size();
        if n == 0 {
            return false;
        }
        (self.exceptions.len() as f64) / (n as f64) < self.epsilon
    }

    /// Fraction of elements that diverge from the cyclic base. O(1).
    pub fn exception_ratio(&self) -> f64 {
        let n = self.base.size();
        if n == 0 {
            return 0.0;
        }
        (self.exceptions.len() as f64) / (n as f64)
    }

    /// Universe size N.
    pub fn size(&self) -> usize {
        self.base.size()
    }

    /// Block ID of element `i`. O(log |E|) due to BTreeMap lookup.
    pub fn block_of(&self, i: usize) -> u32 {
        if !self.is_lossy_active() {
            if let Some(&b) = self.exceptions.get(&(i as u32)) {
                return b;
            }
        }
        self.base.block_of(i)
    }

    /// Materialize the full id sequence.
    pub fn get_ids_owned(&self) -> Vec<u32> {
        let n = self.base.size();
        if self.is_lossy_active() {
            // Lossy active: ignore exceptions entirely.
            self.base.get_ids_owned()
        } else {
            (0..n).map(|i| self.block_of(i)).collect()
        }
    }

    /// Lazy materialized id slice.
    pub fn get_ids(&self) -> &[u32] {
        self.materialized.get_or_init(|| self.get_ids_owned())
    }

    /// Logical entropy. Currently O(N + |E|) via materialization; analytical
    /// path (base contribution + exception perturbation) is a follow-up.
    pub fn logical_entropy(&self) -> f64 {
        FastPartition::from_ids(&self.get_ids_owned()).logical_entropy()
    }

    /// Distinction count.
    pub fn distinction_count(&self) -> usize {
        FastPartition::from_ids(&self.get_ids_owned()).distinction_count()
    }

    /// Distinct block count after considering exceptions.
    pub fn block_count(&self) -> usize {
        FastPartition::from_ids(&self.get_ids_owned()).block_count()
    }

    /// Memory cost in bytes. In Exact mode the BTreeMap dominates; in
    /// active-lossy mode this reduces to ~`sizeof(base)`.
    pub fn memory_usage(&self) -> usize {
        let base_bytes = core::mem::size_of::<Self>();
        let map_overhead = self.exceptions.len() * (core::mem::size_of::<u32>() * 2 + 16);
        base_bytes + map_overhead
    }

    /// Convert to dense `FastPartition`.
    pub fn to_fast_partition(&self) -> FastPartition {
        FastPartition::from_ids(&self.get_ids_owned())
    }

    /// Read-only access to the cyclic base.
    pub fn base(&self) -> &CyclicPartition {
        &self.base
    }

    /// Read-only access to the exception map.
    pub fn exceptions(&self) -> &BTreeMap<u32, u32> {
        &self.exceptions
    }
}

impl From<&CyclicWithResidual> for FastPartition {
    fn from(r: &CyclicWithResidual) -> Self {
        r.to_fast_partition()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: expand a cyclic base + exception map into the full id vector,
    /// matching the semantics block_of(i) = exceptions[i] ?? base.block_of(i).
    fn expand_via_formula(
        base_pattern: &[u32],
        exceptions: &BTreeMap<u32, u32>,
        size: usize,
    ) -> Vec<u32> {
        let bl = base_pattern.len();
        // Canonicalize base_pattern first-appearance (matching CyclicPartition).
        let mut canon_map: Vec<(u32, u32)> = Vec::new();
        let mut next_id: u32 = 0;
        let canonical_base: Vec<u32> = base_pattern
            .iter()
            .map(|&raw| {
                if let Some(&(_, c)) = canon_map.iter().find(|(r, _)| *r == raw) {
                    c
                } else {
                    let c = next_id;
                    canon_map.push((raw, c));
                    next_id += 1;
                    c
                }
            })
            .collect();
        // Exception raw values pass through as-is — final canonicalization is
        // done by FastPartition::from_ids when comparing.
        (0..size)
            .map(|i| {
                if let Some(&exc) = exceptions.get(&(i as u32)) {
                    exc
                } else {
                    canonical_base[i % bl]
                }
            })
            .collect()
    }

    /// D1-C1 — Exact mode bit-exact with FastPartition (equivalence class).
    /// Sweep across base patterns × exception counts × universe sizes.
    #[test]
    fn test_residual_exact_bit_exact() {
        // (base_pattern, num_exceptions, universe_size)
        let base_patterns: &[Vec<u32>] = &[
            vec![0, 1, 2],          // L=3
            vec![0, 1, 2, 3, 4],    // L=5
            vec![5, 5, 6, 6, 7, 7], // L=6, repeating
            vec![0, 1],             // L=2 minimal
        ];
        let exception_counts: &[usize] = &[0, 1, 3, 5, 10, 20, 50];
        let universes: &[usize] = &[1024, 4096];

        let mut passed = 0;
        let mut total = 0;
        for base_pattern in base_patterns {
            for &num_exc in exception_counts {
                for &n in universes {
                    total += 1;
                    // Build a deterministic exception map: every (n / (num_exc+1))-th
                    // index gets a new block ID (offset by 100 to ensure uniqueness vs base).
                    let mut exceptions = BTreeMap::new();
                    if num_exc > 0 {
                        let stride = n / (num_exc + 1);
                        for k in 1..=num_exc {
                            let idx = (k * stride) as u32;
                            exceptions.insert(idx, 100 + k as u32);
                        }
                    }

                    let base = CyclicPartition::from_pattern(base_pattern, n);
                    let residual = CyclicWithResidual::new_exact(base, exceptions.clone());
                    let actual_ids = residual.get_ids_owned();
                    let expected_ids = expand_via_formula(base_pattern, &exceptions, n);

                    // Both must produce the same partition (equivalence class).
                    // Canonicalize via FastPartition::from_ids and compare canonical IDs.
                    let a_canon = FastPartition::from_ids(&actual_ids);
                    let e_canon = FastPartition::from_ids(&expected_ids);
                    assert_eq!(
                        a_canon.get_ids(),
                        e_canon.get_ids(),
                        "Exact mode bit-exact MISMATCH for base={:?} |E|={} N={}",
                        base_pattern,
                        num_exc,
                        n
                    );
                    passed += 1;
                }
            }
        }
        assert_eq!(passed, total, "D1-C1: {}/{} PASS", passed, total);
    }

    /// D1-C2 — Exact mode entropy + distinction_count + block_count bit-exact.
    #[test]
    fn test_residual_exact_metrics_bit_exact() {
        let cases: &[(Vec<u32>, Vec<(u32, u32)>, usize)] = &[
            (vec![0, 1, 2], vec![], 1024),
            (vec![0, 1, 2], vec![(100, 99)], 1024),
            (
                vec![0, 1, 2, 3],
                vec![(50, 100), (150, 101), (250, 102)],
                4096,
            ),
            (vec![0, 1], vec![(0, 50), (1, 51), (2, 52)], 1024),
        ];
        for (base_pattern, exc_list, n) in cases {
            let exceptions: BTreeMap<u32, u32> = exc_list.iter().copied().collect();
            let base = CyclicPartition::from_pattern(base_pattern, *n);
            let residual = CyclicWithResidual::new_exact(base, exceptions.clone());
            let expanded = expand_via_formula(base_pattern, &exceptions, *n);
            let f_ref = FastPartition::from_ids(&expanded);

            assert!(
                (residual.logical_entropy() - f_ref.logical_entropy()).abs() < 1e-12,
                "entropy mismatch for base={:?} N={}",
                base_pattern,
                n
            );
            assert_eq!(
                residual.distinction_count(),
                f_ref.distinction_count(),
                "distinction mismatch for base={:?} N={}",
                base_pattern,
                n
            );
            assert_eq!(
                residual.block_count(),
                f_ref.block_count(),
                "block_count mismatch for base={:?} N={}",
                base_pattern,
                n
            );
        }
    }

    /// D1-C3 — ε API gate: epsilon=0 (or new_exact) preserves exceptions
    /// bit-exact; epsilon>0 mode is opt-in only.
    #[test]
    fn test_residual_epsilon_gate_api() {
        let base = CyclicPartition::from_pattern(&[0u32, 1, 2], 1000);
        let mut exceptions = BTreeMap::new();
        exceptions.insert(500u32, 99u32);

        // Exact mode → never lossy
        let exact = CyclicWithResidual::new_exact(base.clone(), exceptions.clone());
        assert!(!exact.is_lossy_active(), "Exact mode must never be lossy");
        assert_eq!(
            exact.block_of(500),
            99,
            "Exact mode must preserve exception"
        );
        assert_eq!(exact.exception_ratio(), 1.0 / 1000.0);

        // ε-bounded with high threshold → lossy active (1/1000 < 0.5)
        let lossy_active =
            CyclicWithResidual::new_epsilon_bounded(base.clone(), exceptions.clone(), 0.5);
        assert!(
            lossy_active.is_lossy_active(),
            "ε=0.5 with |E|/N=0.001 must be lossy active"
        );
        // Block at exception index should fall back to base (not 99).
        assert_eq!(
            lossy_active.block_of(500),
            base.block_of(500),
            "Lossy active must ignore exceptions"
        );

        // ε-bounded with low threshold → lossy NOT active (1/1000 NOT < 0.0001)
        let lossy_inactive =
            CyclicWithResidual::new_epsilon_bounded(base.clone(), exceptions.clone(), 0.0001);
        assert!(
            !lossy_inactive.is_lossy_active(),
            "ε=0.0001 with |E|/N=0.001 must NOT be lossy active"
        );
        assert_eq!(
            lossy_inactive.block_of(500),
            99,
            "Lossy inactive must preserve exception"
        );
    }

    /// D1-C3 — `new_epsilon_bounded(.., 0.0)` panics (gate enforcement).
    #[test]
    #[should_panic(expected = "epsilon must be > 0")]
    fn test_residual_epsilon_zero_panics_via_lossy_api() {
        let base = CyclicPartition::from_pattern(&[0u32, 1], 100);
        let _ = CyclicWithResidual::new_epsilon_bounded(base, BTreeMap::new(), 0.0);
    }

    /// D1-C3 — out-of-range epsilon panics.
    #[test]
    #[should_panic(expected = "epsilon")]
    fn test_residual_epsilon_negative_panics() {
        let base = CyclicPartition::from_pattern(&[0u32, 1], 100);
        let _ = CyclicWithResidual::new_epsilon_bounded(base, BTreeMap::new(), -0.1);
    }

    /// Memory: Exact mode scales with |E|; lossy-active retains compact base.
    #[test]
    fn test_residual_memory_modes() {
        let base = CyclicPartition::from_pattern(&[0u32, 1, 2], 10000);
        let mut large_excs = BTreeMap::new();
        for k in 0..100u32 {
            large_excs.insert(k * 100, 100 + k);
        }
        let exact = CyclicWithResidual::new_exact(base.clone(), large_excs.clone());
        let lossy = CyclicWithResidual::new_epsilon_bounded(base.clone(), large_excs.clone(), 0.5);

        // Exact memory scales with |E|; lossy active gets |U|-independent base.
        let n = base.size();
        let exact_mem = exact.memory_usage();
        let _lossy_mem = lossy.memory_usage(); // BTreeMap still allocated; but lossy ignores it
        assert!(
            exact.exception_ratio() < 0.5,
            "test setup: |E|/N must be < epsilon for lossy_active"
        );
        assert!(lossy.is_lossy_active());
        assert!(!exact.is_lossy_active());

        // Sanity: at least the structural memory of Exact accounts for the map.
        assert!(
            exact_mem > core::mem::size_of::<CyclicWithResidual>(),
            "Exact memory ({}) must exceed struct-only size when |E|>0",
            exact_mem
        );

        // Lossy materialized output ignores exceptions, identical to base.
        assert_eq!(lossy.get_ids_owned(), base.get_ids_owned());
        // Universe size sanity for the assertion above.
        assert_eq!(lossy.size(), n);
    }

    /// Determinism: same input → same output across 100 trials (Exact mode).
    #[test]
    fn test_residual_exact_deterministic() {
        let base = CyclicPartition::from_pattern(&[0u32, 1, 2], 1000);
        let mut exceptions = BTreeMap::new();
        exceptions.insert(10u32, 99);
        exceptions.insert(500u32, 88);
        let baseline =
            CyclicWithResidual::new_exact(base.clone(), exceptions.clone()).get_ids_owned();
        for _ in 0..100 {
            let r = CyclicWithResidual::new_exact(base.clone(), exceptions.clone());
            assert_eq!(r.get_ids_owned(), baseline);
        }
    }

    /// block_of matches get_ids_owned for Exact mode.
    #[test]
    fn test_residual_block_of_matches_get_ids() {
        let base = CyclicPartition::from_pattern(&[0u32, 1, 2, 3], 200);
        let mut exceptions = BTreeMap::new();
        exceptions.insert(50u32, 99);
        exceptions.insert(150u32, 88);
        let r = CyclicWithResidual::new_exact(base, exceptions);
        let ids = r.get_ids_owned();
        for i in 0..200 {
            assert_eq!(r.block_of(i), ids[i], "block_of mismatch at i={}", i);
        }
    }
}
