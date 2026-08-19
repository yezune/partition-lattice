//! Wide Cyclic Partition — compatibility re-export of the const-generic
//! `crate::cyclic::CyclicPartitionN` at `L = MAX_CYCLE_WIDE = 16`.
//!
//! Relaxing the cycle-length bound: the narrow
//! [`crate::cyclic::CyclicPartition`] (`L = 8`) covers SFM ngram crystals with
//! `cycle_len ≤ 8`. [`WideCyclicPartition`] extends the upper bound to
//! `cycle_len ≤ 16`, keeping patterns of length 9 through 16 in compact form.
//!
//! Prior to 2026-05 this module held a hand-duplicated struct ~87% identical
//! to `cyclic.rs`. Both forms are now the single const-generic
//! `crate::cyclic::CyclicPartitionN`; this module re-exports the `L = 16`
//! alias for path compatibility (`crate::cyclic_wide::WideCyclicPartition`)
//! and retains the L ∈ [9, 16] verification sweep below.
//!
//! ## Memory cost
//!
//! | representation                | bytes at N=4096 | bytes at N=65536 |
//! |---                            |---              |---               |
//! | `FastPartition`               | ~16 KB          | ~275 KB          |
//! | `CyclicPartition` (L=8)       | ~40 B           | ~40 B            |
//! | `WideCyclicPartition` (L=16)  | **~72 B**       | **~72 B**        |
//!
//! Memory remains constant in `|U|`, still well below the 128-byte substrate
//! budget (Sprint A claim C1).

pub use crate::cyclic::{WideCyclicPartition, MAX_CYCLE_WIDE};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimized::FastPartition;

    fn fast_from_pattern(pattern: &[u32], size: usize) -> FastPartition {
        let cl = pattern.len();
        let ids: Vec<u32> = (0..size).map(|i| pattern[i % cl]).collect();
        FastPartition::from_ids(&ids)
    }

    /// A-C1 — Memory cost is constant in |U| and ≤ 128 bytes at L=16.
    #[test]
    fn test_wide_cyclic_memory_under_128_bytes() {
        let small = WideCyclicPartition::from_pattern(&[0, 1, 2], 4096);
        let large = WideCyclicPartition::from_pattern(
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            65536,
        );
        assert_eq!(
            small.memory_usage(),
            large.memory_usage(),
            "WideCyclicPartition compact-form memory MUST be independent of |U|"
        );
        assert!(
            large.memory_usage() <= 128,
            "WideCyclicPartition compact memory ({}) must be ≤ 128 bytes",
            large.memory_usage()
        );
    }

    /// A-C2 — `get_ids_owned` is bit-exact with FastPartition over the
    /// extended L ∈ [9, 16] range × 4 universe sizes.
    #[test]
    fn test_wide_cyclic_bit_exact_get_ids_with_fast_partition() {
        let patterns: Vec<Vec<u32>> = vec![
            (0..9).collect(),                         // L=9
            (0..10).collect(),                        // L=10
            (0..12).collect(),                        // L=12
            (0..14).collect(),                        // L=14
            (0..15).collect(),                        // L=15
            (0..16).collect(),                        // L=16
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0],       // L=10, repeat motif
            vec![5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8], // L=12, blocks of 3
        ];
        let universes = [4096usize, 8192, 16384, 65536];

        let mut passed = 0;
        let mut total = 0;
        for pattern in &patterns {
            for &n in &universes {
                total += 1;
                let wide = WideCyclicPartition::from_pattern(pattern, n);
                let fast = fast_from_pattern(pattern, n);
                let wy_ids = wide.get_ids_owned();
                let fast_ids = fast.get_ids();
                assert_eq!(
                    wy_ids.len(),
                    fast_ids.len(),
                    "ids length mismatch for L={} at N={}",
                    pattern.len(),
                    n
                );
                assert_eq!(
                    wy_ids.as_slice(),
                    fast_ids,
                    "get_ids byte-exact MISMATCH for L={} at N={}",
                    pattern.len(),
                    n
                );
                passed += 1;
            }
        }
        assert_eq!(passed, total, "A-C2 get_ids: {}/{} PASS", passed, total);
    }

    /// A-C2 — logical_entropy is bit-exact (|Δ| < 1e-12) with FastPartition.
    #[test]
    fn test_wide_cyclic_bit_exact_logical_entropy_with_fast_partition() {
        let patterns: Vec<Vec<u32>> = vec![
            (0..9).collect(),
            (0..10).collect(),
            (0..12).collect(),
            (0..14).collect(),
            (0..15).collect(),
            (0..16).collect(),
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            vec![5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
        ];
        let universes = [4096usize, 8192, 16384, 65536];

        for pattern in &patterns {
            for &n in &universes {
                let wide = WideCyclicPartition::from_pattern(pattern, n);
                let fast = fast_from_pattern(pattern, n);
                let h_wide = wide.logical_entropy();
                let h_fast = fast.logical_entropy();
                assert!(
                    (h_wide - h_fast).abs() < 1e-12,
                    "entropy MISMATCH for L={} at N={}: wide={} fast={}",
                    pattern.len(),
                    n,
                    h_wide,
                    h_fast
                );
            }
        }
    }

    /// A-C2 — distinction_count is bit-exact.
    #[test]
    fn test_wide_cyclic_bit_exact_distinction_count_with_fast_partition() {
        let patterns: Vec<Vec<u32>> = vec![
            (0..9).collect(),
            (0..10).collect(),
            (0..12).collect(),
            (0..16).collect(),
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            vec![5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
        ];
        let universes = [4096usize, 8192, 16384, 65536];

        for pattern in &patterns {
            for &n in &universes {
                let wide = WideCyclicPartition::from_pattern(pattern, n);
                let fast = fast_from_pattern(pattern, n);
                assert_eq!(
                    wide.distinction_count(),
                    fast.distinction_count(),
                    "distinction_count MISMATCH for L={} at N={}",
                    pattern.len(),
                    n
                );
            }
        }
    }

    /// A-C2 — block_count is bit-exact.
    #[test]
    fn test_wide_cyclic_bit_exact_block_count_with_fast_partition() {
        let patterns: Vec<Vec<u32>> = vec![
            (0..9).collect(),
            (0..10).collect(),
            (0..12).collect(),
            (0..16).collect(),
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            vec![5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
        ];
        let universes = [4096usize, 8192, 16384, 65536];

        for pattern in &patterns {
            for &n in &universes {
                let wide = WideCyclicPartition::from_pattern(pattern, n);
                let fast = fast_from_pattern(pattern, n);
                assert_eq!(
                    wide.block_count(),
                    fast.block_count(),
                    "block_count MISMATCH for L={} at N={}",
                    pattern.len(),
                    n
                );
            }
        }
    }

    /// A-C3 — First-appearance canonicalization preserves equivalence class
    /// at the extended L range.
    #[test]
    fn test_wide_cyclic_canonicalization_first_appearance() {
        let raw_to_canonical = [
            (
                vec![100u32, 200, 100, 200, 300, 100],
                vec![0u32, 1, 0, 1, 2, 0],
            ),
            (
                vec![97u32, 98, 99, 100, 101, 102, 103, 104, 105],
                (0..9u32).collect(),
            ),
            (
                vec![5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
                vec![0u32, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            ),
            (
                vec![10, 20, 30, 10, 20, 30, 40, 50, 60, 10],
                vec![0u32, 1, 2, 0, 1, 2, 3, 4, 5, 0],
            ),
            ((1..=16u32).collect::<Vec<_>>(), (0..16u32).collect()),
        ];

        for (raw, canonical) in raw_to_canonical.iter() {
            let wide = WideCyclicPartition::from_pattern(raw, 4096);
            assert_eq!(
                wide.pattern(),
                canonical.as_slice(),
                "canonical mismatch for raw {:?}",
                raw
            );
        }
    }

    /// Determinism: same input → same output (100 trials).
    #[test]
    fn test_wide_cyclic_deterministic() {
        let p: Vec<u32> = (0..12).collect();
        let baseline_ids = WideCyclicPartition::from_pattern(&p, 1000).get_ids_owned();
        let baseline_h = WideCyclicPartition::from_pattern(&p, 1000).logical_entropy();
        for _ in 0..100 {
            let c = WideCyclicPartition::from_pattern(&p, 1000);
            assert_eq!(c.get_ids_owned(), baseline_ids);
            assert!((c.logical_entropy() - baseline_h).abs() < 1e-12);
        }
    }

    /// Layer A — try_from_ids round-trip for L ∈ [9, 16]. Verifies that
    /// detection finds the *shortest* cycle that fits, and the recovered
    /// partition is bit-exact with the input.
    #[test]
    fn test_wide_cyclic_try_from_ids_roundtrip_l9_to_l16() {
        let patterns: Vec<Vec<u32>> = vec![
            (0..9u32).collect(),
            (0..10u32).collect(),
            (0..11u32).collect(),
            (0..12u32).collect(),
            (0..13u32).collect(),
            (0..14u32).collect(),
            (0..15u32).collect(),
            (0..16u32).collect(),
        ];
        let n = 4096usize;
        for pattern in &patterns {
            let cl = pattern.len();
            // Build ids manually as ids[i] = pattern[i % cl].
            let ids: Vec<u32> = (0..n).map(|i| pattern[i % cl]).collect();
            let recovered = WideCyclicPartition::try_from_ids(&ids)
                .unwrap_or_else(|| panic!("try_from_ids must succeed for cycle_len = {}", cl));
            assert_eq!(recovered.cycle_len(), cl, "cycle_len mismatch for L={}", cl);
            assert_eq!(recovered.size(), n);
            assert_eq!(
                recovered.get_ids_owned(),
                ids,
                "get_ids round-trip MISMATCH for L={}",
                cl
            );
        }
    }

    /// Layer A — try_from_ids must also accept narrow (L ≤ 8) cyclic input
    /// since WideCyclicPartition is a strict superset. Verifies that the
    /// shortest cycle is recovered (not the maximal 16-period).
    #[test]
    fn test_wide_cyclic_try_from_ids_accepts_narrow_l() {
        for cl in 1..=8usize {
            let pattern: Vec<u32> = (0..cl as u32).collect();
            let n = 1024usize;
            let ids: Vec<u32> = (0..n).map(|i| pattern[i % cl]).collect();
            let recovered = WideCyclicPartition::try_from_ids(&ids)
                .unwrap_or_else(|| panic!("WideCyclic must accept narrow L={}", cl));
            assert_eq!(
                recovered.cycle_len(),
                cl,
                "shortest cycle MUST be recovered for narrow L={}",
                cl
            );
        }
    }

    /// Layer A — block_sizes_analytic bit-exact with FastPartition over the
    /// extended L ∈ [9, 16] range.
    #[test]
    fn test_wide_cyclic_block_sizes_analytic_bit_exact() {
        let patterns: Vec<Vec<u32>> = vec![
            (0..9).collect(),
            (0..10).collect(),
            (0..12).collect(),
            (0..16).collect(),
            vec![0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4],
        ];
        let universes = [4096usize, 65536];
        for pattern in &patterns {
            for &n in &universes {
                let wide = WideCyclicPartition::from_pattern(pattern, n);
                let fast = fast_from_pattern(pattern, n);
                let mut analytic = wide.block_sizes_analytic();
                analytic.sort_unstable_by(|a, b| b.cmp(a));
                let mut counts: crate::HashMap<u32, usize> = crate::HashMap::new();
                for &id in fast.get_ids() {
                    *counts.entry(id).or_insert(0) += 1;
                }
                let mut fast_sizes: Vec<usize> = counts.into_values().collect();
                fast_sizes.sort_unstable_by(|a: &usize, b: &usize| b.cmp(a));
                assert_eq!(
                    analytic,
                    fast_sizes,
                    "block_sizes_analytic MISMATCH for L={} at N={}: \
                     wide={:?} fast={:?}",
                    pattern.len(),
                    n,
                    analytic,
                    fast_sizes,
                );
            }
        }
    }

    /// block_of (O(1) virtual access) matches the materialized id at the
    /// extended L range.
    #[test]
    fn test_wide_cyclic_block_of_matches_get_ids() {
        let pattern: Vec<u32> = (0..14).collect();
        let n = 100;
        let c = WideCyclicPartition::from_pattern(&pattern, n);
        let ids = c.get_ids_owned();
        for i in 0..n {
            assert_eq!(c.block_of(i), ids[i]);
        }
    }
}
