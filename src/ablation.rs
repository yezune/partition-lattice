//! Ablation arms for testing whether the lattice actually contributes.
//!
//! A measurement claiming that the lattice contributes has to show that removing the lattice
//! changes the result. Re-deriving that control per experiment lets the definitions drift,
//! so the arms live here.
//!
//! # Three arms
//!
//! | Arm | Removed | Left |
//! |---|---|---|
//! | [`random_replace`] | structure: keeps the block *size profile*, shuffles membership | coarseness (same `h`) |
//! | [`degenerate`] | every distinction: one block | nothing (`h = 0`) |
//! | `no_meet` | combination: a single component (the caller picks it) | one predictor |
//!
//! `random_replace` is the important one. `degenerate` is easy to beat (`h = 0`, so any
//! structure wins), and `no_meet` only tests combination. Keeping the size profile while
//! shuffling membership is the only arm that asks whether *which element sits in which block*
//! was actually used: coarseness is unchanged, so `h` and headroom cannot tell them apart.
//! (itself an instance of scalar non-injectivity: distinct partitions can share one value).
//!
//! # What this module does not do
//!
//! It builds arms; it does not judge. Deciding that the real partition won is the caller's job,
//! and the threshold should be a relative collapse or an empirical null rather than beating chance.

use crate::Partition;
use std::collections::BTreeMap;

/// Pinned here so results stay bit-exact with the reference implementation.
#[derive(Clone)]
pub struct AblationRng(u64);

impl AblationRng {
    pub fn new(seed: u64) -> Self {
        AblationRng(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn shuffle<T>(v: &mut [T], rng: &mut AblationRng) {
    for i in (1..v.len()).rev() {
        v.swap(i, rng.below(i + 1));
    }
}

/// Reassigns membership at random while preserving the block-size profile.
///
/// The returned ids have the same size multiset as the original, so `h` and headroom are
/// unchanged and coarseness cannot distinguish the two; only membership differs.
///
/// Deterministic: the same `(ids, seed)` gives the same output.
pub fn random_replace(ids: &[u32], seed: u64) -> Vec<u32> {
    let n = ids.len();
    if n == 0 {
        return vec![];
    }
    let mut sizes: BTreeMap<u32, usize> = BTreeMap::new();
    for &k in ids {
        *sizes.entry(k).or_insert(0) += 1;
    }
    let mut order: Vec<usize> = (0..n).collect();
    let mut r = AblationRng::new(seed);
    shuffle(&mut order, &mut r);
    let mut out = vec![0u32; n];
    let mut it = order.into_iter();
    for (k, sz) in sizes {
        for _ in 0..sz {
            out[it.next().unwrap()] = k;
        }
    }
    out
}

/// Removes every distinction, giving the indiscrete partition with `h = 0`.
///
/// This arm is easy to beat. Beating it without also beating [`random_replace`] shows only
/// that anything beats nothing, not that the lattice contributes.
pub fn degenerate(n: usize) -> Vec<u32> {
    vec![0u32; n]
}

/// Both arms as a [`Partition`].
pub fn random_replace_partition(p: &Partition, seed: u64) -> Partition {
    Partition::from_ids(random_replace(&p.get_ids(), seed))
}

pub fn degenerate_partition(n: usize) -> Partition {
    Partition::from_ids(degenerate(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ablation_random_replace_preserves_size_profile() {
        let ids = vec![0, 0, 0, 1, 1, 2, 3, 3, 3, 3];
        let rr = random_replace(&ids, 42);
        let prof = |v: &[u32]| {
            let mut m: BTreeMap<u32, usize> = BTreeMap::new();
            for &k in v {
                *m.entry(k).or_insert(0) += 1;
            }
            m.values().copied().collect::<Vec<_>>()
        };
        assert_eq!(prof(&ids), prof(&rr), "the size profile must be preserved");
    }

    #[test]
    fn test_ablation_random_replace_h_is_indistinguishable() {
        // The point of this arm: coarseness (h) cannot tell it apart from the original.
        let ids = vec![0, 0, 0, 1, 1, 2, 3, 3, 3, 3];
        let g = Partition::from_ids(ids.clone());
        let r = random_replace_partition(&g, 7);
        assert!(
            (g.logical_entropy() - r.logical_entropy()).abs() < 1e-12,
            "h must match; a difference means the size profile broke"
        );
    }

    #[test]
    fn test_ablation_random_replace_actually_moves_membership() {
        // A control that cannot move is not a control.
        let ids: Vec<u32> = (0..64).map(|i| i % 8).collect();
        let rr = random_replace(&ids, 1);
        assert_ne!(ids, rr, "unchanged membership makes the ablation vacuous");
    }

    #[test]
    fn test_ablation_random_replace_is_deterministic() {
        let ids: Vec<u32> = (0..40).map(|i| i % 5).collect();
        assert_eq!(random_replace(&ids, 9), random_replace(&ids, 9));
        assert_ne!(random_replace(&ids, 9), random_replace(&ids, 10));
    }

    #[test]
    fn test_ablation_degenerate_has_zero_entropy() {
        assert_eq!(degenerate_partition(32).logical_entropy(), 0.0);
    }

    #[test]
    fn test_ablation_random_replace_empty_is_empty() {
        assert!(random_replace(&[], 3).is_empty());
    }
}
