//! Maps between finite universes and the partitions they induce.
//!
//! A map `f : [0,N) → [0,M)` transports a partition **backwards**: a partition σ of the
//! codomain pulls back to the partition of the domain whose blocks are the preimages of
//! σ's blocks. This is the contravariant action, and it is the only direction that is
//! always defined — forward images of blocks need not partition the codomain.
//!
//! [`crate::endomap::kernel`] is the special case `pullback(f, discrete)`: pulling back the
//! finest partition separates exactly the points that `f` sends to different targets.
//! `test_pullback_of_discrete_is_kernel` locks that identity so the two cannot drift.
//!
//! **Exact integers and partitions only** — no floating point, no probability.

use crate::Partition;
#[allow(unused_imports)]
use alloc::vec::Vec;

/// Pull a codomain partition back along a map — `f*(σ)`.
///
/// `i ~ j` in `f*(σ)` iff `f(i)` and `f(j)` lie in the same block of `σ`. Block ids of the
/// result are the σ-block ids of the images, so `Partition::from_ids` is the construction.
///
/// `f[i]` must index into `sigma`; a value outside the codomain is a caller error and
/// panics rather than silently producing a wrong partition.
///
/// # Order
///
/// Pullback is monotone: refining σ refines `f*(σ)`. It is a Meet-homomorphism
/// (`f*(σ ∧ τ) = f*(σ) ∧ f*(τ)`) but **not** a Join-homomorphism in general — joining
/// blocks in the codomain can connect domain points that no single preimage links.
///
/// # Examples
///
/// ```
/// use partition_lattice::{map::pullback, Partition};
/// // f collapses {0,1} → 0 and {2,3} → 1; sigma keeps 0 and 1 apart.
/// let sigma = Partition::from_ids(vec![0, 1]);
/// let p = pullback(&[0, 0, 1, 1], &sigma);
/// assert_eq!(p.block_count(), 2);
/// ```
pub fn pullback(f: &[usize], sigma: &Partition) -> Partition {
    let sigma_ids = sigma.get_ids();
    let ids: Vec<u32> = f
        .iter()
        .map(|&fi| {
            debug_assert!(
                fi < sigma_ids.len(),
                "pullback: f maps outside the codomain partition"
            );
            sigma_ids[fi]
        })
        .collect();
    Partition::from_ids(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endomap;

    fn discrete(n: usize) -> Partition {
        Partition::from_ids((0..n as u32).collect())
    }

    fn indiscrete(n: usize) -> Partition {
        Partition::from_ids(alloc::vec![0u32; n])
    }

    // A note on what these tests must be able to see.
    //
    // The first draft used `discrete` for sigma almost everywhere. Deliberately breaking
    // `pullback` to ignore sigma entirely (returning `from_ids(f)`) still passed 4 of 5
    // tests — because pulling back the *discrete* partition IS `from_ids(f)`, so those
    // cases cannot distinguish "uses sigma" from "ignores sigma". The Meet test passed
    // vacuously for the same reason: both sides degraded identically.
    //
    // So every test below that claims sigma matters uses a sigma that is neither discrete
    // nor indiscrete, and `test_ignoring_sigma_would_be_caught` states that requirement
    // as an assertion rather than as a comment.

    /// Sigma that genuinely groups: images 0 and 1 together, image 2 apart.
    fn grouping_sigma() -> Partition {
        Partition::from_ids(alloc::vec![0, 0, 1])
    }

    /// The identity that keeps `endomap::kernel` and `pullback` from drifting apart.
    ///
    /// `kernel` is documented as `c*(discrete)`. If either implementation changes so that
    /// this stops holding, one of the two is wrong — and without this test neither would say so.
    ///
    /// ⚠ This test alone cannot detect a `pullback` that ignores sigma (both sides reduce
    /// to `from_ids(c)`); that is what [`test_ignoring_sigma_would_be_caught`] is for.
    #[test]
    fn test_pullback_of_discrete_is_kernel() {
        for c in [
            alloc::vec![0usize, 0, 1, 1],
            alloc::vec![0usize, 1, 2, 3],
            alloc::vec![2usize, 2, 2, 2],
            alloc::vec![1usize, 0],
        ] {
            let n = c.len();
            assert_eq!(
                pullback(&c, &discrete(n)).get_ids(),
                endomap::kernel(&c).get_ids(),
                "pullback(c, discrete) must equal kernel(c) for c = {c:?}"
            );
        }
    }

    /// Pulling back the coarsest partition distinguishes nothing.
    #[test]
    fn test_pullback_of_indiscrete_is_indiscrete() {
        let p = pullback(&[0, 1, 2, 1], &indiscrete(3));
        assert_eq!(p.block_count(), 1);
    }

    /// **The load-bearing test**: sigma must actually be consulted.
    ///
    /// `f` is injective, so `kernel(f)` — and any implementation that ignores sigma —
    /// separates all three points. A correct pullback merges 0 and 1, because their
    /// images 0 and 1 sit in the same sigma-block.
    #[test]
    fn test_ignoring_sigma_would_be_caught() {
        let f = [0usize, 1, 2];
        let p = pullback(&f, &grouping_sigma());
        let ids = p.get_ids();
        assert_eq!(p.block_count(), 2, "sigma merges the images 0 and 1");
        assert_eq!(ids[0], ids[1], "0 and 1 land in the same sigma-block");
        assert_ne!(ids[0], ids[2], "2 lands in the other block");
        // Spelled out: the wrong answer this test exists to reject.
        assert_ne!(
            p.get_ids(),
            endomap::kernel(&f).get_ids(),
            "an implementation that ignored sigma would return kernel(f) here"
        );
    }

    /// Preimages, not images: points sharing a target are always merged, and a coarser
    /// sigma merges strictly more.
    #[test]
    fn test_pullback_merges_points_with_equal_images() {
        let f = [0usize, 0, 1, 2, 2];
        let fine = pullback(&f, &discrete(3));
        let ids = fine.get_ids();
        assert_eq!(ids[0], ids[1], "0 and 1 share the image 0");
        assert_eq!(ids[3], ids[4], "3 and 4 share the image 2");
        assert_ne!(ids[0], ids[2], "distinct images stay apart under discrete sigma");

        // Coarsening sigma can only merge further — monotonicity of f*.
        let coarse = pullback(&f, &grouping_sigma());
        assert_eq!(coarse.block_count(), 2);
        assert_eq!(fine.block_count(), 3);
    }

    /// Meet-homomorphism — the structural property the Loom `map_hom` statement checks.
    ///
    /// The two sigmas are chosen so that neither is discrete and their Meet is strictly
    /// finer than both; an implementation ignoring sigma makes both sides equal for the
    /// wrong reason, so we also assert the value is not the sigma-blind one.
    #[test]
    fn test_pullback_preserves_meet() {
        let f = [0usize, 1, 2, 1, 0];
        let s1 = Partition::from_ids(alloc::vec![0, 0, 1]);
        let s2 = Partition::from_ids(alloc::vec![0, 1, 1]);
        let meet = s1.differentiate(&s2);
        assert_eq!(meet.block_count(), 3, "the two sigmas must actually cut each other");

        let lhs = pullback(&f, &meet);
        let rhs = pullback(&f, &s1).differentiate(&pullback(&f, &s2));
        assert_eq!(lhs.get_ids(), rhs.get_ids());

        // Non-vacuity: pulling back s1 alone must differ from pulling back their Meet.
        assert_ne!(
            pullback(&f, &s1).get_ids(),
            lhs.get_ids(),
            "if these agreed, the Meet test would pass without exercising sigma"
        );
    }

    /// Join is **not** preserved in general — the documented asymmetry, asserted.
    ///
    /// The counterexample needs the connecting chain to pass through a codomain element
    /// that `f` never hits. A first draft of this test put both chain endpoints inside the
    /// image and Join *was* preserved — the failure was the test's, not the claim's, and it
    /// is the reason this comment names the condition instead of leaving it implicit.
    #[test]
    fn test_pullback_does_not_preserve_join_in_general() {
        // Image of f is {0, 2}; codomain element 1 is never hit.
        let f = [0usize, 2];
        let s1 = Partition::from_ids(alloc::vec![0, 0, 1]); // {0,1} {2}
        let s2 = Partition::from_ids(alloc::vec![0, 1, 1]); // {0} {1,2}
        // In the codomain, 0 ~ 1 (s1) and 1 ~ 2 (s2): everything joins, but only via 1.
        let joined = pullback(&f, &s1.integrate(&s2));
        let separate = pullback(&f, &s1).integrate(&pullback(&f, &s2));
        assert_eq!(joined.block_count(), 1, "the codomain Join links 0 and 2 through 1");
        assert_eq!(
            separate.block_count(),
            2,
            "no domain chain reaches from 0 to 2 — the linking element has no preimage"
        );
    }

    /// The empty map yields the empty partition rather than panicking.
    #[test]
    fn test_pullback_empty_domain() {
        let p = pullback(&[], &discrete(3));
        assert_eq!(p.get_ids().len(), 0);
    }
}
