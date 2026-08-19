//! Discrete curvature `κ(Σ) = Var(d_Σ)` over a set of partitions.
//!
//! `κ(Σ)` is the variance, over pairs `(i, j)`, of `d_Σ(i, j)` — the fraction of partitions in
//! `Σ` that *distinguish* `i` from `j`. It is background-independent: it depends only on which
//! pairs the partitions separate, not on any embedding or coordinates.
//!
//! Under repeated refinement `κ` decreases toward 0. ⚠ Read that as **homogenisation, not
//! direction**: `κ` is a second moment and vanishes at *both* extremes (with the discrete
//! partition `d_Σ ≡ 1`, with the indiscrete one `d_Σ ≡ 0` — either way the variance is zero).
//! For direction, use the **first** moment of the same distribution, which separates refinement
//! from coarsening by sign. Report the two together; `κ` alone cannot be interpreted.
//!
//! This is a ratio (`f64`), not an exact count, which is why it lives here rather than in
//! [`crate::incidence`].

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::FloatExt;
use crate::Partition;
#[allow(unused_imports)]
use alloc::borrow::ToOwned;
#[allow(unused_imports)]
use alloc::boxed::Box;
#[allow(unused_imports)]
use alloc::string::{String, ToString};
#[allow(unused_imports)]
use alloc::vec::Vec;

/// Discrete curvature `κ(Σ) = Var(d_Σ)` — the variance of the pairwise distinction ratio.
///
/// `n` is the universe size (the range of the pairwise sweep); `crystal` is the set Σ.
/// `d_Σ(i,j) = |{π ∈ Σ : π separates i and j}| / |Σ|`; κ is their population variance.
///
/// Edge cases: empty `crystal`, `n < 2`, or no pairs → `0.0` (undefined curvature reads flat).
/// If every partition separates the same pairs, `d_Σ` is constant and κ = 0.
pub fn discrete_curvature(n: usize, crystal: &[Partition]) -> f64 {
    dsigma_moments(n, crystal).1
}

/// The d_Σ **profile** — the list of pairwise distinction ratios.
///
/// Returns `d_Σ(i,j)` for every pair `(i < j)` in canonical order (`i` ascending, then `j`
/// ascending). Both [`discrete_curvature`] and [`dsigma_moments`] delegate here, so there is
/// exactly one implementation of the ratio.
///
/// **Why a profile:** a single scalar is non-injective and cannot support a discrimination
/// (a single scalar is non-injective). The profile keeps the whole distribution,
/// claim; the profile keeps the whole distribution.
///
/// Edge cases: empty crystal or `n < 2` → empty list.
pub fn dsigma_profile(n: usize, crystal: &[Partition]) -> Vec<f64> {
    let mut dists = Vec::new();
    if crystal.is_empty() || n < 2 {
        return dists;
    }
    for i in 0..n {
        for j in i + 1..n {
            let count = crystal
                .iter()
                .filter(|p| {
                    let ids = p.get_ids();
                    ids[i] != ids[j]
                })
                .count();
            dists.push(count as f64 / crystal.len() as f64);
        }
    }
    dists
}

/// The **(mean, variance)** of the d_Σ distribution — direction and homogeneity.
///
/// **mean = direction**: it rises as the crystal refines (1 in the discrete limit) and falls
/// as it coarsens (0 in the indiscrete limit). **variance = homogeneity** (this is κ): it
///
/// A controlled re-measurement showed that
/// vanishes at *both* extremes and therefore carries no direction. That is the structural
/// reason κ reads as homogenisation in both directions: a variance discards location.
///
/// Edge case: an empty profile gives `(0.0, 0.0)`.
pub fn dsigma_moments(n: usize, crystal: &[Partition]) -> (f64, f64) {
    let dists = dsigma_profile(n, crystal);
    if dists.is_empty() {
        return (0.0, 0.0);
    }
    let mean = dists.iter().sum::<f64>() / dists.len() as f64;
    let var = dists.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / dists.len() as f64;
    (mean, var)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(ids: &[u32]) -> Partition {
        Partition::from_ids(ids.to_vec())
    }

    // ── d_Sigma profile and moments ──────────────────────────────────────
    // kappa = Var(d_Sigma) loses direction because a variance discards location.
    // The first moment of the same distribution separates the two extremes.

    /// The profile lists the pairwise d_Σ values. Worked example: crystal = [[0,0,1,1],[0,1,0,1]], n = 4
    /// → pairs (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) = [1/2, 1/2, 1, 1, 1/2, 1/2].
    #[test]
    fn t_dsigma_profile_hand_computed() {
        let crystal = vec![p(&[0, 0, 1, 1]), p(&[0, 1, 0, 1])];
        let prof = dsigma_profile(4, &crystal);
        assert_eq!(prof.len(), 6, "4C2 = 6 pairs");
        let expected = [0.5, 0.5, 1.0, 1.0, 0.5, 0.5];
        for (got, want) in prof.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-12, "profile {prof:?} != {expected:?}");
        }
    }

    /// ★ The point: **κ cannot tell the two extremes apart; the mean can.**
    /// Discrete (everything separated) and indiscrete (nothing separated) both give Var = 0,
    /// while the mean splits them into 1 and 0 — direction lives in the first moment.
    #[test]
    fn t_dsigma_mean_separates_extremes_where_kappa_cannot() {
        let discrete = vec![p(&[0, 1, 2, 3])];
        let indiscrete = vec![p(&[0, 0, 0, 0])];

        let (mean_d, var_d) = dsigma_moments(4, &discrete);
        let (mean_i, var_i) = dsigma_moments(4, &indiscrete);

        // kappa collapses both extremes to the same value.
        assert_eq!(var_d, 0.0, "discrete: Var=0");
        assert_eq!(var_i, 0.0, "indiscrete: Var=0");
        assert_eq!(
            discrete_curvature(4, &discrete),
            discrete_curvature(4, &indiscrete),
            "kappa cannot separate the two extremes"
        );

        // The mean does separate them.
        assert_eq!(mean_d, 1.0, "discrete: every pair is distinguished, so mean d_Sigma = 1");
        assert_eq!(mean_i, 0.0, "indiscrete: no pair is distinguished, so mean d_Sigma = 0");
        assert!(mean_d > mean_i, "direction lives in the first moment");
    }

    /// [`discrete_curvature`] is the variance of the profile, and must stay
    /// **bit-exact** with the direct computation.
    #[test]
    fn t_discrete_curvature_delegates_to_profile_bit_exact() {
        let cases: Vec<Vec<Partition>> = vec![
            vec![p(&[0, 0, 1, 1]), p(&[0, 1, 0, 1])],
            vec![p(&[0, 1, 2, 3])],
            vec![p(&[0, 0, 0, 0])],
            vec![p(&[0, 0, 1, 1]), p(&[0, 1, 1, 1]), p(&[0, 1, 2, 2])],
        ];
        for c in &cases {
            let (_, var) = dsigma_moments(4, c);
            assert_eq!(
                discrete_curvature(4, c),
                var,
                "κ ≡ Var(profile) bit-exact"
            );
        }
    }

    /// Edge cases: empty crystal or n < 2 → empty profile, moments (0, 0).
    #[test]
    fn t_dsigma_boundary_empty() {
        assert!(dsigma_profile(4, &[]).is_empty(), "an empty crystal yields an empty profile");
        assert!(dsigma_profile(1, &[p(&[0])]).is_empty(), "n < 2 leaves no pairs");
        assert_eq!(dsigma_moments(4, &[]), (0.0, 0.0));
        assert_eq!(dsigma_moments(1, &[p(&[0])]), (0.0, 0.0));
    }

    /// Hand-computed κ: crystal = [[0,0,1,1],[0,1,0,1]], n = 4 → Var(d_Σ) = 1/18.
    /// d_Σ pairs = [1/2,1/2,1,1,1/2,1/2], mean=2/3, var=(4·(1/6)²+2·(1/3)²)/6 = (1/3)/6 = 1/18.
    #[test]
    fn t_discrete_curvature_hand_computed() {
        let crystal = vec![p(&[0, 0, 1, 1]), p(&[0, 1, 0, 1])];
        let kappa = discrete_curvature(4, &crystal);
        assert!(
            (kappa - 1.0 / 18.0).abs() < 1e-12,
            "κ = 1/18 = {}, got {}",
            1.0 / 18.0,
            kappa
        );
    }

    /// A single discrete partition separates every pair (d_Σ = 1), so Var = 0.
    #[test]
    fn t_discrete_curvature_uniform_is_zero() {
        assert_eq!(
            discrete_curvature(4, &[p(&[0, 1, 2, 3])]),
            0.0,
            "discrete: every pair has d_Sigma = 1, so var is 0"
        );
        assert_eq!(
            discrete_curvature(4, &[p(&[0, 0, 0, 0])]),
            0.0,
            "bottom: every pair has d_Sigma = 0, so var is 0"
        );
    }

    /// Edge cases: empty crystal or n < 2 → 0.0.
    #[test]
    fn t_discrete_curvature_edge_cases() {
        assert_eq!(discrete_curvature(4, &[]), 0.0, "empty crystal");
        assert_eq!(discrete_curvature(1, &[p(&[0])]), 0.0, "n<2");
        assert_eq!(discrete_curvature(0, &[p(&[0, 0])]), 0.0, "n=0");
    }
}
