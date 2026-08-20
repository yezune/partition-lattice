//! Value-free (float-free) primitives for the refine-select-coarsen cycle.
//!
//! Closes the cycle Meet → Δ → Select → Join with **zero floating-point operations**:
//! - Meet   = [`Partition::differentiate`] (integer block ids)
//! - Δ      = [`new_distinction_count`] (integer `|dit|`, the integer counterpart of Δh)
//! - Select = [`antichain_select`] (the non-dominated maximal elements of the dit-set
//!            inclusion order — an antichain; integer `refines` comparisons, no scalar
//!            argmax and no Boltzmann collapse)
//! - Join   = [`Partition::integrate`] (integer)
//!
//! No scalar entropy, divergence, or Boltzmann weighting is used, so the cycle is realisable
//! on an integer ALU with no FPU. Collapsing to a single choice is deferred to the *consumer*
//! rather than performed here.

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

/// Integer counterpart of Δh: how many new distinctions a candidate adds to `base`,
///
/// `|dit(base ∧ σ)| − |dit(base)|`. Always ≥ 0 because Meet is monotone.
/// No float entropy is needed: the integer `|dit|` preserves the ordering of Δh.
pub fn new_distinction_count(base: &Partition, candidate: &Partition) -> usize {
    base.differentiate(candidate)
        .dit_count()
        .saturating_sub(base.dit_count())
}

/// Extract the nondominated indices — the general antichain primitive.
///
/// `dominates(j, i)` means "j dominates i", so `i` is dropped. The *relation* is injected by
/// the caller (subset refinement, directed Pareto, …) while the extraction *algorithm* — an
/// O(k²) filter — lives here once. Callers with opposite policies (richness-max vs
/// closeness-min) share the loop but keep their own relation. The result is the nondominated
/// indices in ascending order; an empty result is possible and the fallback is the caller's policy.
pub fn nondominated_indices(k: usize, dominates: impl Fn(usize, usize) -> bool) -> Vec<usize> {
    (0..k)
        .filter(|&i| !(0..k).any(|j| j != i && dominates(j, i)))
        .collect()
}

/// Value-free antichain select: the ⊆-maximal (richest, nondominated) candidate indices.
///
/// Over the partial order of new-distinction sets `D_i = dit(base ∧ σ_i) ∖ dit(base)`, this
/// returns the **maximal, nondominated** elements — those `i` not strictly contained in any
/// `D_j`. Incomparability is preserved rather than forced into a total order; no scalar argmax.
///
/// **Why this can be value-free:** because all candidates share the floor `dit(base)`,
/// `D_i ⊆ D_j ⟺ dit(meet_i) ⊆ dit(meet_j) ⟺ meet_j.refines(meet_i)`.
/// ⊆-comparison of dit-sets reduces to `refines` (integer, O(N)) — no O(N²) set materialisation,
/// no floats. The extraction loop delegates to [`nondominated_indices`].
pub fn antichain_select(base: &Partition, candidates: &[Partition]) -> Vec<usize> {
    let meets: Vec<Partition> = candidates.iter().map(|c| base.differentiate(c)).collect();
    // dominates(j, i): meet_j strictly contains i's new distinctions. Equal dit sets keep the lower index.
    nondominated_indices(meets.len(), |j, i| {
        let j_refines_i = meets[j].refines(&meets[i]); // dit(meet_i) ⊆ dit(meet_j)
        let i_refines_j = meets[i].refines(&meets[j]);
        if j_refines_i && i_refines_j {
            j < i // equal dit sets: keep the lower index
        } else {
            j_refines_i // j strict richer
        }
    })
}

/// The select result of one value-free cycle step.
pub struct PureStep {
    /// The antichain: ⊆-maximal nondominated candidates, kept as a set rather than collapsed.
    pub antichain: Vec<usize>,
    /// The single `|dit|`-maximal candidate, for callers advancing one trajectory.
    pub richest: usize,
}

/// One value-free step: Meet → Δ → antichain select against the candidates.
///
/// Advancing is the caller's choice: `state.differentiate(&candidates[step.richest])` for a
/// single trajectory, or keep `step.antichain` as a set. No floating point.
pub fn pure_pcc_step(state: &Partition, candidates: &[Partition]) -> PureStep {
    let antichain = antichain_select(state, candidates);
    let richest = (0..candidates.len())
        .max_by_key(|&i| new_distinction_count(state, &candidates[i]))
        .unwrap_or(0);
    PureStep { antichain, richest }
}

/// **Goal-directed reversible select** — the target-seeking sibling of [`pure_pcc_step`].
///
/// For each candidate it considers two reversible transitions — refine (Meet) or coarsen
/// (exact Join) — and picks the one that most reduces `d_⊕(next, target)`, accepting it only
/// when it strictly improves. Returns `(cand_idx, coarsen?, next)`, or `None` if nothing
/// improves.
///
/// `d_⊕` is intrinsic to the dit structure and is used here only as an *ordering* (argmin), not
/// as an external reward. ⚠ Unlike [`pure_pcc_step`], the `d_⊕` comparison is a float, so this
/// function is outside the module's float-free guarantee. A size mismatch between target,
/// candidate and state is skipped rather than panicking. Coarsening uses the *exact* Join,
/// since reaching a coarser target requires *removing* distinctions, which Meet alone cannot do.
pub fn act_toward_step(
    state: &Partition,
    candidates: &[Partition],
    target: &Partition,
) -> Option<(usize, bool, Partition)> {
    if target.size() != state.size() {
        return None; // different universe: graceful no-op
    }
    let cur_d = state.ditset_xor_distance(target);
    let mut best: Option<(usize, bool, f64, Partition)> = None;
    for (i, c) in candidates.iter().enumerate() {
        if c.size() != state.size() {
            continue;
        }
        let moves = [
            (false, state.differentiate(c)),  // refine: adds distinctions
            (true, state.integrate_exact(c)), // coarsen: removes distinctions
        ];
        for (coarsen, next) in moves {
            let d = next.ditset_xor_distance(target);
            if d < cur_d && best.as_ref().is_none_or(|(_, _, bd, _)| d < *bd) {
                best = Some((i, coarsen, d, next));
            }
        }
    }
    best.map(|(i, coarsen, _d, next)| (i, coarsen, next))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(ids: &[u32]) -> Partition {
        Partition::from_ids(ids.to_vec())
    }

    /// A family of one-pair merges is pairwise incomparable, so the antichain is everything.
    fn one_pair_merged(n: usize) -> Vec<Partition> {
        let mut out = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                let mut ids: Vec<u32> = (0..n as u32).collect();
                ids[b] = ids[a];
                out.push(Partition::from_ids(ids));
            }
        }
        out
    }

    #[test]
    fn test_new_distinction_count_integer_delta() {
        let top = p(&[0, 0, 0, 0]);
        // top∧[0,0,1,1]=[0,0,1,1] → dit_count=N²−Σ|B|²=16−8=8 (ordered); base top=0 → Δ=8.
        assert_eq!(new_distinction_count(&top, &p(&[0, 0, 1, 1])), 8);
        // No new distinctions (top ∧ top = top), so 0.
        assert_eq!(new_distinction_count(&top, &top), 0);
    }

    #[test]
    fn test_antichain_select_incomparable_keeps_all() {
        // base = top with one-pair-merged candidates: all incomparable, so the antichain is everything.
        let top = p(&[0, 0, 0, 0]);
        let cands = one_pair_merged(4); // C(4,2)=6
        let sel = antichain_select(&top, &cands);
        assert_eq!(sel.len(), 6, "incomparable candidates are all selected: {sel:?}");
    }

    #[test]
    fn test_antichain_select_chain_keeps_richest() {
        // base = top with a chain of candidates: only the maximal (richest, discrete) one survives.
        let top = p(&[0, 0, 0, 0]);
        let cands = vec![p(&[0, 0, 0, 1]), p(&[0, 0, 1, 2]), p(&[0, 1, 2, 3])]; // increasing |dit| chain
        let sel = antichain_select(&top, &cands);
        assert_eq!(sel, vec![2], "only the richest (discrete) candidate is non-dominated");
    }

    #[test]
    fn test_antichain_select_dedup_identical() {
        let top = p(&[0, 0, 0, 0]);
        let cands = vec![p(&[0, 0, 1, 1]), p(&[0, 0, 1, 1]), p(&[0, 0, 1, 1])];
        // Three identical dit sets collapse to the lowest index.
        assert_eq!(antichain_select(&top, &cands), vec![0]);
    }

    #[test]
    fn test_pure_pcc_step_richest_and_antichain() {
        let top = p(&[0, 0, 0, 0]);
        let cands = vec![p(&[0, 0, 0, 1]), p(&[0, 1, 2, 3])]; // the latter is richer
        let step = pure_pcc_step(&top, &cands);
        assert_eq!(step.richest, 1, "the discrete partition maximises |dit|");
        // A chain (0 <= 1), so the antichain holds only the richest.
        assert_eq!(step.antichain, vec![1]);
        // After a Meet the accumulated |dit| is non-negative, computed with integers only.
        let next = top.differentiate(&cands[step.richest]);
        assert!(next.dit_count() >= top.dit_count());
    }

    #[test]
    fn test_act_toward_step_reversible_descent() {
        // From the indiscrete state, refining by candidate [0,0,1,1] reaches the target.
        let state = p(&[0, 0, 0, 0]);
        let target = p(&[0, 0, 1, 1]);
        let cands = vec![p(&[0, 0, 1, 1])];
        let (i, coarsen, next) = act_toward_step(&state, &cands, &target).expect("an improving transition exists");
        assert_eq!(i, 0);
        assert!(!coarsen, "reached by refining (Meet)");
        assert_eq!(next.ditset_xor_distance(&target), 0.0, "target reached (d_xor = 0)");
        // Already at the target (d_xor = 0), so no improving transition exists.
        assert!(act_toward_step(&target, &cands, &target).is_none(), "no improvement at d_xor = 0");
        // A universe-size mismatch returns None rather than panicking.
        assert!(act_toward_step(&state, &cands, &p(&[0, 0, 0])).is_none(), "size mismatch yields None");
    }

    // === nondominated_indices: antichain extraction ===

    /// General antichain extraction with an injected relation (Pareto-min on `(usize, usize)`).
    #[test]
    fn test_nondominated_indices_generic() {
        let prof = [(2usize, 6usize), (6, 2), (3, 3), (7, 7)];
        let sel = nondominated_indices(prof.len(), |j, i| {
            prof[j].0 <= prof[i].0
                && prof[j].1 <= prof[i].1
                && (prof[j].0 < prof[i].0 || prof[j].1 < prof[i].1)
        });
        // (7,7) is dominated by (2,6), (6,2) and (3,3) and drops out; the other three are incomparable.
        assert_eq!(sel, vec![0, 1, 2]);
        // Empty input yields an empty result; any fallback is the caller's policy.
        assert!(nondominated_indices(0, |_, _| false).is_empty());
    }

    /// The subset antichain (richness-max) and directed Pareto (closeness-min) are *not* the same:
    /// on the same input they select opposite ends. Only the *algorithm* is shared; the relation
    /// is a distinct policy.
    #[test]
    fn test_subset_vs_directed_pareto_select_opposite_ends() {
        let base = p(&[0, 0, 0, 0]);
        let c0 = p(&[0, 0, 0, 1]); // small |D|, close to base
        let c1 = p(&[0, 1, 2, 3]); // maximal |D| (discrete, far from base)
                                   // The one-dimensional subset antichain picks the richest candidate, c1.
        assert_eq!(antichain_select(&base, &[c0.clone(), c1.clone()]), vec![1]);
        // The two-dimensional directed Pareto minimum picks the closest candidate, c0: the opposite end.
        let prof: Vec<(usize, usize)> = [&c0, &c1]
            .iter()
            .map(|c| {
                (
                    new_distinction_count(&base, c),
                    new_distinction_count(c, &base),
                )
            })
            .collect();
        let pareto = nondominated_indices(prof.len(), |j, i| {
            prof[j].0 <= prof[i].0
                && prof[j].1 <= prof[i].1
                && (prof[j].0 < prof[i].0 || prof[j].1 < prof[i].1)
        });
        assert_eq!(
            pareto,
            vec![0],
            "directed Pareto picks the closest, the opposite end from the subset antichain"
        );
    }
}
