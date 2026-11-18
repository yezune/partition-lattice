//! Partition-native invariants of a finite endomap `c : [0,N) → [0,N)`.
//!
//! Generalises [`crate::orbit`] from a *group* action (every point periodic) to an arbitrary
//! endomap, where transients exist. The measure of that gap is [`tau`]: `tau(c) == 0` iff `c`
//! is a bijection, i.e. iff the orbit-partition assumption holds.
//!
//! **Exact integers and partitions only** — no floating point, no probability.
//!

use crate::Partition;
#[allow(unused_imports)]
use alloc::vec::Vec;

/// Kernel partition of `c` — `i ~ j` iff `c(i) = c(j)`.
///
/// This is `c*(discrete)`, the pullback of the finest partition. Block ids *are* the images,
/// so `Partition::from_ids` is the whole construction.
pub fn kernel(c: &[usize]) -> Partition {
    Partition::from_ids(c.iter().map(|&y| y as u32).collect())
}

/// Component partition `γ_c` — weakly connected components of the functional graph `i → c(i)`.
///
/// Derived as a **Join fold over edge partitions**, not by a bespoke graph traversal: the
/// components *are* the transitive closure of the edge relation, and Join *is* that closure.
/// Keeping it a lattice operation is the point: no hand-written algorithm where the lattice
/// already says it.
///
/// ⚠ `integrate` (exact) is the right call here: this is a **structural** computation, not an
/// iterative learning loop, so the advice to prefer a shallow Join there does not apply.
///
/// Each weakly connected component holds exactly one periodic orbit, so
/// `components(c).block_count()` is also the number of eigenbehaviours.
pub fn components(c: &[usize]) -> Partition {
    let n = c.len();
    if n == 0 {
        return Partition::from_ids(Vec::new());
    }
    // Start discrete; merge each edge {i, c(i)} in turn.
    let mut acc = Partition::from_ids((0..n as u32).collect());
    for (i, &target) in c.iter().enumerate() {
        if target == i || target >= n {
            continue;
        }
        let mut edge: Vec<u32> = (0..n as u32).collect();
        let (lo, hi) = if i < target { (i, target) } else { (target, i) };
        edge[hi] = edge[lo];
        acc = acc.integrate(&Partition::from_ids(edge));
    }
    acc
}

/// Fixed-point count `#Fix(c) = |{ i : c(i) = i }|` — the equaliser of `c` and the identity.
///
/// Paired with [`components`] (`#Def`, the coequaliser) this bounds the cycle structure
/// *without running a single orbit*, since every fixed point is a length-1 cycle owning its
/// own component:
///
/// - `fix_count(c) <= components(c).block_count()`
/// - equality **iff** no cycle is longer than 1 — so equality certifies that no orbit of `c`
///   can enter a non-trivial cycle, whatever the starting state
/// - the difference counts the non-trivial periodic orbits
///
/// All three are sealed exhaustively over every self-map with `n <= 5` in this module's tests
/// against an independent graph-walking oracle.
pub fn fix_count(c: &[usize]) -> usize {
    c.iter().enumerate().filter(|&(i, &y)| y == i).count()
}

/// Eventual image `Per(c) = c^n(O)` for `n ≥ |O|` — the periodic points.
fn eventual_image(c: &[usize]) -> Vec<bool> {
    let n = c.len();
    let mut live = alloc::vec![true; n];
    for _ in 0..n {
        let mut next = alloc::vec![false; n];
        for i in 0..n {
            if live[i] && c[i] < n {
                next[c[i]] = true;
            }
        }
        if next == live {
            break;
        }
        live = next;
    }
    live
}

/// Transient count `τ(c) = |O \ Per(c)|` — how far `c` is from being a permutation.
///
/// `tau(c) == 0` iff `c` is a bijection; that is exactly the condition
/// [`crate::orbit::OrbitPartition`] assumes, so `tau` measures the distance to that substrate.
pub fn tau(c: &[usize]) -> usize {
    eventual_image(c).iter().filter(|&&p| !p).count()
}

/// Composition in the standard order: `(f∘g)(x) = f(g(x))`.
///
/// Read as the *left plate* `g ∗ e = g ∘ e` — "the environment moves, then the body turns" —
/// the right factor acts first.
pub fn compose(f: &[usize], g: &[usize]) -> Vec<usize> {
    g.iter().map(|&x| f[x]).collect()
}

/// Apply the permutation `sigma` (with precomputed inverse) by conjugation: `σ ∘ c ∘ σ⁻¹`.
fn conjugate(c: &[usize], sigma: &[usize], sigma_inv: &[usize]) -> Vec<usize> {
    (0..c.len()).map(|x| sigma[c[sigma_inv[x]]]).collect()
}

/// **Organisation** of an endomap — its conjugacy class under renaming of the carrier,
/// represented by the lexicographically least conjugate.
///
/// This is the label-blind quotient at the level of *self-maps*: [`crate::Partition`] forgets
/// which element carries which block id, and `organisation` forgets it one layer up, for the
/// map itself. Two endomaps share an organisation exactly when one is a relabelling of the
/// other, so every invariant of the organisation (block sizes of the kernel, cycle lengths,
/// `tau`, `fix_count`) is automatically name-blind — label blindness, one layer up.
///
/// Cost is `n! · n` — intended for exhaustive certification at small `n`, not a hot path.
pub fn organisation(c: &[usize]) -> Vec<usize> {
    let n = c.len();
    if n == 0 {
        return Vec::new();
    }
    let mut sigma: Vec<usize> = (0..n).collect();
    let mut inv = alloc::vec![0usize; n];
    let mut best: Option<Vec<usize>> = None;
    // Heap's algorithm, iterative.
    let mut stack = alloc::vec![0usize; n];
    let visit = |sigma: &[usize], inv: &mut Vec<usize>, best: &mut Option<Vec<usize>>| {
        for (i, &v) in sigma.iter().enumerate() {
            inv[v] = i;
        }
        let conj = conjugate(c, sigma, inv);
        if best.as_ref().is_none_or(|b| conj < *b) {
            *best = Some(conj);
        }
    };
    visit(&sigma, &mut inv, &mut best);
    let mut i = 0usize;
    while i < n {
        if stack[i] < i {
            let j = if i % 2 == 0 { 0 } else { stack[i] };
            sigma.swap(j, i);
            visit(&sigma, &mut inv, &mut best);
            stack[i] += 1;
            i = 0;
        } else {
            stack[i] = 0;
            i += 1;
        }
    }
    best.unwrap_or_default()
}

/// **One-sided survival** — `e` perturbs `g` without dissolving it: `Org(g ∘ e) = Org(g)`.
///
/// "The environment gives no information, only perturbation, and what counts as a perturbation
/// is settled by the body" — formally, the predicate is not a function of `e` alone.
pub fn survives(g: &[usize], e: &[usize]) -> bool {
    organisation(&compose(g, e)) == organisation(g)
}

/// **Structural coupling (mutual survival)** — both plates keep their organisation:
/// `Org(g ∘ e) = Org(g)` *and* `Org(e ∘ g) = Org(e)`.
///
/// Symmetric by construction. Strictly narrower than [`survives`] from `n = 2` on: one-sided
/// survival admits `e = id` against every `g`, mutual survival does not.
///
/// ⚠ This is a *same-carrier* relation — both maps live on the same finite `[0,n)`. It says
/// nothing about corresponding two *different* carriers, and conflating the two is the
/// standard error.
pub fn is_mutually_surviving(g: &[usize], e: &[usize]) -> bool {
    survives(g, e) && survives(e, g)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn discrete(n: usize) -> Partition {
        Partition::from_ids((0..n as u32).collect())
    }

    /// Permutations have no transients; non-injective maps do.
    #[test]
    fn test_endomap_tau_is_zero_iff_bijective() {
        assert_eq!(tau(&[1, 2, 0]), 0, "3-cycle is a permutation");
        assert_eq!(tau(&[0, 1, 2]), 0, "identity");
        assert!(tau(&[0, 0, 1]) > 0, "non-injective must have transients");
        assert_eq!(tau(&[0, 0]), 1, "one point falls out of the eventual image");
    }

    /// The graph count and the lattice measure must agree.
    ///
    /// `h(ker c) = h(discrete)` iff `c` injective iff `τ(c) = 0`. Two independent routes
    /// to the same fact; disagreement would refute the construction.
    #[test]
    fn test_endomap_tau_zero_agrees_with_entropy_preservation() {
        for n in 1..=5usize {
            let mut c = alloc::vec![0usize; n];
            loop {
                let by_graph = tau(&c) == 0;
                let by_lattice =
                    (kernel(&c).logical_entropy() - discrete(n).logical_entropy()).abs() < 1e-12;
                assert_eq!(by_graph, by_lattice, "τ vs h disagree at c = {c:?}");

                // odometer over all maps [0,n)^n
                let mut i = 0;
                loop {
                    if i == n {
                        break;
                    }
                    c[i] += 1;
                    if c[i] < n {
                        break;
                    }
                    c[i] = 0;
                    i += 1;
                }
                if i == n {
                    break;
                }
            }
        }
    }

    /// One periodic orbit per component, so the component count equals the orbit count.
    #[test]
    fn test_endomap_components_count_matches_orbit_count() {
        // c = [1,0,3,2] — two 2-cycles, two components.
        assert_eq!(components(&[1, 0, 3, 2]).block_count(), 2);
        // c = [1,2,0,0] — one 3-cycle with element 3 hanging off it: one component.
        assert_eq!(components(&[1, 2, 0, 0]).block_count(), 1);
        // constant map: everything flows to one place.
        assert_eq!(components(&[0, 0, 0, 0]).block_count(), 1);
        // identity: every point its own component.
        assert_eq!(components(&[0, 1, 2, 3]).block_count(), 4);
    }

    /// Frame check: for a permutation the component partition is the orbit partition.
    #[test]
    fn test_endomap_components_are_orbits_for_permutations() {
        // (0 1)(2 3 4): orbits {0,1} and {2,3,4}.
        let c = [1usize, 0, 3, 4, 2];
        let g = components(&c);
        assert_eq!(g.block_count(), 2);
        assert_eq!(g.get_ids()[0], g.get_ids()[1]);
        assert_eq!(g.get_ids()[2], g.get_ids()[3]);
        assert_eq!(g.get_ids()[3], g.get_ids()[4]);
        assert_ne!(g.get_ids()[0], g.get_ids()[2]);
    }

    /// `components(c)` is always coarser than or equal to `kernel(c)`: a fibre lies inside one component.
    #[test]
    fn test_endomap_components_coarsen_kernel() {
        for c in [
            alloc::vec![0usize, 0, 1, 1],
            alloc::vec![1, 2, 0, 0],
            alloc::vec![0, 0, 0, 0],
            alloc::vec![1, 0, 3, 2],
        ] {
            let (g, k) = (components(&c), kernel(&c));
            assert!(k.refines(&g), "ker must refine γ for c = {c:?}");
        }
    }

    /// Enumerate every self-map `[0,n) -> [0,n)` (an odometer over the `n^n` tables).
    fn for_each_self_map(n: usize, mut f: impl FnMut(&[usize])) {
        let mut c = alloc::vec![0usize; n];
        loop {
            f(&c);
            let mut i = 0;
            loop {
                if i == n {
                    return;
                }
                c[i] += 1;
                if c[i] < n {
                    break;
                }
                c[i] = 0;
                i += 1;
            }
        }
    }

    /// Reference oracle — the multiset of periodic orbit lengths, obtained by walking each
    /// point onto its cycle and measuring that cycle directly.
    ///
    /// Deliberately independent of [`components`] and [`fix_count`]: it walks the functional
    /// graph rather than folding Joins, so agreement is evidence and not tautology. Requires
    /// a well-formed endomap (`c[i] < c.len()` for every `i`), which the odometer guarantees.
    fn cycle_lengths(c: &[usize]) -> Vec<usize> {
        let n = c.len();
        let mut counted = alloc::vec![false; n];
        let mut lengths = Vec::new();
        for start in 0..n {
            // n steps are enough to land on the cycle of this point's component.
            let mut x = start;
            for _ in 0..n {
                x = c[x];
            }
            if counted[x] {
                continue;
            }
            let mut len = 0usize;
            let mut y = x;
            loop {
                counted[y] = true;
                y = c[y];
                len += 1;
                if y == x {
                    break;
                }
            }
            lengths.push(len);
        }
        lengths
    }

    /// `#Fix` is the equaliser of `c` and the identity.
    #[test]
    fn test_endomap_fix_count_counts_fixed_points() {
        assert_eq!(fix_count(&[0, 1, 2, 3]), 4, "identity fixes everything");
        assert_eq!(fix_count(&[1, 0, 3, 2]), 0, "two 2-cycles fix nothing");
        assert_eq!(fix_count(&[0, 0, 0, 0]), 1, "constant map fixes its image");
        assert_eq!(
            fix_count(&[0, 1, 1, 2]),
            2,
            "0 and 1 are fixed, 2 and 3 are transient"
        );
        assert_eq!(fix_count(&[] as &[usize]), 0, "empty map");
    }

    /// DY4 — every fixed point is a length-1 cycle owning its own component, so
    /// `#Fix <= #Def`. Exhaustive over all self-maps for `n <= 5`.
    #[test]
    fn test_endomap_dy4_fix_count_at_most_component_count() {
        for n in 1..=5usize {
            for_each_self_map(n, |c| {
                assert!(
                    fix_count(c) <= components(c).block_count(),
                    "DY4 violated at c = {c:?}"
                );
            });
        }
    }

    /// DY6 — `#Fix = #Def` exactly when no cycle is longer than 1.
    ///
    /// This is the pre-orbit termination gate: equality certifies that no orbit of `c` can
    /// enter a non-trivial cycle, whatever the starting state.
    #[test]
    fn test_endomap_dy6_fix_equals_def_iff_no_long_cycles() {
        for n in 1..=5usize {
            for_each_self_map(n, |c| {
                let equal = fix_count(c) == components(c).block_count();
                let all_trivial = cycle_lengths(c).iter().all(|&len| len == 1);
                assert_eq!(equal, all_trivial, "DY6 violated at c = {c:?}");
            });
        }
    }

    /// `#Def - #Fix` counts the non-trivial periodic orbits.
    ///
    /// The oracle must also reproduce `#Def` itself (one cycle per weakly connected
    /// component); disagreement would refute [`components`], not just this corollary.
    #[test]
    fn test_endomap_def_minus_fix_counts_nontrivial_orbits() {
        for n in 1..=5usize {
            for_each_self_map(n, |c| {
                let lengths = cycle_lengths(c);
                let def = components(c).block_count();
                assert_eq!(
                    lengths.len(),
                    def,
                    "orbit count vs component count at c = {c:?}"
                );
                let nontrivial = lengths.iter().filter(|&&len| len >= 2).count();
                assert_eq!(
                    def - fix_count(c),
                    nontrivial,
                    "corollary violated at c = {c:?}"
                );
            });
        }
    }

    // ── Structural coupling (mutual survival) — independent recount in this kernel ──────

    /// `compose` is `(f∘g)(x) = f(g(x))` — the right factor acts first.
    #[test]
    fn test_endomap_compose_applies_right_factor_first() {
        let g = [0usize, 0, 0];
        let f = [1usize, 2, 0];
        assert_eq!(compose(&f, &g), alloc::vec![1, 1, 1]);
        assert_eq!(compose(&g, &f), alloc::vec![0, 0, 0], "composition is not commutative");
    }

    /// The organisation forgets element names but not shape.
    #[test]
    fn test_endomap_organisation_is_invariant_under_renaming() {
        let chain_a = [0usize, 0, 1];
        let chain_b = [1usize, 2, 2];
        assert_eq!(organisation(&chain_a), organisation(&chain_b), "same shape, renamed");
        let star = [0usize, 0, 0];
        assert_ne!(organisation(&chain_a), organisation(&star), "chain is not a star");
    }

    /// ★ The real invariance test: *every* conjugate lands on the same representative.
    ///
    /// The weaker "representative keeps the invariants" check passes even if `organisation`
    /// is the identity (nothing is normalised), so it cannot witness the quotient. This one
    /// conjugates by each permutation and demands one class.
    #[test]
    fn test_endomap_organisation_is_constant_on_conjugacy_classes() {
        for n in 2..=4usize {
            let mut sigmas: Vec<Vec<usize>> = Vec::new();
            let mut perm: Vec<usize> = (0..n).collect();
            // enumerate S_n by repeated next-permutation
            loop {
                sigmas.push(perm.clone());
                // next lexicographic permutation
                let Some(i) = (0..n.saturating_sub(1)).rev().find(|&i| perm[i] < perm[i + 1])
                else {
                    break;
                };
                let j = (i + 1..n).rev().find(|&j| perm[j] > perm[i]).unwrap();
                perm.swap(i, j);
                perm[i + 1..].reverse();
            }
            assert!(sigmas.len() > 1, "S_{n} must have more than one element");
            for_each_self_map(n, |c| {
                let base = organisation(c);
                for sigma in &sigmas {
                    let mut inv = alloc::vec![0usize; n];
                    for (i, &v) in sigma.iter().enumerate() {
                        inv[v] = i;
                    }
                    let conj: Vec<usize> = (0..n).map(|x| sigma[c[inv[x]]]).collect();
                    assert_eq!(
                        organisation(&conj),
                        base,
                        "conjugate {conj:?} of {c:?} by {sigma:?} left the class"
                    );
                }
            });
        }
    }

    /// Weaker companion — the representative keeps the name-blind invariants.
    /// ⚠ This one is *not* evidence of normalisation (it passes for the identity map too);
    /// the test above is. Kept because it checks the invariants themselves are name-blind.
    #[test]
    fn test_endomap_organisation_preserves_name_blind_invariants() {
        for n in 1..=4usize {
            for_each_self_map(n, |c| {
                let rep = organisation(c);
                assert_eq!(fix_count(&rep), fix_count(c), "fix_count at {c:?}");
                assert_eq!(tau(&rep), tau(c), "tau at {c:?}");
                assert_eq!(
                    components(&rep).block_count(),
                    components(c).block_count(),
                    "component count at {c:?}"
                );
            });
        }
    }

    /// Structural coupling is symmetric by construction (a conjunction of two plates).
    #[test]
    fn test_endomap_mutual_survival_is_symmetric() {
        for n in 1..=3usize {
            for_each_self_map(n, |g| {
                let g = g.to_vec();
                for_each_self_map(n, |e| {
                    assert_eq!(
                        is_mutually_surviving(&g, e),
                        is_mutually_surviving(e, &g),
                        "asymmetry at g={g:?} e={e:?}"
                    );
                });
            });
        }
    }

    /// ★ Census of mutual vs one-sided survival over every pair of self-maps.
    ///
    /// Mutual survival `|W(E)| = 1/5/30/319` against one-sided `1/10/203/6,594` for
    /// `|E| = 1..4`. These counts were first obtained in a separate formalisation; this
    /// recomputes them in this crate's own kernel so the verification is not borrowed.
    #[test]
    fn test_endomap_mutual_survival_census_matches_reference_counts() {
        let expected_mutual = [1usize, 5, 30, 319];
        let expected_one_sided = [1usize, 10, 203, 6_594];
        for n in 1..=4usize {
            let mut mutual = 0usize;
            let mut one_sided = 0usize;
            for_each_self_map(n, |g| {
                let g = g.to_vec();
                for_each_self_map(n, |e| {
                    if survives(&g, e) {
                        one_sided += 1;
                    }
                    if is_mutually_surviving(&g, e) {
                        mutual += 1;
                    }
                });
            });
            assert_eq!(mutual, expected_mutual[n - 1], "|W(E)| mismatch at n={n}");
            assert_eq!(one_sided, expected_one_sided[n - 1], "one-sided mismatch at n={n}");
            if n >= 2 {
                assert!(mutual < one_sided, "n={n}: coupling must be strictly narrower");
            }
        }
    }

    /// The identity perturbs everything on one side, almost nothing mutually — this is *why* the two counts diverge.
    #[test]
    fn test_endomap_identity_survives_one_sided_but_rarely_mutually() {
        let n = 4usize;
        let id: Vec<usize> = (0..n).collect();
        let mut one_sided = 0usize;
        let mut mutual = 0usize;
        for_each_self_map(n, |g| {
            assert!(survives(g, &id), "id must be a perturbation of every g: {g:?}");
            one_sided += 1;
            if is_mutually_surviving(g, &id) {
                mutual += 1;
            }
        });
        assert_eq!(one_sided, 256, "all self-maps counted");
        assert!(mutual < one_sided, "mutual survival with id is strictly rarer");
    }

}
