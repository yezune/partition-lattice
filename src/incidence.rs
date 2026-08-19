//! Incidence-algebra coordinates on the partition lattice.
//!
//! Production SSOT for the three "sister coordinates" grounded on the Π(U) Hasse +
//! dit-set incidence substrate — exact integers, with no floats and no probabilities.
//!
//! These are *structural-store analysis* utilities (descriptive and diagnostic);
//! per the grounding measurements, the dit-metric saturates on real text, so the proven
//! utility regime is *structural partition stores*, not real-text pipelines.
//!
//! - **Logical Möbius cumulant**:
//!   [`dit_intersection_count`], [`logical_cumulant`], [`is_logically_free`].
//!   κ_S = (−1)^{|S|−1}·|⋂ dit(π_i)|/N² (signed k-fold dit-intersection).
//! - **Logical magnitude homology**:
//!   [`dit_xor_count`], [`geodesic_excess`], [`is_modular_pair`], [`mh1_isolated_pairs`].
//!   Meet/Join = d_⊕-geodesic interpolants.
//! - **Logical Morse compression**:
//!   [`refine_coarsen_closure`], [`critical_generator_base`], [`regeneration_loss`].
//!   Lossless store compression to its canonical irreducible generating base.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::FloatExt;
use crate::HashSet;
use crate::Partition;
#[allow(unused_imports)]
use alloc::borrow::ToOwned;
#[allow(unused_imports)]
use alloc::boxed::Box;
#[allow(unused_imports)]
use alloc::string::{String, ToString};
#[allow(unused_imports)]
use alloc::vec::Vec;

/// dit-profile key (block-id assignment) — label-blind identity for de-dup / membership.
fn key(p: &Partition) -> Vec<u32> {
    p.get_ids().to_vec()
}

/// |dit(p)| = number of ordered distinction pairs = N² − Σ|B|² (exact integer).
#[inline]
pub fn dit_count(p: &Partition) -> i64 {
    p.distinction_count() as i64
}

/// |dit(a) △ dit(b)| = ordered symmetric-difference count = 2·|dit(a∧b)| − |dit(a)| − |dit(b)|.
///
/// This is the integer numerator of the ditset-XOR metric d_⊕(a,b) = |dit△dit|/N².
#[inline]
pub fn dit_xor_count(a: &Partition, b: &Partition) -> i64 {
    2 * dit_count(&a.differentiate(b)) - dit_count(a) - dit_count(b)
}

/// **Destroyed distinctions** `destr(a, b) = |dit(a) ∩ indit(b)|` — how many distinctions `a`
/// makes that `b` erases.
///
/// Equals `|dit(a)| − |dit(a) ∩ dit(b)|`, so it delegates to the two counting functions above.
///
/// Two identities follow:
/// - `destr(a, b) == 0` ⟺ `b.refines(a)` (dit inclusion is refinement)
/// - `creat(a, b) − destr(a, b) == dit_count(b) − dit_count(a)` = `N²·(h(b) − h(a))`
///
/// **Why it is needed.** `Δh` is a *net* change that conflates creation with destruction. A
/// lattice automorphism (induced by a permutation) preserves the multiset of block sizes, so it
/// leaves `h` invariant *by construction* — while `destr` still sees the move.
///
/// Derived from distinction counts alone — no probability, no logarithm. ⚠ A single scalar is
/// non-injective, so do not use this alone as a discriminator: the informative object is the
#[inline]
pub fn destr(a: &Partition, b: &Partition) -> i64 {
    dit_count(a) - dit_intersection_count(&[a, b])
}

/// **Created distinctions** `creat(a, b) = |dit(b) ∩ indit(a)|` — how many distinctions `b` adds.
///
/// `creat(a, b) == destr(b, a)` (the transpose). See [`destr`] for details.
#[inline]
pub fn creat(a: &Partition, b: &Partition) -> i64 {
    destr(b, a)
}

/// |⋂_{p∈parts} dit(p)| — ordered pairs separated by *every* partition in `parts`.
///
/// the k-fold dit-intersection. Empty / singleton `parts` handled by convention
/// (|dit| of the single partition; 0 for empty).
pub fn dit_intersection_count(parts: &[&Partition]) -> i64 {
    match parts {
        [] => 0,
        [p] => dit_count(p),
        _ => {
            let n = parts[0].get_ids().len();
            let mut cnt = 0i64;
            for x in 0..n {
                for y in 0..n {
                    if x != y && parts.iter().all(|p| p.get_ids()[x] != p.get_ids()[y]) {
                        cnt += 1;
                    }
                }
            }
            cnt
        }
    }
}

/// Logical Möbius cumulant κ_S (integer numerator, over N²) of a partition family.
///
/// Closed form: **κ_S = (−1)^{|S|−1}·|⋂_{i∈S} dit(π_i)|**, the signed k-fold dit-intersection.
/// Pair → −I₂ ≤ 0 (subadditivity); triple → +|∩₃| ≥ 0; κ_S = 0 ⟺ logically free.
/// Use the *family vector* {κ_S}, not a single scalar, for discrimination (A4 / YD-1).
pub fn logical_cumulant(parts: &[&Partition]) -> i64 {
    let inter = dit_intersection_count(parts);
    if parts.len() % 2 == 1 {
        inter
    } else {
        -inter
    }
}

/// `true` iff the family is *logically free* at its order: ⋂ dit(π_i) = ∅ (κ_S = 0).
pub fn is_logically_free(parts: &[&Partition]) -> bool {
    parts.len() >= 2 && dit_intersection_count(parts) == 0
}

/// Geodesic excess of `via` on the d_⊕-segment a—b: d_⊕(a,via)+d_⊕(via,b) − d_⊕(a,b) ≥ 0.
///
/// `via` lies on a d_⊕-geodesic between a,b ⟺ excess = 0. Theorem 1: `a∧b` (Meet)
/// always has excess 0 (geodesic interpolant). Join has excess 0 ⟺ the pair is modular.
pub fn geodesic_excess(a: &Partition, via: &Partition, b: &Partition) -> i64 {
    dit_xor_count(a, via) + dit_xor_count(via, b) - dit_xor_count(a, b)
}

/// `true` iff (a,b) is a *modular pair*: |dit(a)|+|dit(b)| = |dit(a∧b)|+|dit(a∨b)|.
///
/// Theorem 2: Join a∨b lies on a d_⊕-geodesic between a,b ⟺ this holds.
/// h is generally submodular; equality fails on non-modular (non-distributive) pairs.
pub fn is_modular_pair(a: &Partition, b: &Partition) -> bool {
    dit_count(a) + dit_count(b) == dit_count(&a.differentiate(b)) + dit_count(&a.integrate(b))
}

/// Magnitude-homology MH₁ descriptor of a store: for each geodesic length ℓ, the number of
/// distance-ℓ pairs with *no interior store point* (geodesically-isolated = Hasse covering).
///
/// MH₀,₀ = |store|; this returns the length-graded MH₁ rank vector (sorted by ℓ).
pub fn mh1_isolated_pairs(store: &[Partition]) -> Vec<(i64, u64)> {
    use alloc::collections::BTreeMap;
    let mut out: BTreeMap<i64, u64> = BTreeMap::new();
    for (i, a) in store.iter().enumerate() {
        for (k, b) in store.iter().enumerate() {
            if i == k {
                continue;
            }
            let l = dit_xor_count(a, b);
            if l == 0 {
                continue;
            }
            let isolated = !store
                .iter()
                .enumerate()
                .any(|(j, c)| j != i && j != k && geodesic_excess(a, c, b) == 0);
            if isolated {
                *out.entry(l).or_insert(0) += 1;
            }
        }
    }
    out.into_iter().collect()
}

/// Meet/Join-closure of `gen` — the smallest sublattice (by dit-profile) containing it,
/// capped at `cap` elements (for tractability on large universes).
///
/// the closure operator under which a store's canonical generating base regenerates it.
pub fn refine_coarsen_closure(gen: &[Partition], cap: usize) -> Vec<Partition> {
    let mut set: Vec<Partition> = Vec::new();
    let mut keys: HashSet<Vec<u32>> = HashSet::new();
    for g in gen {
        if keys.insert(key(g)) {
            set.push(g.clone());
        }
    }
    loop {
        let mut added = false;
        let snap = set.clone();
        'o: for i in 0..snap.len() {
            for j in 0..snap.len() {
                if i == j {
                    continue;
                }
                for m in [snap[i].differentiate(&snap[j]), snap[i].integrate(&snap[j])] {
                    if keys.insert(key(&m)) {
                        set.push(m);
                        added = true;
                        if set.len() >= cap {
                            break 'o;
                        }
                    }
                }
            }
        }
        if !added || set.len() >= cap {
            break;
        }
    }
    set
}

/// Regeneration loss: number of `store` cells whose dit-profile is NOT in `closure(gen)`.
/// Zero ⟺ `gen` is a lossless generating base of `store`.
pub fn regeneration_loss(gen: &[Partition], store: &[Partition], cap: usize) -> usize {
    let cl = refine_coarsen_closure(gen, cap);
    let ck: HashSet<Vec<u32>> = cl.iter().map(key).collect();
    store.iter().filter(|s| !ck.contains(&key(s))).count()
}

/// Canonical irreducible generating base G ⊆ store (the discrete-Morse critical cells).
///
/// Greedily drops dit-reconstructible (matched) cells, keeping only the closure-irreducible
/// (μ-residue) generators. Guarantees `regeneration_loss(&G, store, cap) == 0` by construction.
/// `|G| < |store|` exactly when the store is implicationally redundant (compressible).
/// **Load-bearing** (Cohen d 3.89): an arbitrary same-size subset fails to
/// regenerate, so the selection is genuine lossless compression, not store-recall repackaging.
pub fn critical_generator_base(store: &[Partition], cap: usize) -> Vec<Partition> {
    let mut g: Vec<Partition> = store.to_vec();
    loop {
        let mut dropped = false;
        for idx in 0..g.len() {
            let trial: Vec<Partition> = g
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, p)| p.clone())
                .collect();
            if regeneration_loss(&trial, store, cap) == 0 {
                g = trial;
                dropped = true;
                break;
            }
        }
        if !dropped {
            break;
        }
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(ids: &[u32]) -> Partition {
        Partition::from_ids(ids.to_vec())
    }

    #[test]
    fn test_incidence_dit_xor_matches_metric() {
        let a = p(&[0, 0, 1, 1]);
        let b = p(&[0, 1, 0, 1]);
        // d_xor_count = ditset_xor_distance · N² (integer == float·16).
        let n2 = 16.0;
        assert!((dit_xor_count(&a, &b) as f64 - a.ditset_xor_distance(&b) * n2).abs() < 1e-9);
        // dc: a=8, b=8, a∧b=discrete=12.
        assert_eq!(
            (
                dit_count(&a),
                dit_count(&b),
                dit_count(&a.differentiate(&b))
            ),
            (8, 8, 12)
        );
    }

    #[test]
    fn test_incidence_cumulant_closed_form() {
        let a = p(&[0, 0, 1, 1]);
        let b = p(&[0, 1, 0, 1]);
        // κ_pair = −|dit∩dit| ≤ 0; here dit∩ = 4 → κ = −4.
        assert_eq!(logical_cumulant(&[&a, &b]), -4);
        assert_eq!(
            logical_cumulant(&[&a, &b]),
            -dit_intersection_count(&[&a, &b])
        );
        // subadditivity: |dit(a∧b)| ≤ |dit(a)|+|dit(b)| ⟺ κ ≤ 0.
        assert!(logical_cumulant(&[&a, &b]) <= 0);
    }

    #[test]
    fn test_incidence_meet_geodesic_always() {
        // Theorem 1: Meet a∧b always lies on the d_⊕-geodesic (excess 0).
        let a = p(&[0, 0, 1, 1]);
        let b = p(&[0, 1, 0, 1]);
        assert_eq!(geodesic_excess(&a, &a.differentiate(&b), &b), 0);
        // Theorem 2: this pair is non-modular → Join NOT on geodesic.
        assert!(!is_modular_pair(&a, &b));
        assert!(geodesic_excess(&a, &a.integrate(&b), &b) > 0);
    }

    #[test]
    fn test_incidence_critical_base_lossless_and_load_bearing() {
        // Redundant store: {a, b, a∧b, a∨b} (a∧b, a∨b reconstructible from a,b).
        let a = p(&[0, 0, 1, 1]);
        let b = p(&[0, 1, 0, 1]);
        let store = vec![a.clone(), b.clone(), a.differentiate(&b), a.integrate(&b)];
        let g = critical_generator_base(&store, 64);
        // lossless by construction.
        assert_eq!(regeneration_loss(&g, &store, 64), 0);
        // compressible (a∧b, a∨b are matched/reducible).
        assert!(g.len() < store.len(), "redundant store compresses");
    }
}

#[cfg(test)]
mod destr_creat_tests {
    use super::*;
    use crate::Partition;

    fn p(ids: &[u32]) -> Partition {
        Partition::from_ids(ids.to_vec())
    }

    /// Identity (i) — `destr == 0` iff the other partition refines this one.
    ///
    /// In this crate's order: `b.refines(a)` ⟺ dit(b) ⊇ dit(a) ⟺ dit(a) ∩ indit(b) = ∅.
    #[test]
    fn test_incidence_destr_is_zero_iff_refines() {
        for n in 2..=5 {
            for a_ids in crate::incidence::tests_support::enumerate(n) {
                for b_ids in crate::incidence::tests_support::enumerate(n) {
                    let (a, b) = (p(&a_ids), p(&b_ids));
                    assert_eq!(
                        destr(&a, &b) == 0,
                        b.refines(&a),
                        "destr({a_ids:?},{b_ids:?}) = {} but refines = {}",
                        destr(&a, &b),
                        b.refines(&a)
                    );
                }
            }
        }
    }

    /// Identity (ii) — `|E|²·(h(b) − h(a)) = creat − destr`. Exact: it is an integer identity.
    #[test]
    fn test_incidence_creat_minus_destr_is_dit_delta() {
        for n in 2..=5 {
            for a_ids in crate::incidence::tests_support::enumerate(n) {
                for b_ids in crate::incidence::tests_support::enumerate(n) {
                    let (a, b) = (p(&a_ids), p(&b_ids));
                    assert_eq!(
                        creat(&a, &b) - destr(&a, &b),
                        dit_count(&b) - dit_count(&a),
                        "identity (ii) broke at {a_ids:?} / {b_ids:?}"
                    );
                }
            }
        }
    }

    /// `creat(a,b) = destr(b,a)` — the two components are transposes of one another.
    #[test]
    fn test_incidence_creat_is_transposed_destr() {
        for n in 2..=5 {
            for a_ids in crate::incidence::tests_support::enumerate(n) {
                for b_ids in crate::incidence::tests_support::enumerate(n) {
                    let (a, b) = (p(&a_ids), p(&b_ids));
                    assert_eq!(creat(&a, &b), destr(&b, &a));
                }
            }
        }
    }

    /// ★ The load-bearing check — an automorphism leaves `h` fixed, but `destr` still sees it.
    ///
    /// A permutation induces an order automorphism of Π(U), so block sizes — and hence `h` — are
    /// preserved. A check that looks only at Δh cannot detect the move **in principle**.
    #[test]
    fn test_incidence_destr_sees_what_h_cannot_under_automorphism() {
        let n = 4;
        let reverse: Vec<usize> = (0..n).rev().collect();
        let mut moved = 0usize;
        for ids in crate::incidence::tests_support::enumerate(n) {
            let a = p(&ids);
            let Some(b) = a.try_pullback_permutation(&reverse) else { continue };
            if a.get_ids() == b.get_ids() {
                continue; // fixed point: nothing moves, nothing to count
            }
            moved += 1;
            assert_eq!(dit_count(&a), dit_count(&b), "h must be invariant under automorphism");
            assert!(
                destr(&a, &b) > 0,
                "destr must be positive where h is blind: {ids:?}"
            );
        }
        assert!(moved > 0, "no partition was actually moved — the test would be vacuous");
    }
}

/// Test-only partition enumeration (restricted-growth strings); kept local so this crate
/// depends on nothing else.
#[cfg(test)]
pub(crate) mod tests_support {
    use alloc::vec::Vec;

    pub fn enumerate(n: usize) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        let mut cur = alloc::vec![0u32; n];
        rec(&mut cur, &mut out, 0, n, 0);
        out
    }

    fn rec(cur: &mut Vec<u32>, out: &mut Vec<Vec<u32>>, i: usize, n: usize, max: u32) {
        if i == n {
            out.push(cur.clone());
            return;
        }
        for b in 0..=max {
            cur[i] = b;
            rec(cur, out, i + 1, n, max);
        }
        cur[i] = max;
        rec(cur, out, i + 1, n, max + 1);
    }
}
