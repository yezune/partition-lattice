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
/// Keeping it a lattice operation is the point (LOOM clean-room: no hand-written algorithm
/// where the lattice already says it).
///
/// ⚠ `integrate` (exact) is the right call here: this is a **structural** computation, not a
/// learning loop, so the ShallowJoin guidance (LBP-2) does not apply.
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
}
