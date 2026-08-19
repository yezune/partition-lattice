//! Simplicial homology over GF(2) for complexes derived from the partition lattice.
//!
//! Completes the incidence / homology substrate — alongside the Möbius *cumulant*
//! ([`crate::incidence::logical_cumulant`], over the Boolean lattice of a partition *family*)
//! and magnitude homology ([`crate::incidence::mh1_isolated_pairs`], over the d_⊕ metric) —
//! with the two remaining canonical Rota/order-theoretic invariants:
//!
//! 1. **General-dimension Betti numbers** β_k of an arbitrary finite simplicial complex, via
//!    GF(2) chain-complex ranks (the dit-ring D(U) = (P(dit), △, ∩) is GF(2) linear algebra, so
//!    ∂ = symmetric-difference-of-faces is the dit-ring chain complex).
//! 2. **Interval Möbius** μ(σ,τ) of Π(N) — the *lattice-interval* Möbius (distinct from the cumulant's
//!    *family* cumulant) — with the Rota/Philip-Hall identity μ(σ,τ) = χ̃(Δ(σ,τ)) (reduced Euler
//!    characteristic of the open interval's order complex). All A-region integers (no real-q χ_q,
//!    which is not admissible here).
//!
//! Theory: Alexandrov poset manifolds; see the accompanying formal proof.
//! Formal (Alexandrov structure): [alexandrov_poset_topology.dit] (42p/3d/0f).
//! External: McCord (1966, finite spaces ≃ order complexes) · Björner (1980, Π_n ≃ wedge of
//! (n−1)! spheres) · Rota (1964, Möbius/Euler). These classical theorems are REPRODUCED here as
//! verification; the novel L-VM content is the lattice-native reduction + the dit-ring GF(2)
//! homology grounding.
//!
//! Scope (honest): this grounds the *computation* of topological invariants (structural/math),
//! not a capability/learning claim. `interval_mobius`/order-complex over the full Π(n) enumerate
//! Bell(n) partitions → intended for small n; the `SimplicialComplex` homology itself is general.

use crate::unified::Partition;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::FloatExt;
use crate::HashMap;
#[allow(unused_imports)]
use alloc::borrow::ToOwned;
#[allow(unused_imports)]
use alloc::boxed::Box;
use alloc::collections::BTreeSet;
#[allow(unused_imports)]
use alloc::string::{String, ToString};
#[allow(unused_imports)]
use alloc::vec::Vec;

// ─────────────────────────────────────────────────────────────
// General-dimension GF(2) simplicial homology
// ─────────────────────────────────────────────────────────────

/// A finite abstract simplicial complex: `by_dim[k]` = the k-simplices, each a sorted `Vec` of
/// vertex indices. The complex is assumed downward-closed (every face of a simplex is present);
/// [`SimplicialComplex::from_graph`] enforces this for ≤2-dim inputs.
#[derive(Clone, Debug, Default)]
pub struct SimplicialComplex {
    by_dim: Vec<Vec<Vec<usize>>>,
}

impl SimplicialComplex {
    /// Build from explicit, downward-closed simplex lists grouped by dimension.
    pub fn from_dim_lists(by_dim: Vec<Vec<Vec<usize>>>) -> Self {
        SimplicialComplex { by_dim }
    }

    /// Build a ≤2-dimensional complex from vertices/edges/triangles, enforcing downward closure
    /// (all triangle edges and all edge vertices are added). Vertices `0..n_verts` are included.
    pub fn from_graph(n_verts: usize, edges: &[[usize; 2]], tris: &[[usize; 3]]) -> Self {
        let mut v: BTreeSet<usize> = (0..n_verts).collect();
        let mut e: BTreeSet<[usize; 2]> = BTreeSet::new();
        let mut t: BTreeSet<[usize; 3]> = BTreeSet::new();
        let sort2 = |a: usize, b: usize| if a < b { [a, b] } else { [b, a] };
        for ed in edges {
            let s = sort2(ed[0], ed[1]);
            v.insert(s[0]);
            v.insert(s[1]);
            e.insert(s);
        }
        for tr in tris {
            let mut m = *tr;
            m.sort_unstable();
            t.insert(m);
            v.insert(m[0]);
            v.insert(m[1]);
            v.insert(m[2]);
            e.insert(sort2(m[0], m[1]));
            e.insert(sort2(m[1], m[2]));
            e.insert(sort2(m[0], m[2]));
        }
        let by_dim = vec![
            v.iter().map(|&x| vec![x]).collect(),
            e.iter().map(|s| s.to_vec()).collect(),
            t.iter().map(|s| s.to_vec()).collect(),
        ];
        SimplicialComplex { by_dim }
    }

    /// Number of k-simplices f_k.
    pub fn f(&self, k: usize) -> usize {
        self.by_dim.get(k).map(|v| v.len()).unwrap_or(0)
    }

    /// Highest present dimension.
    pub fn max_dim(&self) -> usize {
        self.by_dim.len().saturating_sub(1)
    }

    /// rank(∂_k): C_k → C_{k−1} over GF(2). Each k-simplex's boundary = Σ (drop-one) faces.
    pub fn boundary_rank(&self, k: usize) -> usize {
        if k == 0 || k >= self.by_dim.len() {
            return 0;
        }
        let faces = &self.by_dim[k - 1];
        let face_idx: HashMap<&Vec<usize>, usize> =
            faces.iter().enumerate().map(|(i, f)| (f, i)).collect();
        let cols: Vec<BTreeSet<usize>> = self.by_dim[k]
            .iter()
            .map(|simp| {
                let mut col = BTreeSet::new();
                for drop in 0..simp.len() {
                    let face: Vec<usize> = simp
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != drop)
                        .map(|(_, &v)| v)
                        .collect();
                    let ri = *face_idx
                        .get(&face)
                        .expect("complex not downward-closed: missing face");
                    if !col.remove(&ri) {
                        col.insert(ri);
                    }
                }
                col
            })
            .collect();
        gf2_rank_sparse(cols)
    }

    /// k-th Betti number β_k = |C_k| − rank(∂_k) − rank(∂_{k+1}) (a GF(2) dimension, ≥ 0).
    pub fn betti(&self, k: usize) -> i64 {
        self.f(k) as i64 - self.boundary_rank(k) as i64 - self.boundary_rank(k + 1) as i64
    }

    /// All Betti numbers β_0..β_maxdim.
    pub fn betti_numbers(&self) -> Vec<i64> {
        (0..=self.max_dim()).map(|k| self.betti(k)).collect()
    }

    /// Integer Euler characteristic χ = Σ (−1)^k f_k (A-region; plain face counts, no real q).
    pub fn euler(&self) -> i64 {
        (0..self.by_dim.len())
            .map(|k| {
                let f = self.f(k) as i64;
                if k % 2 == 0 {
                    f
                } else {
                    -f
                }
            })
            .sum()
    }

    /// Reduced Euler characteristic χ̃ = χ − 1 (= Rota/Hall interval Möbius μ(σ,τ)).
    pub fn reduced_euler(&self) -> i64 {
        self.euler() - 1
    }
}

/// Sparse GF(2) matrix rank: each column = the set of row-indices that are 1. Gaussian elimination
/// keyed by each column's lowest row (symmetric-difference = GF(2) add = the dit-ring △).
pub fn gf2_rank_sparse(cols: Vec<BTreeSet<usize>>) -> usize {
    let mut pivots: HashMap<usize, BTreeSet<usize>> = HashMap::new();
    let mut rank = 0;
    for mut col in cols {
        while let Some(&low) = col.iter().next() {
            if let Some(p) = pivots.get(&low) {
                for &r in p {
                    if !col.remove(&r) {
                        col.insert(r);
                    }
                }
            } else {
                break;
            }
        }
        if let Some(&low) = col.iter().next() {
            pivots.insert(low, col);
            rank += 1;
        }
    }
    rank
}

// ─────────────────────────────────────────────────────────────
// Partition lattice Π(n): enumeration, order, order complex, interval Möbius
// ─────────────────────────────────────────────────────────────

/// All partitions of `[n]` (restricted-growth-string enumeration → canonical [`Partition`]).
pub fn all_partitions(n: usize) -> Vec<Partition> {
    let mut out = Vec::new();
    if n == 0 {
        return out;
    }
    let mut rgs = vec![0u32; n];
    loop {
        out.push(Partition::from_ids(rgs.clone()));
        let mut i = n - 1;
        loop {
            if i == 0 {
                return out;
            }
            let m = rgs[..i].iter().copied().max().unwrap_or(0);
            if rgs[i] <= m {
                rgs[i] += 1;
                for j in (i + 1)..n {
                    rgs[j] = 0;
                }
                break;
            }
            i -= 1;
        }
    }
}

/// `a ⊑ b` (a below b in refinement order where finer = higher): b refines a.
/// 0̂ = coarsest (1 block) = bottom; 1̂ = finest (n singletons) = top.
fn below(a: &Partition, b: &Partition) -> bool {
    b.refines(a)
}
fn strictly_below(a: &Partition, b: &Partition) -> bool {
    below(a, b) && a.get_ids() != b.get_ids()
}

/// Indices of the bottom (single block) and top (all singletons) of Π(n) within `parts`.
pub fn bottom_top(parts: &[Partition], n: usize) -> Option<(usize, usize)> {
    let bot = parts.iter().position(|p| p.block_count() == 1)?;
    let top = parts.iter().position(|p| p.block_count() == n)?;
    Some((bot, top))
}

/// Order complex of the OPEN interval (parts\[lo\], parts\[hi\]): vertices = elements strictly
/// between, simplices = chains. Downward-closed (every sub-chain is present).
pub fn order_complex_open_interval(parts: &[Partition], lo: usize, hi: usize) -> SimplicialComplex {
    let verts: Vec<usize> = (0..parts.len())
        .filter(|&x| strictly_below(&parts[lo], &parts[x]) && strictly_below(&parts[x], &parts[hi]))
        .collect();
    let vidx: HashMap<usize, usize> = verts.iter().enumerate().map(|(i, &e)| (e, i)).collect();
    let mut order: Vec<usize> = verts.clone();
    order.sort_by_key(|&e| parts[e].block_count());

    let mut by_dim: Vec<Vec<Vec<usize>>> = Vec::new();
    fn dfs(
        chain: &mut Vec<usize>,
        last: usize,
        order: &[usize],
        parts: &[Partition],
        vidx: &HashMap<usize, usize>,
        by_dim: &mut Vec<Vec<Vec<usize>>>,
    ) {
        let dim = chain.len() - 1;
        while by_dim.len() <= dim {
            by_dim.push(Vec::new());
        }
        let mut simplex: Vec<usize> = chain.iter().map(|e| vidx[e]).collect();
        simplex.sort_unstable();
        by_dim[dim].push(simplex);
        for &e in order {
            if strictly_below(&parts[last], &parts[e]) {
                chain.push(e);
                dfs(chain, e, order, parts, vidx, by_dim);
                chain.pop();
            }
        }
    }
    for &start in &order {
        let mut chain = vec![start];
        dfs(&mut chain, start, &order, parts, &vidx, &mut by_dim);
    }
    SimplicialComplex::from_dim_lists(by_dim)
}

/// Interval Möbius μ(parts\[lo\], parts\[hi\]) via the recursive definition μ(x,x)=1,
/// μ(x,y)=−Σ_{x≤z<y} μ(x,z), computed bottom-up (no recomputation). By Rota/Hall this equals
/// the reduced Euler characteristic of the open interval's order complex.
pub fn interval_mobius(parts: &[Partition], lo: usize, hi: usize) -> i64 {
    if parts[lo].get_ids() == parts[hi].get_ids() {
        return 1;
    }
    // elements z with lo ⊑ z ⊑ hi, in increasing rank order
    let mut zs: Vec<usize> = (0..parts.len())
        .filter(|&z| below(&parts[lo], &parts[z]) && below(&parts[z], &parts[hi]))
        .collect();
    zs.sort_by_key(|&z| parts[z].block_count());
    let mut mu: HashMap<usize, i64> = HashMap::new();
    for &z in &zs {
        if parts[z].get_ids() == parts[lo].get_ids() {
            mu.insert(z, 1);
            continue;
        }
        let mut acc = 0i64;
        for &y in &zs {
            if strictly_below(&parts[y], &parts[z]) {
                acc += mu[&y];
            }
        }
        mu.insert(z, -acc);
    }
    mu[&hi]
}

/// β₀ (number of connected components) of a graph via the native lattice connectivity: Join
/// (= Union-Find transitive closure) of the per-edge partitions, then `block_count`. This is the
/// SSOT for connectivity and cross-checks the homological β₀ = |V| − rank(∂₁).
pub fn betti0_via_coarsen(n_verts: usize, edges: &[[usize; 2]]) -> usize {
    if n_verts == 0 {
        return 0;
    }
    let mut acc = Partition::from_ids((0..n_verts as u32).collect());
    for e in edges {
        let mut ids: Vec<u32> = (0..n_verts as u32).collect();
        ids[e[1]] = ids[e[0]];
        acc = acc.integrate(&Partition::from_ids(ids));
    }
    acc.block_count()
}

// ─────────────────────────────────────────────────────────────
// Canonical finite complexes
// ─────────────────────────────────────────────────────────────

/// Hollow circle S¹ = cycle graph C_n (n ≥ 3): β = (1, 1), χ = 0.
pub fn circle(n: usize) -> SimplicialComplex {
    let edges: Vec<[usize; 2]> = (0..n).map(|i| [i, (i + 1) % n]).collect();
    SimplicialComplex::from_graph(n, &edges, &[])
}

/// Filled disk D² = fan-triangulation of an n-gon (n ≥ 3): β = (1, 0), χ = 1.
pub fn disk_fan(n: usize) -> SimplicialComplex {
    assert!(n >= 3);
    let mut edges: Vec<[usize; 2]> = (0..n).map(|i| [i, (i + 1) % n]).collect();
    let mut tris = Vec::new();
    for k in 1..(n - 1) {
        edges.push([0, k + 1]);
        tris.push([0, k, k + 1]);
    }
    SimplicialComplex::from_graph(n, &edges, &tris)
}

/// Boundary of a tetrahedron = 2-sphere S²: β = (1, 0, 1), χ = 2.
pub fn sphere_tetra() -> SimplicialComplex {
    SimplicialComplex::from_graph(
        4,
        &[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        &[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn factorial(k: usize) -> i64 {
        (1..=k as i64).product::<i64>().max(1)
    }

    // ── discriminator (circle / disk / disconnected), oracle = classical topology ──

    #[test]
    fn t_pdn_topo_discriminator() {
        let circle = circle(6);
        let disk = disk_fan(6);
        assert_eq!(
            (circle.betti(0), circle.betti(1)),
            (1, 1),
            "circle: 1 component, 1 hole"
        );
        assert_eq!(
            (disk.betti(0), disk.betti(1)),
            (1, 0),
            "disk: 1 component, 0 holes"
        );
        // disconnected: two separate edges
        let disc = SimplicialComplex::from_graph(4, &[[0, 1], [2, 3]], &[]);
        assert_eq!(
            (disc.betti(0), disc.betti(1)),
            (2, 0),
            "disconnected: 2 components"
        );
    }

    #[test]
    fn t_sphere_has_a_void() {
        let s = sphere_tetra();
        assert_eq!(s.betti_numbers(), vec![1, 0, 1], "S²: β = (1,0,1)");
        assert_eq!(s.euler(), 2, "χ(S²) = 2");
    }

    #[test]
    fn t_filling_hole_flips_beta1() {
        assert_eq!(circle(7).betti(1), 1);
        assert_eq!(disk_fan(7).betti(1), 0);
    }

    #[test]
    fn t_native_beta0_matches_homological() {
        let edges = [[0usize, 1], [1, 2], [2, 0], [3, 4]];
        let c = SimplicialComplex::from_graph(5, &edges, &[]);
        assert_eq!(
            betti0_via_coarsen(5, &edges) as i64,
            c.betti(0),
            "Join β₀ = homological β₀"
        );
    }

    // ── Π(N) itself: Björner wedge-of-spheres + Rota interval Möbius + poset-manifold verdict ──

    /// Björner (1980): Δ(Π̄_n) ≃ ⋁_{(n−1)!} S^{n−3} ⟹ REDUCED top-Betti β̃_{n−3} = (n−1)!.
    /// (For n=3, d=0: wedge of 2! S⁰ = 3 points, so unreduced β₀=3 but reduced β̃₀=2=2!.)
    #[test]
    fn t_bjorner_wedge_of_spheres() {
        for n in 3..=5 {
            let parts = all_partitions(n);
            let (bot, top) = bottom_top(&parts, n).unwrap();
            let oc = order_complex_open_interval(&parts, bot, top);
            let d = n - 3;
            // reduced Betti: β̃_0 = β_0 − 1, β̃_k = β_k for k ≥ 1
            let reduced_top = oc.betti(d) - if d == 0 { 1 } else { 0 };
            assert_eq!(
                reduced_top,
                factorial(n - 1),
                "Π_{}: β̃_{} = {}!",
                n,
                d,
                n - 1
            );
            // all lower reduced Betti vanish (genuine wedge of top-dim spheres)
            for k in 0..d {
                let red = oc.betti(k) - if k == 0 { 1 } else { 0 };
                assert_eq!(red, 0, "Π_{}: β̃_{} = 0", n, k);
            }
        }
    }

    /// Rota/Hall: interval Möbius μ(σ,τ) = χ̃(Δ(σ,τ)); and μ(Π_n) = (−1)^{n−1}(n−1)!.
    #[test]
    fn t_rota_mobius_equals_reduced_euler_and_closed_form() {
        for n in 2..=5 {
            let parts = all_partitions(n);
            let (bot, top) = bottom_top(&parts, n).unwrap();
            let mu = interval_mobius(&parts, bot, top);
            let sign = if (n - 1) % 2 == 0 { 1 } else { -1 };
            assert_eq!(
                mu,
                sign * factorial(n - 1),
                "μ(Π_{}) = (−1)^{}·{}!",
                n,
                n - 1,
                n - 1
            );
            if n >= 3 {
                let oc = order_complex_open_interval(&parts, bot, top);
                assert_eq!(mu, oc.reduced_euler(), "Π_{}: μ = χ̃ (Rota)", n);
            }
        }
    }

    /// General interval Möbius = reduced Euler over many sub-intervals (not just the whole lattice).
    #[test]
    fn t_interval_mobius_general() {
        let n = 5;
        let parts = all_partitions(n);
        let mut checked = 0;
        for lo in 0..parts.len() {
            for hi in 0..parts.len() {
                if lo != hi && strictly_below(&parts[lo], &parts[hi]) {
                    let mu = interval_mobius(&parts, lo, hi);
                    let oc = order_complex_open_interval(&parts, lo, hi);
                    assert_eq!(mu, oc.reduced_euler(), "μ(σ,τ) = χ̃ for [{},{}]", lo, hi);
                    checked += 1;
                }
            }
        }
        assert!(checked > 50, "checked {} intervals", checked);
    }

    /// Poset-manifold verdict: top-Betti (n−1)! > 1 for n≥4 ⟹ Π_n is NOT a closed manifold
    /// (closed connected d-manifold has β_d ≤ 1); Π_3 proper part = 3 points = 0-manifold.
    #[test]
    fn t_partition_lattice_is_not_a_manifold() {
        let p3 = all_partitions(3);
        let (b3, t3) = bottom_top(&p3, 3).unwrap();
        let oc3 = order_complex_open_interval(&p3, b3, t3);
        assert_eq!(oc3.betti(0), 3, "Π_3 proper part = 3 points (0-manifold)");
        assert_eq!(oc3.max_dim(), 0);
        for n in 4..=5 {
            let parts = all_partitions(n);
            let (bot, top) = bottom_top(&parts, n).unwrap();
            let oc = order_complex_open_interval(&parts, bot, top);
            assert!(
                oc.betti(n - 3) > 1,
                "Π_{}: top Betti > 1 ⇒ not a manifold",
                n
            );
        }
    }

    /// Euler–Poincaré χ = Σ(−1)^k β_k (internal correctness of the GF(2) ranks).
    #[test]
    fn t_euler_poincare() {
        for n in 3..=5 {
            let parts = all_partitions(n);
            let (bot, top) = bottom_top(&parts, n).unwrap();
            let oc = order_complex_open_interval(&parts, bot, top);
            let alt: i64 = (0..=oc.max_dim())
                .map(|k| {
                    if k % 2 == 0 {
                        oc.betti(k)
                    } else {
                        -oc.betti(k)
                    }
                })
                .sum();
            assert_eq!(alt, oc.euler(), "χ = Σ(−1)^k β_k for Π_{}", n);
        }
        // and for the canonical complexes
        for c in [circle(5), disk_fan(6), sphere_tetra()] {
            let alt: i64 = (0..=c.max_dim())
                .map(|k| if k % 2 == 0 { c.betti(k) } else { -c.betti(k) })
                .sum();
            assert_eq!(alt, c.euler());
        }
    }

    #[test]
    fn t_bell_numbers_enumeration() {
        // sanity: |Π(n)| = Bell(n): 1,2,5,15,52
        assert_eq!(all_partitions(1).len(), 1);
        assert_eq!(all_partitions(2).len(), 2);
        assert_eq!(all_partitions(3).len(), 5);
        assert_eq!(all_partitions(4).len(), 15);
        assert_eq!(all_partitions(5).len(), 52);
    }
}
