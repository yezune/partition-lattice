//! Tropical semiring operations and shortest-path algorithms.
//!
//! The (min, +) tropical semiring is used for D_L-guided proof navigation
//! and general shortest-path computation over partition lattice cost matrices.

use crate::unified::Partition;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::FloatExt;
#[allow(unused_imports)]
use alloc::borrow::ToOwned;
#[allow(unused_imports)]
use alloc::boxed::Box;
#[allow(unused_imports)]
use alloc::string::{String, ToString};
#[allow(unused_imports)]
use alloc::vec::Vec;

/// Tropical matrix multiplication: `C[i][j] = min_k (A[i][k] + B[k][j])`.
///
/// Implements the (min, +) semiring product used in shortest-path algorithms.
pub fn tropical_matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![f64::INFINITY; n]; n];

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let val = a[i][k] + b[k][j];
                if val < c[i][j] {
                    c[i][j] = val;
                }
            }
        }
    }
    c
}

/// Floyd-Warshall all-pairs shortest paths via tropical semiring.
///
/// Input: cost matrix (0 on diagonal, ∞ for no edge, weight for edges).
/// Output: shortest path distances for all pairs.
pub fn floyd_warshall(mat: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = mat.len();
    let mut dist = mat.to_vec();

    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let via_k = dist[i][k] + dist[k][j];
                if via_k < dist[i][j] {
                    dist[i][j] = via_k;
                }
            }
        }
    }
    dist
}

/// Build a D_L cost matrix from a set of partitions.
///
/// `mat[i][j]` = |D_L(π_i ‖ π_j)| = |h(π_i ∧ π_j) − h(π_j)|.
/// For i=j, cost = 0.
pub fn build_dl_cost_matrix(partitions: &[Partition]) -> Vec<Vec<f64>> {
    let n = partitions.len();
    let mut mat = vec![vec![f64::INFINITY; n]; n];

    for i in 0..n {
        mat[i][i] = 0.0;
        for j in 0..n {
            if i != j {
                let dl = partitions[i].logical_divergence(&partitions[j]).abs();
                mat[i][j] = dl;
            }
        }
    }
    mat
}

/// Boolean transitive closure (reachability) — the **INT-only dual** of
/// [`floyd_warshall`] for the case where only *reachability* (is there a path
/// `i ↝ j`?) is needed, not the numeric distance.
///
/// The (min, +) tropical closure and the boolean-semiring closure agree on the
/// reachability relation: `reachability_closure(adj)[i][j]` is `true` iff
/// `floyd_warshall(mat)[i][j].is_finite()` for `mat` built with `0` on the
/// diagonal, `1` for each edge, `∞` elsewhere. Callers that consume the tropical
/// closure *only* through `.is_finite()` (e.g. `tlw::federation_order_params`,
/// the per-alpha-tick coherence gate) can therefore delegate here to drop f64
/// entirely — a DCE-T4/LBP-1 win on FPU-less RISC-V targets where each f64
/// add/compare is tens of soft-float cycles.
///
/// `adj[i][j] == true` iff a direct edge `i → j` exists; the diagonal is always
/// treated as reachable (`i ↝ i`). Rows are bit-packed into `u64` words so the
/// Warshall pass is `O(n³ / 64)` INT OR-operations with zero allocation churn
/// inside the closure loop.
pub fn reachability_closure(adj: &[Vec<bool>]) -> Vec<Vec<bool>> {
    let n = adj.len();
    if n == 0 {
        return Vec::new();
    }
    let words = n.div_ceil(64);
    // rows[i] = bitset of nodes reachable from i (diagonal pre-set).
    let mut rows: Vec<Vec<u64>> = vec![vec![0u64; words]; n];
    for (i, row) in rows.iter_mut().enumerate() {
        row[i / 64] |= 1u64 << (i % 64); // i ↝ i
        for (j, &edge) in adj[i].iter().enumerate() {
            if edge {
                row[j / 64] |= 1u64 << (j % 64);
            }
        }
    }
    // Warshall: if i can reach k, i inherits everything k can reach.
    // Row k is invariant during pass k (the only writer would be i == k, a no-op
    // since k ↝ k is already set), so the clone is a stable snapshot.
    for k in 0..n {
        let (kw, kb) = (k / 64, 1u64 << (k % 64));
        let row_k = rows[k].clone();
        for row_i in rows.iter_mut() {
            if row_i[kw] & kb != 0 {
                for w in 0..words {
                    row_i[w] |= row_k[w];
                }
            }
        }
    }
    // Unpack to the dense bool matrix the callers expect.
    let mut out = vec![vec![false; n]; n];
    for (i, row_i) in rows.iter().enumerate() {
        for (j, out_ij) in out[i].iter_mut().enumerate() {
            *out_ij = row_i[j / 64] & (1u64 << (j % 64)) != 0;
        }
    }
    out
}

/// Global-time order parameters `(Q_scc, ρ̂)` for a federation of universes. From `n` nodes and
/// the **boolean reachability closure** of the directed `bridges` (delegated to [`reachability_closure`]):
/// - `Q_scc` = `(|largest SCC| − 1)/(n − 1)` ∈ `[0,1]` — zero iff the causal graph is acyclic.
/// - `ρ̂` = causal comparability = `|{ {i,j} : i↝j ∨ j↝i }| / C(n,2)` ∈ `[0,1]` — the connected fraction.
///
/// **A genuine global time ⟺ `Q_scc = 0 ∧ ρ̂ > 0`** — three cases: empty (0,0), DAG (0,>0), cycle (>0,·).
/// An SCC is a mutual-reachability class (`i↝j ∧ j↝i`).
///
/// ★ **Float-free:** the readout consumes the closure only as *reachability*, so instead of the
/// `(min,+)` `floyd_warshall` (f64, O(M³)) it delegates to the boolean transitive closure
/// [`reachability_closure`] (integer, O(M³/64)). Dropping the per-tick soft-float is what makes
/// this usable on an FPU-less target. (Consumers that need actual *distances* — not just
/// reachability — keep using [`floyd_warshall`].)
pub fn federation_order_params(n: usize, bridges: &[(usize, usize)]) -> (f64, f64) {
    if n <= 1 {
        return (0.0, 0.0);
    }
    let mut adj = vec![vec![false; n]; n];
    for &(i, j) in bridges {
        if i < n && j < n {
            adj[i][j] = true;
        }
    }
    let reach = reachability_closure(&adj);
                                            // Q_scc: size of the largest strongly connected component.
    let mut seen = vec![false; n];
    let mut largest = 1usize;
    for i in 0..n {
        if seen[i] {
            continue;
        }
        let mut comp = 1usize;
        seen[i] = true;
        for j in (i + 1)..n {
            if !seen[j] && reach[i][j] && reach[j][i] {
                seen[j] = true;
                comp += 1;
            }
        }
        largest = largest.max(comp);
    }
    let q_scc = (largest as f64 - 1.0) / (n as f64 - 1.0);
    // rho_hat: comparable (reachable either way) unordered pairs over C(n, 2).
    let (mut comparable, mut total) = (0usize, 0usize);
    for i in 0..n {
        for j in (i + 1)..n {
            total += 1;
            if reach[i][j] || reach[j][i] {
                comparable += 1;
            }
        }
    }
    let rho_hat = if total == 0 {
        0.0
    } else {
        comparable as f64 / total as f64
    };
    (q_scc, rho_hat)
}

/// Coherence regime of a federation — the decision rule that classifies `(Q_scc, ρ̂)` against a
/// threshold τ. The rule lives here, next to the measurement it classifies, so the two cannot
/// drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Coherence {
    /// `Q_scc = 0` (a global time exists) but `ρ̂ < τ` → partial binding: disconnected islands.
    Islands,
    /// `Q_scc = 0` and `ρ̂ ≥ τ` → a coherent "now".
    Bound,
    /// `Q_scc > 0` → a causal loop: no global time.
    Ctc,
}

/// Partition the nodes of a directed graph by **identical reachability set** —
/// `i ~ j` iff `reachability_closure(adj)[i] == reachability_closure(adj)[j]`.
///
/// The partition-valued sibling of [`reachability_closure`]: the closure answers
/// "who reaches whom", this answers "who is *indistinguishable by* who they
/// reach". Two structural facts make it a degeneracy detector for a bridge graph:
///
/// - **Saturated graph ⇒ indiscrete.** Every node reaching every node makes all
///   rows equal, so the result is the one-block partition and `h(π) = 0`.
/// - **Edgeless graph ⇒ discrete.** Each node reaches only itself, so all rows
///   differ and `h(π)` is maximal.
///
/// So `h(π) > 0` says the graph is *neither* saturated *nor* trivially
/// disconnected — it carries structure that tells its nodes apart.
///
/// INT-only (no f64 on this path — DCE-T4/LBP-1) and **bit-exact with the f64
/// route**: grouping by `floyd_warshall(mat)[i][j].is_finite()` patterns yields
/// the same partition (sealed by
/// `test_tropical_reachability_partition_matches_floyd_warshall_finiteness`).
/// Block ids are assigned in first-appearance order ⇒ deterministic per `adj`.
///
/// `adj[i][j] == true` iff a direct edge `i → j` exists; the diagonal is always
/// reachable. Empty input ⇒ empty partition.
pub fn reachability_partition(adj: &[Vec<bool>]) -> Partition {
    let n = adj.len();
    if n == 0 {
        return Partition::from_ids(Vec::new());
    }
    let reach = reachability_closure(adj);
    let mut ids = vec![0u32; n];
    let mut seen: Vec<&Vec<bool>> = Vec::new();
    for (i, row) in reach.iter().enumerate() {
        let id = match seen.iter().position(|s| *s == row) {
            Some(k) => k,
            None => {
                seen.push(row);
                seen.len() - 1
            }
        };
        ids[i] = id as u32;
    }
    Partition::from_ids(ids)
}

/// The classification rule `(Q_scc, ρ̂, τ) → Coherence`. `Q_scc > 1e-9` gives `Ctc`; otherwise
/// `ρ̂ ≥ τ` gives `Bound` and anything else `Islands`. τ is a policy the caller declares.
#[inline]
pub fn classify_coherence(q_scc: f64, rho_hat: f64, tau: f64) -> Coherence {
    if q_scc > 1e-9 {
        Coherence::Ctc
    } else if rho_hat >= tau {
        Coherence::Bound
    } else {
        Coherence::Islands
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 0/1/∞ reachability cost matrix (the shape `federation_order_params`
    /// feeds to `floyd_warshall`) and the equivalent boolean adjacency for
    /// `reachability_closure`, from the same edge set.
    fn build_pair(n: usize, edges: &[(usize, usize)]) -> (Vec<Vec<f64>>, Vec<Vec<bool>>) {
        let mut mat = vec![vec![f64::INFINITY; n]; n];
        let mut adj = vec![vec![false; n]; n];
        for (i, row) in mat.iter_mut().enumerate() {
            row[i] = 0.0;
        }
        for &(i, j) in edges {
            if i < n && j < n {
                mat[i][j] = 1.0;
                adj[i][j] = true;
            }
        }
        (mat, adj)
    }

    /// `reachability_closure` must produce the SAME reachability relation as
    /// `floyd_warshall(...).is_finite()` — this is the invariant that lets
    /// `federation_order_params` swap the f64 closure for the INT one while
    /// keeping `(q_scc, ρ̂)` bit-exact.
    #[test]
    fn t_classify_coherence_rule() {
        // Ctc iff Q_scc>0 (dominates); else Bound iff ρ̂≥τ; else Islands.
        assert_eq!(classify_coherence(0.5, 0.9, 0.5), Coherence::Ctc); // cycle wins over ρ̂
        assert_eq!(classify_coherence(0.0, 0.9, 0.5), Coherence::Bound); // DAG + connected
        assert_eq!(classify_coherence(0.0, 0.3, 0.5), Coherence::Islands); // DAG + under-connected
        assert_eq!(classify_coherence(0.0, 0.5, 0.5), Coherence::Bound); // ρ̂==τ boundary → Bound
        assert_eq!(classify_coherence(0.0, 0.0, 0.5), Coherence::Islands); // empty
        assert_eq!(classify_coherence(1e-10, 0.9, 0.5), Coherence::Bound); // below 1e-9 → not Ctc
    }

    #[test]
    fn t_reachability_closure_matches_floyd_finiteness() {
        // Deterministic pseudo-random graphs (no external RNG — vary by index).
        let topos: Vec<(usize, Vec<(usize, usize)>)> = vec![
            (1, vec![]),
            (4, vec![]),
            (4, vec![(0, 1), (1, 2), (2, 3), (3, 0)]), // 4-cycle
            (4, vec![(0, 1), (1, 2), (2, 3)]),         // chain / DAG
            (5, vec![(0, 1), (1, 2), (2, 0), (3, 4)]), // SCC{0,1,2} + edge
            (6, vec![(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]),
            (2, vec![(0, 1), (1, 0)]),
            (
                7,
                vec![(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (6, 6)],
            ),
        ];
        for (n, edges) in &topos {
            let (mat, adj) = build_pair(*n, edges);
            let dist = floyd_warshall(&mat);
            let reach = reachability_closure(&adj);
            for i in 0..*n {
                for j in 0..*n {
                    assert_eq!(
                        reach[i][j],
                        dist[i][j].is_finite(),
                        "reach[{i}][{j}] ≠ floyd finiteness (n={n}, edges={edges:?})"
                    );
                }
            }
        }
    }

    /// LCG-driven random edge sets over a range of `n` (incl. sizes that cross
    /// the 64-bit bitpack word boundary) — stress the equivalence at scale.
    #[test]
    fn t_reachability_closure_matches_floyd_random() {
        let mut state: u64 = 0x9E3779B97F4A7C15; // fixed seed (no Date/rand)
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        for &n in &[3usize, 8, 16, 33, 65, 70] {
            for _trial in 0..8 {
                let mut edges = Vec::new();
                let density = (next() % 4) as usize + 1; // 1..=4 edges per node avg
                for _ in 0..(n * density) {
                    let i = (next() as usize) % n;
                    let j = (next() as usize) % n;
                    edges.push((i, j));
                }
                let (mat, adj) = build_pair(n, &edges);
                let dist = floyd_warshall(&mat);
                let reach = reachability_closure(&adj);
                for i in 0..n {
                    for j in 0..n {
                        assert_eq!(
                            reach[i][j],
                            dist[i][j].is_finite(),
                            "mismatch reach[{i}][{j}] (n={n})"
                        );
                    }
                }
            }
        }
    }

    /// Diagonal is always reachable (i reaches itself), independent of edges.
    #[test]
    fn t_reachability_closure_diagonal_reachable() {
        let adj = vec![vec![false; 5]; 5];
        let reach = reachability_closure(&adj);
        for i in 0..5 {
            assert!(reach[i][i], "i reaches itself");
            for j in 0..5 {
                if i != j {
                    assert!(!reach[i][j], "no edges ⇒ no off-diagonal reach");
                }
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // reachability_partition
    // ───────────────────────────────────────────────────────────────────────

    /// Nodes with identical reachable sets share a block.
    #[test]
    fn test_tropical_reachability_partition_groups_identical_reach_rows() {
        // 0->2 and 1->2: both 0 and 1 reach {self, 2}, but self differs, so the rows differ.
        // 2 reaches only itself, giving three blocks.
        let (_, adj) = build_pair(3, &[(0, 2), (1, 2)]);
        let p = reachability_partition(&adj);
        assert_eq!(p.num_blocks(), 3, "all three reachability rows differ");

        // 0 <-> 1 plus both reaching 2: the reachable sets of 0 and 1 both become {0, 1, 2}.
        let (_, adj2) = build_pair(3, &[(0, 1), (1, 0), (0, 2), (1, 2)]);
        let p2 = reachability_partition(&adj2);
        assert_eq!(p2.num_blocks(), 2, "0 and 1 share a block, 2 stands alone");
    }

    /// In a complete graph every reachability row is identical, so the partition is indiscrete and h = 0.
    #[test]
    fn test_tropical_reachability_partition_complete_graph_is_indiscrete() {
        let edges: Vec<(usize, usize)> =
            (0..4).flat_map(|i| (0..4).map(move |j| (i, j))).filter(|(i, j)| i != j).collect();
        let (_, adj) = build_pair(4, &edges);
        let p = reachability_partition(&adj);
        assert_eq!(p.num_blocks(), 1, "a saturated graph is indiscrete");
        assert_eq!(p.logical_entropy(), 0.0, "indiscrete implies h = 0");
    }

    /// With no edges each node reaches only itself, so all rows differ and the partition is discrete.
    #[test]
    fn test_tropical_reachability_partition_no_edges_is_discrete() {
        let (_, adj) = build_pair(4, &[]);
        let p = reachability_partition(&adj);
        assert_eq!(p.num_blocks(), 4, "no edges means discrete");
        assert!(p.logical_entropy() > 0.0);
    }

    /// **Bit-exact seal against the definition.** The induced partition is defined by the
    /// finiteness pattern of `ground_from`; the boolean path here is pinned to agree with it
    /// on the same edge set — the same verdict, without floating point.
    #[test]
    fn test_tropical_reachability_partition_matches_floyd_warshall_finiteness() {
        for edges in [
            vec![],
            vec![(0usize, 1usize)],
            vec![(0, 1), (1, 2)],
            vec![(0, 1), (1, 0), (2, 3)],
            vec![(0, 3), (1, 3), (2, 3)], // star into a sink
            vec![(0, 1), (1, 2), (2, 0), (3, 0)],
        ] {
            let n = 4;
            let (mat, adj) = build_pair(n, &edges);
            let closure = floyd_warshall(&mat);
            // f64 path: nodes with the same finiteness pattern land in the same block.
            let mut ids_f64 = vec![0u32; n];
            let mut seen: Vec<Vec<bool>> = Vec::new();
            for i in 0..n {
                let pat: Vec<bool> = (0..n).map(|j| closure[i][j].is_finite()).collect();
                let id = match seen.iter().position(|s| *s == pat) {
                    Some(k) => k,
                    None => {
                        seen.push(pat);
                        seen.len() - 1
                    }
                };
                ids_f64[i] = id as u32;
            }
            let p_f64 = Partition::from_ids(ids_f64);
            let p_bool = reachability_partition(&adj);
            assert_eq!(
                p_bool.num_blocks(),
                p_f64.num_blocks(),
                "boolean path disagrees with the ground_from finiteness path (edges={:?})",
                edges
            );
            assert_eq!(
                p_bool.logical_entropy(),
                p_f64.logical_entropy(),
                "h mismatch (edges={:?})",
                edges
            );
        }
    }
}
