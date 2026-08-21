//! Human-facing `Display` ladder for [`Partition`].
//!
//! The codebase's first `impl fmt::Display for Partition`. Replaces the
//! derive-`Debug` `FastPartition` BitVec dump with a scale-aware, label-blind,
//! *delegation-only* rendering. Pure DISPLAY — the engine (learning loop,
//! `tlw::rank`, the 6 backends) is untouched; every rung reads existing
//! accessors (`to_blocks`/`get_ids`/`dit_count`/`num_blocks`/
//! `block_sizes`) and re-implements no metric logic (SSOT).
//!
//! Ladder, governed by `N = size()`:
//! - **Rung S** (`N ≤ n_set`): block-set notation `{{0,1},{2,3,4}}` — faithful.
//! - **Rung V** (`n_set < N ≤ n_vec`): canonical id-vector `[0,0,1,1,1]` — faithful.
//! - **Rung C** (`N > n_vec`): lossy `⟨…⟩` count-summary (block-size multiset).
//!
//! Correction sealed here: the count-summary `h` is
//! `logical_entropy() = 1 − Σ|B|²/N²` (rendered as the exact rational
//! `dit_count()/N²`), **not** the edge-density `|dit|/C(N,2)`.
//!
//! Pair convention: the human surface reports `|dit|` as
//! **unordered** `C(N,2)`; the engine keeps ordered-N². This is a boundary.

use crate::Partition;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

/// Thresholds selecting the display rung. Cosmetic/explicit — not learned.
#[derive(Debug, Clone, Copy)]
pub struct DisplayConfig {
    /// Max `N` for block-set notation (Rung S).
    pub n_set: usize,
    /// Max `N` for id-vector notation (Rung V); above this → count-summary.
    pub n_vec: usize,
    /// Max distinct size-groups shown in the count-summary before eliding.
    pub size_cap: usize,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        DisplayConfig {
            n_set: 24,
            n_vec: 256,
            size_cap: 12,
        }
    }
}

/// Reduce `num/den` to lowest terms (den assumed > 0).
fn reduce(num: usize, den: usize) -> (usize, usize) {
    let mut a = num;
    let mut b = den;
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    let g = a.max(1);
    (num / g, den / g)
}

/// `sep`-join `parts` without relying on the `alloc::slice::Join` trait
/// (which is not in scope under `#![no_std]`).
fn join(parts: &[String], sep: &str) -> String {
    let mut out = String::new();
    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            out.push_str(sep);
        }
        out.push_str(part);
    }
    out
}

/// C(n, 2) — unordered pair count.
fn comb2(n: usize) -> usize {
    n * n.saturating_sub(1) / 2
}

/// Right-align `s` into width `w` (unicode-aware).
fn pad_left(s: &str, w: usize) -> String {
    let mut out = String::new();
    for _ in s.chars().count()..w {
        out.push(' ');
    }
    out.push_str(s);
    out
}

/// Bundle-grid data (single π): (block sizes in canonical `to_blocks` order, K×K grid).
/// `grid[i][i] = C(sᵢ,2)` (indit), `grid[i][j≠i] = sᵢ·sⱼ` (dit bundle Bᵢ⊗Bⱼ).
fn bundle_grid_data(p: &Partition) -> (Vec<usize>, Vec<Vec<usize>>) {
    let sizes: Vec<usize> = p.to_blocks().iter().map(|b| b.len()).collect();
    let k = sizes.len();
    let mut grid: Vec<Vec<usize>> = Vec::with_capacity(k);
    for i in 0..k {
        let mut row: Vec<usize> = Vec::with_capacity(k);
        for j in 0..k {
            row.push(if i == j {
                comb2(sizes[i])
            } else {
                sizes[i] * sizes[j]
            });
        }
        grid.push(row);
    }
    (sizes, grid)
}

/// Contingency data (pair): (row block sizes, col block sizes, `X[i][j] = |Bᵢ ∩ Cⱼ|`)
/// in canonical `to_blocks` order. Empty triple if universe sizes differ.
fn contingency_data(a: &Partition, b: &Partition) -> (Vec<usize>, Vec<usize>, Vec<Vec<usize>>) {
    let n = a.size();
    if n != b.size() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let a_blocks = a.to_blocks();
    let b_blocks = b.to_blocks();
    let (ra, cb) = (a_blocks.len(), b_blocks.len());
    let mut row_of: Vec<usize> = Vec::with_capacity(n);
    row_of.resize(n, 0);
    for (bi, blk) in a_blocks.iter().enumerate() {
        for &e in blk {
            row_of[e] = bi;
        }
    }
    let mut col_of: Vec<usize> = Vec::with_capacity(n);
    col_of.resize(n, 0);
    for (bi, blk) in b_blocks.iter().enumerate() {
        for &e in blk {
            col_of[e] = bi;
        }
    }
    let mut x: Vec<Vec<usize>> = Vec::with_capacity(ra);
    for _ in 0..ra {
        let mut row: Vec<usize> = Vec::with_capacity(cb);
        row.resize(cb, 0);
        x.push(row);
    }
    for e in 0..n {
        x[row_of[e]][col_of[e]] += 1;
    }
    let row_sizes: Vec<usize> = a_blocks.iter().map(|b| b.len()).collect();
    let col_sizes: Vec<usize> = b_blocks.iter().map(|b| b.len()).collect();
    (row_sizes, col_sizes, x)
}

/// `|dit(a) ∩ dit(b)|` (ordered) from the contingency grid:
/// `N² − Σ|Bᵢ|² − Σ|Cⱼ|² + Σ|Bᵢ∩Cⱼ|²` (all-integer, label-blind).
fn dit_intersection_ordered(a: &Partition, b: &Partition) -> usize {
    let n = a.size();
    let (rs, cs, x) = contingency_data(a, b);
    let sr: usize = rs.iter().map(|s| s * s).sum();
    let sc: usize = cs.iter().map(|s| s * s).sum();
    let sx: usize = x.iter().flatten().map(|&v| v * v).sum();
    n * n + sx - sr - sc
}

impl Partition {
    /// Scale-aware human rendering. See module docs.
    pub fn render(&self, cfg: &DisplayConfig) -> String {
        let n = self.size();
        if n == 0 {
            return "{}".to_string();
        }
        if n <= cfg.n_set {
            self.block_set_notation()
        } else if n <= cfg.n_vec {
            self.id_vector_notation()
        } else {
            self.count_summary()
        }
    }

    /// Rung S — block-set notation `{{0,1},{2,3,4}}` (faithful, label-blind).
    ///
    /// Delegates to [`Partition::to_blocks`] (min-element sorted), so the output
    /// depends only on block *structure*, never on stored block-id values.
    pub fn block_set_notation(&self) -> String {
        let blocks = self.to_blocks();
        if blocks.is_empty() {
            return "{}".to_string();
        }
        let inner: Vec<String> = blocks
            .iter()
            .map(|b| {
                let elems: Vec<String> = b.iter().map(|e| e.to_string()).collect();
                format!("{{{}}}", join(&elems, ","))
            })
            .collect();
        format!("{{{}}}", join(&inner, ","))
    }

    /// Rung V — canonical (first-appearance normalized) id-vector `[0,0,1,1,1]`.
    ///
    /// Canonicalized from [`Partition::to_blocks`] so it is label-invariant
    /// regardless of the backend's stored ids.
    pub fn id_vector_notation(&self) -> String {
        let n = self.size();
        if n == 0 {
            return "[]".to_string();
        }
        let mut ids: Vec<u32> = Vec::with_capacity(n);
        ids.resize(n, 0);
        for (bi, blk) in self.to_blocks().iter().enumerate() {
            for &e in blk {
                ids[e] = bi as u32;
            }
        }
        let body: Vec<String> = ids.iter().map(|x| x.to_string()).collect();
        format!("[{}]", join(&body, ","))
    }

    /// One-line compact form `0.1|2.3.4` (blocks by `|`, elements by `.`).
    pub fn compact(&self) -> String {
        let parts: Vec<String> = self
            .to_blocks()
            .iter()
            .map(|b| {
                let elems: Vec<String> = b.iter().map(|e| e.to_string()).collect();
                join(&elems, ".")
            })
            .collect();
        join(&parts, "|")
    }

    /// Rung C — lossy count-summary `⟨K=… | sizes=[…] | h=p/q | |dit|=… (of …) unordered⟩`.
    ///
    /// **RC2 seal:** `h` is `logical_entropy() = dit_count()/N²` (the
    /// exact rational), *not* the edge-density `|dit|_unordered/C(N,2)`.
    /// `|dit|` is reported unordered (RC1 display convention). Lossy: keeps only
    /// the block-size multiset (`⟨…⟩` marks it non-faithful).
    pub fn count_summary(&self) -> String {
        let n = self.size();
        let k = self.num_blocks();
        let dit_ord = self.dit_count(); // N² − Σ|B|² (ordered, = h numerator)
        let dit_unord = dit_ord / 2;
        let c2 = n * n.saturating_sub(1) / 2; // C(N,2)
        let (hn, hd) = if n == 0 { (0, 1) } else { reduce(dit_ord, n * n) };
        let h_str = if hd == 1 {
            hn.to_string()
        } else {
            format!("{}/{}", hn, hd)
        };
        // block-size multiset, sorted desc, run-length, capped.
        let mut sizes: Vec<usize> = self.block_sizes().into_iter().map(|(_, sz)| sz).collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        let mut groups: Vec<(usize, usize)> = Vec::new(); // (size, multiplicity)
        for s in sizes {
            match groups.last_mut() {
                Some(last) if last.0 == s => last.1 += 1,
                _ => groups.push((s, 1)),
            }
        }
        let cap = DisplayConfig::default().size_cap;
        let shown: Vec<String> = groups
            .iter()
            .take(cap)
            .map(|(s, m)| {
                if *m > 1 {
                    format!("{}×{}", s, m)
                } else {
                    s.to_string()
                }
            })
            .collect();
        let mut sizes_str = join(&shown, ",");
        if groups.len() > cap {
            sizes_str.push_str(&format!(",…+{} more", groups.len() - cap));
        }
        format!(
            "⟨K={} | sizes=[{}] | h={} | |dit|={} (of {}) unordered⟩",
            k, sizes_str, h_str, dit_unord, c2
        )
    }

    /// **Bundle-grid card** (single π) — the complete-multipartite structure view
    /// `dit(π) = ⋃_{i<j} Bᵢ⊗Bⱼ` rendered as a K×K grid: bracketed
    /// diagonal `[C(sᵢ,2)]` = indit (in-block non-dits), off-diagonal `sᵢ·sⱼ` = the dit
    /// bundle. Footer: `Σupper = |dit|` (unordered), invariant `Σupper + Σdiag = C(N,2)`,
    /// `h`. Pure DISPLAY (block-size only; delegates to `to_blocks`/`dit_count`).
    pub fn bundle_grid(&self) -> String {
        let (sizes, grid) = bundle_grid_data(self);
        let k = sizes.len();
        if k == 0 {
            return "{}".to_string();
        }
        let n = self.size();
        let labels: Vec<String> = (0..k).map(|i| format!("B{}({})", i, sizes[i])).collect();
        let cell = |i: usize, j: usize| -> String {
            if i == j {
                format!("[{}]", grid[i][j])
            } else {
                grid[i][j].to_string()
            }
        };
        let mut w = 0usize;
        for l in &labels {
            w = w.max(l.chars().count());
        }
        for i in 0..k {
            for j in 0..k {
                w = w.max(cell(i, j).chars().count());
            }
        }
        w += 2;
        let mut lines: Vec<String> = Vec::new();
        let mut hdr = pad_left("", w);
        for l in &labels {
            hdr.push_str(&pad_left(l, w));
        }
        lines.push(hdr);
        for i in 0..k {
            let mut row = pad_left(&labels[i], w);
            for j in 0..k {
                row.push_str(&pad_left(&cell(i, j), w));
            }
            lines.push(row);
        }
        let dit_unord = self.dit_count() / 2;
        let sigma_diag: usize = sizes.iter().map(|&s| comb2(s)).sum();
        let (hn, hd) = if n == 0 {
            (0, 1)
        } else {
            reduce(self.dit_count(), n * n)
        };
        let h_str = if hd == 1 {
            hn.to_string()
        } else {
            format!("{}/{}", hn, hd)
        };
        lines.push(format!(
            "Σupper = {} = |dit| (unordered)   Σdiag = {}   {} + {} = {} = C({},2)   h = {}",
            dit_unord,
            sigma_diag,
            dit_unord,
            sigma_diag,
            dit_unord + sigma_diag,
            n,
            h_str
        ));
        join(&lines, "\n")
    }

    /// **Refinement view** (pair) — renders the Meet=∪ (exact) / Join⊆∩ (gap) asymmetry
    /// `a∧b` (`differentiate`) with `dit = dit(a) ⊔ dit(b)` exact;
    /// `a∨b` (`integrate`, **EXACT — measurement-only**, learning loops use ShallowJoin)
    /// with `dit ⊆ dit(a)∩dit(b)` and the transitive-closure gap. Pure DISPLAY.
    pub fn refine_view(&self, other: &Self) -> String {
        let n = self.size();
        if n != other.size() {
            return format!(
                "refine_view: universe size mismatch ({} vs {})",
                n,
                other.size()
            );
        }
        let i_ord = dit_intersection_ordered(self, other);
        let meet_p = self.differentiate(other);
        let join_p = self.integrate(other); // EXACT (measurement-only)
        let dit_meet = meet_p.dit_count();
        let dit_join = join_p.dit_count();
        let mut lines: Vec<String> = Vec::new();
        lines.push(format!("a     = {}", self.block_set_notation()));
        lines.push(format!("b     = {}", other.block_set_notation()));
        lines.push(format!("a ∧ b = {}", meet_p.block_set_notation()));
        lines.push(format!(
            "        dit(a∧b) = dit(a) ⊔ dit(b)   [EXACT ∪, |dit|={} unordered]",
            dit_meet / 2
        ));
        lines.push(format!("a ∨ b = {}", join_p.block_set_notation()));
        lines.push(format!(
            "        dit(a∨b) ⊆ dit(a)∩dit(b) = {}   [|dit(a∨b)|={}, gap={} unordered]",
            i_ord / 2,
            dit_join / 2,
            (i_ord - dit_join) / 2
        ));
        join(&lines, "\n")
    }

    /// **Contingency card** (pair) — the `|Bᵢ∩Cⱼ|` grid with marginals (
    /// Phase 2); the pairwise structure substrate for d_⊕/D_L. Meet = nonempty cells,
    /// Join = components; the Meet/Join/gap summary is appended. Pure DISPLAY (transient
    /// boundary, not persisted). NOTE: this is the A-region `{|B∩C|}` substrate — it does
    /// NOT use a log-based ARI/NMI: those presuppose a background distribution.
    pub fn contingency_card(&self, other: &Self) -> String {
        let n = self.size();
        if n != other.size() {
            return format!(
                "contingency_card: universe size mismatch ({} vs {})",
                n,
                other.size()
            );
        }
        let (row_sizes, col_sizes, x) = contingency_data(self, other);
        let (ra, cb) = (row_sizes.len(), col_sizes.len());
        let col_labels: Vec<String> = (0..cb).map(|j| format!("C{}", j)).collect();
        let mut w = 3usize;
        for l in &col_labels {
            w = w.max(l.chars().count());
        }
        for s in row_sizes.iter().chain(col_sizes.iter()) {
            w = w.max(s.to_string().chars().count());
        }
        for row in &x {
            for v in row {
                w = w.max(v.to_string().chars().count());
            }
        }
        w += 2;
        let rowhdr_w = 5usize;
        let mut lines: Vec<String> = Vec::new();
        let mut hdr = pad_left("", rowhdr_w);
        for l in &col_labels {
            hdr.push_str(&pad_left(l, w));
        }
        hdr.push_str(&pad_left("|Bi|", w));
        lines.push(hdr);
        for i in 0..ra {
            let mut row = pad_left(&format!("B{}", i), rowhdr_w);
            for j in 0..cb {
                row.push_str(&pad_left(&x[i][j].to_string(), w));
            }
            row.push_str(&pad_left(&row_sizes[i].to_string(), w));
            lines.push(row);
        }
        let mut foot = pad_left("|Cj|", rowhdr_w);
        for j in 0..cb {
            foot.push_str(&pad_left(&col_sizes[j].to_string(), w));
        }
        foot.push_str(&pad_left(&n.to_string(), w));
        lines.push(foot);
        // Meet/Join/gap summary (delegates to unified.rs Meet/Join).
        let i_ord = dit_intersection_ordered(self, other);
        let dit_join = self.integrate(other).dit_count();
        let dit_meet = self.differentiate(other).dit_count();
        lines.push(format!(
            "Meet (∧) = nonempty cells, |dit|={} (∪ exact)   Join (∨) = components, |dit|={}, gap={} (⊆∩={})",
            dit_meet / 2,
            dit_join / 2,
            (i_ord - dit_join) / 2,
            i_ord / 2
        ));
        join(&lines, "\n")
    }
}

impl fmt::Display for Partition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render(&DisplayConfig::default()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Partition;

    fn p(ids: &[u32]) -> Partition {
        Partition::from_ids(ids.to_vec())
    }

    // ---- REF partitions ----
    // REF-A: {{0,1},{2,3,4}}          REF-B: {{0,3,6,9},{1,4,7,10},{2,5,8,11}}
    // REF-C: π1={{0,1,2},{3,4,5}}, π2={{0,3},{1,4},{2,5}}

    #[test]
    fn test_display_ref_a() {
        assert_eq!(p(&[0, 0, 1, 1, 1]).to_string(), "{{0,1},{2,3,4}}");
    }

    #[test]
    fn test_display_ref_b() {
        assert_eq!(
            p(&[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]).to_string(),
            "{{0,3,6,9},{1,4,7,10},{2,5,8,11}}"
        );
    }

    #[test]
    fn test_display_ref_c() {
        assert_eq!(p(&[0, 0, 0, 1, 1, 1]).to_string(), "{{0,1,2},{3,4,5}}");
        assert_eq!(p(&[0, 1, 2, 0, 1, 2]).to_string(), "{{0,3},{1,4},{2,5}}");
    }

    #[test]
    fn test_label_invariance() {
        // Permuting block-id VALUES must not change the render.
        assert_eq!(p(&[1, 1, 0, 0, 0]).to_string(), p(&[0, 0, 1, 1, 1]).to_string());
        assert_eq!(p(&[2, 2, 7, 7, 7]).to_string(), "{{0,1},{2,3,4}}");
    }

    #[test]
    fn test_count_summary_ref_a() {
        assert_eq!(
            p(&[0, 0, 1, 1, 1]).count_summary(),
            "⟨K=2 | sizes=[3,2] | h=12/25 | |dit|=6 (of 10) unordered⟩"
        );
    }

    #[test]
    fn test_count_summary_runlength() {
        // REF-B: sizes run-length [4×3]; h = 2/3 (logical entropy),
        // NOT the edge-density 8/11 (=48/66) — RC2 seal.
        assert_eq!(
            p(&[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]).count_summary(),
            "⟨K=3 | sizes=[4×3] | h=2/3 | |dit|=48 (of 66) unordered⟩"
        );
    }

    #[test]
    fn test_display_h_delegates() {
        // RC2: displayed h equals logical_entropy(), and is NOT edge-density.
        let a = p(&[0, 0, 1, 1, 1]);
        // parse "h=12/25" out of the summary
        let s = a.count_summary();
        let hpart = s.split("h=").nth(1).unwrap().split(' ').next().unwrap();
        let mut it = hpart.split('/');
        let num: f64 = it.next().unwrap().parse().unwrap();
        let den: f64 = it.next().unwrap().parse().unwrap();
        let h_disp = num / den;
        assert!((h_disp - a.logical_entropy()).abs() < 1e-12, "h must delegate to logical_entropy");
        let edge_density = 6.0 / 10.0; // |dit|_unord / C(5,2)
        assert!((h_disp - edge_density).abs() > 1e-6, "h must NOT be edge-density (RC2)");
    }

    #[test]
    fn test_compact() {
        assert_eq!(p(&[0, 0, 1, 1, 1]).compact(), "0.1|2.3.4");
        assert_eq!(
            p(&[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]).compact(),
            "0.3.6.9|1.4.7.10|2.5.8.11"
        );
    }

    #[test]
    fn test_rung_selection() {
        let cfg = DisplayConfig::default();
        // N=5 ≤ 24 → block-set
        assert_eq!(p(&[0, 0, 1, 1, 1]).render(&cfg), "{{0,1},{2,3,4}}");
        // 24 < N=40 ≤ 256 → id-vector (two blocks by parity)
        let ids: Vec<u32> = (0..40).map(|i| (i % 2) as u32).collect();
        let mid = Partition::from_ids(ids.clone());
        let expect_v = {
            let body: Vec<String> = ids.iter().map(|x| x.to_string()).collect();
            format!("[{}]", body.join(","))
        };
        assert_eq!(mid.render(&cfg), expect_v);
        // N=300 > 256 → count-summary (starts with ⟨)
        let big = Partition::from_ids((0..300).map(|i| (i % 3) as u32).collect());
        assert!(big.render(&cfg).starts_with("⟨K=3"));
    }

    #[test]
    fn test_empty() {
        assert_eq!(Partition::from_ids(Vec::new()).to_string(), "{}");
    }

    // ── Phase 2: structure view (bundle grid / contingency / refine) ──

    #[test]
    fn test_bundle_grid_data_invariant() {
        // MP-1: Σupper == |dit| (unordered) == dit_count()/2, and the
        // complete-multipartite identity Σupper + Σdiag == C(N,2).
        let refs: [Vec<u32>; 4] = [
            vec![0, 0, 1, 1, 1],                            // REF-A
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],       // REF-B
            vec![0, 0, 0, 1, 1, 1],                          // REF-C π1
            vec![0, 1, 2, 3, 4],                             // discrete
        ];
        for ids in refs {
            let p = Partition::from_ids(ids);
            let n = p.size();
            let (sizes, grid) = bundle_grid_data(&p);
            let k = sizes.len();
            let mut sigma_upper = 0usize;
            let mut sigma_diag = 0usize;
            for i in 0..k {
                for j in 0..k {
                    if i < j {
                        sigma_upper += grid[i][j];
                    } else if i == j {
                        sigma_diag += grid[i][j];
                    }
                }
            }
            assert_eq!(sigma_upper, p.dit_count() / 2, "Σupper == |dit| unordered");
            assert_eq!(sigma_upper + sigma_diag, comb2(n), "Σupper + Σdiag == C(N,2)");
        }
    }

    #[test]
    fn test_contingency_mp1_bit_exact() {
        // MP-1: grid-derived moments are bit-exact with the metric SSOT (unified.rs).
        let pairs: [(Vec<u32>, Vec<u32>); 3] = [
            (vec![0, 0, 0, 1, 1, 1], vec![0, 1, 2, 0, 1, 2]), // REF-C
            (vec![0, 0, 1, 1], vec![0, 1, 0, 1]),             // the RC3 colliding pair
            (vec![0, 0, 1, 1, 2], vec![0, 1, 0, 1, 2]),
        ];
        for (a_ids, b_ids) in pairs {
            let a = Partition::from_ids(a_ids);
            let b = Partition::from_ids(b_ids);
            let n = a.size();
            let i_ord = dit_intersection_ordered(&a, &b);
            let dit_a = a.dit_count();
            let dit_b = b.dit_count();
            // (1) grid d_⊕ == ditset_xor_distance (SSOT)
            let d_xor_ord = dit_a + dit_b - 2 * i_ord;
            let expect = a.ditset_xor_distance(&b);
            assert!(
                ((d_xor_ord as f64) / ((n * n) as f64) - expect).abs() < 1e-12,
                "grid d_⊕ must equal ditset_xor_distance"
            );
            // (2) |dit(a∧b)| == dit_a + dit_b − I  (Meet = ∪, exact)
            assert_eq!(
                a.differentiate(&b).dit_count(),
                dit_a + dit_b - i_ord,
                "|dit(meet)| == dit_a + dit_b − |dit∩|"
            );
            // (3) Join ⊆ ∩ ⟹ gap ≥ 0
            let dit_join = a.integrate(&b).dit_count();
            assert!(i_ord >= dit_join, "|dit(a∨b)| ≤ |dit(a)∩dit(b)|");
        }
    }

    #[test]
    fn test_refine_view_renders() {
        // REF-C: Meet = 6 singletons (∪ exact), Join = 1 block, gap = 6 (unordered).
        let a = Partition::from_ids(vec![0, 0, 0, 1, 1, 1]);
        let b = Partition::from_ids(vec![0, 1, 2, 0, 1, 2]);
        let s = a.refine_view(&b);
        assert!(s.contains("a ∧ b = {{0},{1},{2},{3},{4},{5}}"), "Meet block-set\n{}", s);
        assert!(s.contains("EXACT ∪"), "Meet exactness label\n{}", s);
        assert!(s.contains("gap=6 unordered"), "Join gap = 6\n{}", s);
    }

    #[test]
    fn test_bundle_grid_renders() {
        let p = Partition::from_ids(vec![0, 0, 1, 1, 1]); // REF-A
        let s = p.bundle_grid();
        assert!(s.contains("B0(2)"), "block label\n{}", s);
        assert!(s.contains("[1]") && s.contains("[3]"), "bracketed indit diagonal\n{}", s);
        assert!(s.contains("10 = C(5,2)"), "invariant footer\n{}", s);
        assert!(s.contains("h = 12/25"), "h delegates (RC2)\n{}", s);
    }

    #[test]
    fn test_contingency_card_renders() {
        let a = Partition::from_ids(vec![0, 0, 0, 1, 1, 1]);
        let b = Partition::from_ids(vec![0, 1, 2, 0, 1, 2]);
        let s = a.contingency_card(&b);
        assert!(s.contains("C0"), "col label\n{}", s);
        assert!(s.contains("|Bi|"), "row marginal header\n{}", s);
        assert!(s.contains("gap=6"), "Meet/Join summary\n{}", s);
    }

    #[test]
    fn test_pair_view_size_mismatch_guard() {
        let a = Partition::from_ids(vec![0, 0, 1, 1]);
        let c = Partition::from_ids(vec![0, 1, 2]);
        assert!(a.refine_view(&c).contains("mismatch"));
        assert!(a.contingency_card(&c).contains("mismatch"));
    }
}
