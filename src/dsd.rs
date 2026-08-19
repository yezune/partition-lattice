//! Direct-sum decompositions (DSD) — the **linear-algebra image** of partition logic,
//! and the fact that it is *not* a lattice.
//!
//! Ellerman's yoga: if a subset corresponds to the subspace it spans, then a **partition
//! corresponds to a direct-sum decomposition**.
//! > "Thus we may say that the vector space version of a set partition is a DSD."
//! > — Ellerman, *Partition Logic Duality* (arXiv:2409.18134)
//!
//! # ⚠ Order convention — the easiest thing to get wrong here
//!
//! This crate uses the **order-dual** of Ellerman's convention (see the crate README).
//! Ellerman puts the indiscrete partition (the "blob") at the bottom as `0`; this crate puts
//! it at the top as `Top`. **The lattice-theoretic name of the same operation therefore
//! flips:**
//!
//! | The operation | Ellerman | This crate | Total? |
//! |---|---|---|:-:|
//! | Common refinement — block intersections `{Vᵢ ∩ Wⱼ}` | `join` (lub) | **`Meet`** = `differentiate` (glb) | **partial** |
//! | Common coarsening | `meet` (glb) | **`Join`** = `integrate` (lub) | total |
//! | Indiscrete (one block) | blob `0`, **least** | `Top`, **greatest** | — |
//!
//! ⟹ The source's *"DSD(V) is a meet-semilattice"* reads, in this crate's vocabulary, as
//! **"Join-semilattice"**. Translating the sentence literally makes it exactly false.
//!
//! **Why the partiality matters.** The operation that stops being total under linearisation
//! is the common refinement — the one that *adds* distinctions. Whatever you build on top of
//! refinement inherits that partiality once you move to subspaces.
//!
//! This module does not argue the point; it **measures** it, by exhaustive enumeration.
//!
//! # Source
//!
//! Ellerman, *The Quantum Logic of Direct-Sum Decompositions*, arXiv:1604.01087v2;
//! *Logic J. IGPL* 26(1):1–13.
//! - Compatibility (p. 3): π ↔ σ iff the proto-join `{Vᵢ ∩ Wⱼ}` spans all of `V`.
//!   When two DSDs are the eigenspace decompositions of two operators, that span is the
//!   **kernel of the commutator** ⟹ commuting implies compatible. **Compatibility = commutativity.**
//! - "not a join-semilattice" (p. 4) / "meet-semilattice", "partial partition algebra" (p. 5)
//! - Interval isomorphism (p. 7): for any DSD π, the interval `[0, π]` is order-isomorphic to
//!   set-partition logic on its block set.
//!
//! # Representation
//!
//! Vectors of `V = GF(2)ⁿ` are bitmasks over `0..2ⁿ`, and a **subspace is the bitmask of the
//! vectors it contains** (bit `v` set ⟺ vector `v` is a member). Intersection is then a single
//! `&`, which is what makes the core operation cheap. `n ≤ 6` (2⁶ = 64 vectors) fits in a `u64`.
//!
//! # Scope
//!
//! Exhaustive enumeration over *pairs* tops out at `n ≤ 4` — this is the **available maximum,
//! not a sufficiency claim**. `D₂(5) = 540,145`, so the pair count reaches `1.46×10¹¹` and is
//! out of reach in principle. Enumeration *counts* alone can still be checked up to `n = 5`.

use std::collections::HashSet;

/// A subspace of `GF(2)ⁿ` — a **bitmask over its member vectors**.
///
/// Bit `v` set means vector `v ∈ 0..2ⁿ` belongs. The zero vector is always a member, so a
/// valid subspace always has bit 0 set. Defined for `n ≤ 6`.
pub type Subspace = u64;

/// The zero subspace `{0}`.
pub const ZERO_SUBSPACE: Subspace = 1;

/// Dimension of the subspace: `|S| = 2^d`, so `d = log₂(popcount)`.
#[inline]
pub fn dim(s: Subspace) -> usize {
    debug_assert!(s & 1 == 1, "a subspace must contain the zero vector");
    s.count_ones().trailing_zeros() as usize
}

/// The sum `A + B = {a ⊕ b}` of two subspaces.
pub fn sum(a: Subspace, b: Subspace) -> Subspace {
    let mut out: Subspace = 0;
    let mut av = a;
    while av != 0 {
        let i = av.trailing_zeros();
        av &= av - 1;
        let mut bv = b;
        while bv != 0 {
            let j = bv.trailing_zeros();
            bv &= bv - 1;
            out |= 1u64 << (i ^ j);
        }
    }
    out
}

/// The subspace spanned by a set of vectors.
pub fn span(vectors: impl IntoIterator<Item = u32>) -> Subspace {
    let mut s = ZERO_SUBSPACE;
    for v in vectors {
        if s & (1u64 << v) == 0 {
            s = sum(s, ZERO_SUBSPACE | (1u64 << v));
        }
    }
    s
}

/// All of `GF(2)ⁿ` (every one of the `2ⁿ` vectors).
#[inline]
pub fn full_space(n: usize) -> Subspace {
    assert!(n <= 6, "this representation is valid only up to n = 6 (2^6 = 64 vectors)");
    if n == 6 {
        u64::MAX
    } else {
        (1u64 << (1usize << n)) - 1
    }
}

/// Every **non-zero** subspace of `GF(2)ⁿ`, in ascending order (deterministic).
pub fn all_nonzero_subspaces(n: usize) -> Vec<Subspace> {
    let full = full_space(n);
    let mut seen: HashSet<Subspace> = HashSet::new();
    seen.insert(ZERO_SUBSPACE);
    let mut frontier = vec![ZERO_SUBSPACE];
    while let Some(s) = frontier.pop() {
        if s == full {
            continue;
        }
        for v in 1..(1u32 << n) {
            if s & (1u64 << v) != 0 {
                continue;
            }
            let t = sum(s, ZERO_SUBSPACE | (1u64 << v));
            if seen.insert(t) {
                frontier.push(t);
            }
        }
    }
    seen.remove(&ZERO_SUBSPACE);
    let mut out: Vec<Subspace> = seen.into_iter().collect();
    out.sort_unstable();
    out
}

/// A direct-sum decomposition of `GF(2)ⁿ`: a family of non-zero subspaces such that every
/// vector is **uniquely** a sum of one vector from each.
///
/// Blocks are kept sorted, which makes both enumeration and equality deterministic.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Dsd {
    blocks: Vec<Subspace>,
}

impl Dsd {
    /// Build from sorted blocks. Validity is *not* checked — see [`Dsd::is_valid_direct_sum`].
    pub fn from_sorted_blocks(blocks: Vec<Subspace>) -> Self {
        debug_assert!(blocks.windows(2).all(|w| w[0] < w[1]), "blocks must be in ascending order");
        Self { blocks }
    }

    pub fn blocks(&self) -> &[Subspace] {
        &self.blocks
    }

    /// Number of blocks — the analogue of a partition's block count; it fixes the interval size `Bell(len)`.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Is this really a direct-sum decomposition of `GF(2)ⁿ`? An independent re-check.
    ///
    /// If the dimensions sum to `n` **and** the union spans `V`, the sum map is a surjection between
    /// equally sized sets, hence a bijection. This deliberately takes a *different* route from the enumerator.
    pub fn is_valid_direct_sum(&self, n: usize) -> bool {
        if self.blocks.iter().any(|&b| b == ZERO_SUBSPACE || b & 1 == 0) {
            return false;
        }
        if self.blocks.iter().map(|&b| dim(b)).sum::<usize>() != n {
            return false;
        }
        let mut acc = ZERO_SUBSPACE;
        for &b in &self.blocks {
            acc = sum(acc, b);
        }
        acc == full_space(n)
    }

    /// **Refinement**: is every block of `self` contained in some block of `other`?
    ///
    /// In this crate's order that is `self ⪯ other` — `self` is the **finer** one.
    /// (In Ellerman's order it reads `other ⪯ self`.)
    pub fn refines(&self, other: &Dsd) -> bool {
        self.blocks
            .iter()
            .all(|&b| other.blocks.iter().any(|&c| b & !c == 0))
    }

    /// Enumerate every DSD of `GF(2)ⁿ`, each **exactly once**, in sorted order.
    ///
    /// Subspaces are visited in ascending order with strictly increasing indices, and a block is
    /// added only when it meets the running span **trivially** — the incremental direct-sum invariant.
    pub fn enumerate(n: usize) -> Vec<Dsd> {
        let subs = all_nonzero_subspaces(n);
        let mut out = Vec::new();
        let mut chosen: Vec<Subspace> = Vec::new();
        Self::rec(n, &subs, 0, ZERO_SUBSPACE, 0, &mut chosen, &mut out);
        out.sort();
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn rec(
        n: usize,
        subs: &[Subspace],
        start: usize,
        acc_span: Subspace,
        acc_dim: usize,
        chosen: &mut Vec<Subspace>,
        out: &mut Vec<Dsd>,
    ) {
        if acc_dim == n {
            out.push(Dsd { blocks: chosen.clone() });
            return;
        }
        for (offset, &s) in subs[start..].iter().enumerate() {
            let d = dim(s);
            if acc_dim + d > n {
                continue;
            }
            // Direct-sum invariant: a new block may meet the accumulated span only at the zero vector.
            if s & acc_span != ZERO_SUBSPACE {
                continue;
            }
            chosen.push(s);
            Self::rec(n, subs, start + offset + 1, sum(acc_span, s), acc_dim + d, chosen, out);
            chosen.pop();
        }
    }
}

/// **proto-meet** — the candidate for `Meet`: the non-zero pairwise intersections `{Vᵢ ∩ Wⱼ}`.
///
/// This is the same object Ellerman calls the proto-**join** (see the order convention above).
/// It is the actual `Meet` only when it spans `V`; otherwise `Meet` is **undefined**.
pub fn proto_refinement(a: &Dsd, b: &Dsd) -> Vec<Subspace> {
    let mut out: Vec<Subspace> = Vec::new();
    for &x in a.blocks() {
        for &y in b.blocks() {
            let m = x & y;
            if m != ZERO_SUBSPACE {
                out.push(m);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// **Compatibility** — does the proto-meet span the whole space? (Ellerman, p. 3.)
///
/// For eigenspace DSDs this condition is exactly **commutativity of the two operators**.
pub fn is_compatible(a: &Dsd, b: &Dsd, n: usize) -> bool {
    let mut acc = ZERO_SUBSPACE;
    for m in proto_refinement(a, b) {
        acc = sum(acc, m);
    }
    acc == full_space(n)
}

/// Embed a set partition of `Π(Uₙ)` into `DSD(GF(2)ⁿ)` via `B ↦ span{eᵤ : u ∈ B}`.
///
/// `block_of[u]` is the block index of element `u`.
pub fn embed_set_partition(block_of: &[usize], n: usize) -> Dsd {
    let nblocks = block_of.iter().copied().max().map_or(0, |m| m + 1);
    let mut blocks: Vec<Subspace> = (0..nblocks)
        .map(|b| {
            span(
                (0..n)
                    .filter(|&u| block_of[u] == b)
                    .map(|u| 1u32 << u),
            )
        })
        .filter(|&s| s != ZERO_SUBSPACE)
        .collect();
    blocks.sort_unstable();
    blocks.dedup();
    Dsd::from_sorted_blocks(blocks)
}

/// The standard-basis DSD `ω_std = {⟨e₁⟩, …, ⟨eₙ⟩}` — the minimum of the embedded image.
pub fn standard_basis_dsd(n: usize) -> Dsd {
    let mut blocks: Vec<Subspace> =
        (0..n).map(|u| ZERO_SUBSPACE | (1u64 << (1u32 << u))).collect();
    blocks.sort_unstable();
    Dsd::from_sorted_blocks(blocks)
}

/// All partitions of an `n`-element set, as `block_of` vectors (restricted growth strings).
fn all_set_partitions(n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if n == 0 {
        out.push(Vec::new());
        return out;
    }
    let mut cur = vec![0usize; n];
    fn rec(i: usize, n: usize, maxb: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if i == n {
            out.push(cur.clone());
            return;
        }
        for b in 0..=maxb {
            cur[i] = b;
            rec(i + 1, n, if b == maxb { maxb + 1 } else { maxb }, cur, out);
        }
    }
    rec(1, n, 1, &mut cur, &mut out);
    out
}

/// The image of all of `Π(Uₙ)` under the embedding.
pub fn embed_all_set_partitions(n: usize) -> Vec<Dsd> {
    all_set_partitions(n)
        .iter()
        .map(|bo| embed_set_partition(bo, n))
        .collect()
}

/// **Degenerate control** — measures the **bit-identical predicate** over `Π(Uₙ)`.
///
/// ⚠ This does *not* re-implement compatibility: it embeds the set partitions and calls the same
/// [`is_compatible`]. A control must use the **same ruler** as the treatment; a separate copy can
/// silently become always-true (the first version of this one literally was `x == x && y == y`).
///
/// The result is zero for a *mathematical* reason: the image is a lattice interval, so `Meet` is
/// total inside it. The phenomenon is therefore **invisible in principle** to set partitions.
///
/// Returns `(incompatible pairs, total distinct unordered pairs)`.
pub fn set_partition_incompatible_pairs(n: usize) -> (usize, usize) {
    let embedded = embed_all_set_partitions(n);
    count_incompatible_distinct_pairs(&embedded, n)
}

/// Count incompatible pairs over distinct unordered `{i,j}`, `i < j` — shared by treatment and control.
///
/// ⚠ **Reflexive pairs `(π,π)` are excluded.** A DSD is always compatible with itself (its
/// proto-meet is the diagonal `{Vᵢ}` = π itself, which spans `V`), so those pairs carry no
/// information and would only inflate the denominator.
pub fn count_incompatible_distinct_pairs(elems: &[Dsd], n: usize) -> (usize, usize) {
    let k = elems.len();
    let mut bad = 0;
    let mut total = 0;
    for i in 0..k {
        for j in (i + 1)..k {
            total += 1;
            if !is_compatible(&elems[i], &elems[j], n) {
                bad += 1;
            }
        }
    }
    (bad, total)
}

/// The poset over an exhaustively enumerated `DSD(GF(2)ⁿ)` — order, intervals, partial operations.
///
/// **lub/glb are found by searching the poset, not by the source's construction formula.**
/// Assuming the formula would *presuppose* the theorem being checked rather than verify it.
pub struct DsdPoset {
    n: usize,
    elems: Vec<Dsd>,
    /// `up[i]` = every coarsening of `i` (including itself) — the interval `[i, Top]`.
    up: Vec<Vec<u32>>,
    /// `down[i]` = every refinement of `i` (including itself).
    down: Vec<Vec<u32>>,
}

impl DsdPoset {
    pub fn build(n: usize) -> Self {
        let elems = Dsd::enumerate(n);
        let k = elems.len();
        let mut up = vec![Vec::new(); k];
        let mut down = vec![Vec::new(); k];
        for i in 0..k {
            for j in 0..k {
                if elems[i].refines(&elems[j]) {
                    up[i].push(j as u32);
                    down[j].push(i as u32);
                }
            }
        }
        Self { n, elems, up, down }
    }

    pub fn len(&self) -> usize {
        self.elems.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elems.is_empty()
    }

    pub fn get(&self, i: usize) -> &Dsd {
        &self.elems[i]
    }

    pub fn index_of(&self, d: &Dsd) -> Option<usize> {
        self.elems.iter().position(|e| e == d)
    }

    /// The interval `[π, Top]` — every coarsening of π. Same set as the source's `[0, π]`.
    pub fn interval_above(&self, i: usize) -> &[u32] {
        &self.up[i]
    }

    /// **`Join`** (common coarsening, lub) — **total**, by Prop. 2 of the source.
    pub fn lub(&self, i: usize, j: usize) -> Option<usize> {
        Self::extreme(&self.up[i], &self.up[j], &self.up)
    }

    /// **`Meet`** (common refinement, glb) — **partial**; `None` when it does not exist.
    pub fn glb(&self, i: usize, j: usize) -> Option<usize> {
        Self::extreme(&self.down[i], &self.down[j], &self.down)
    }

    /// Within a set `S` of common bounds, find the unique element whose own directed set contains all
    /// of `S` (`up` gives the least element = lub, `down` the greatest = glb).
    fn extreme(a: &[u32], b: &[u32], dir: &[Vec<u32>]) -> Option<usize> {
        let mut s: Vec<u32> = Vec::new();
        let (mut x, mut y) = (0usize, 0usize);
        while x < a.len() && y < b.len() {
            match a[x].cmp(&b[y]) {
                std::cmp::Ordering::Less => x += 1,
                std::cmp::Ordering::Greater => y += 1,
                std::cmp::Ordering::Equal => {
                    s.push(a[x]);
                    x += 1;
                    y += 1;
                }
            }
        }
        let mut found = None;
        for &m in &s {
            let dm = &dir[m as usize];
            if s.iter().all(|c| dm.binary_search(c).is_ok()) {
                debug_assert!(found.is_none(), "the extremal element is not unique, so antisymmetry fails");
                found = Some(m as usize);
            }
        }
        found
    }

    /// Are the two DSDs compatible — does the proto-meet span `V`?
    pub fn is_compatible(&self, i: usize, j: usize) -> bool {
        is_compatible(&self.elems[i], &self.elems[j], self.n)
    }

    /// `(incompatible pairs, total distinct unordered pairs)` — how partial `Meet` is, quantified.
    ///
    /// Delegates to the **same function** the degenerate control uses.
    pub fn incompatible_unordered_pairs(&self) -> (usize, usize) {
        count_incompatible_distinct_pairs(&self.elems, self.n)
    }

    /// The minimal elements (finest) — unlike `Π(U)`, there are **several**.
    pub fn minimal_elements(&self) -> Vec<usize> {
        (0..self.len()).filter(|&i| self.down[i].len() == 1).collect()
    }

    /// The maximum (`Top`) — the single blob `{V}`.
    pub fn maximal_elements(&self) -> Vec<usize> {
        (0..self.len()).filter(|&i| self.up[i].len() == 1).collect()
    }
}
