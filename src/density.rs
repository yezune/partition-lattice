//! The **density-matrix view** of a partition — and the fact that `h(ρ) = 1 − tr(ρ²)` is
//! *not* a new measure.
//!
//! # Source
//!
//! Ellerman (arXiv:1604.01087v2 §A) defines a partition's density matrix directly from its
//! indit set:
//!
//! ```text
//!   ρ_jk(π) = 1/|U|   if (u_j, u_k) ∈ indit(π)      (same block)
//!             0       otherwise
//! ```
//!
//! That is, `ρ(π) = M_π / N` where `M_π` is the equivalence indicator matrix. The arithmetic
//! then follows immediately:
//!
//! ```text
//!   tr(ρ(π)²) = |indit(π)| / N² = Σ_B |B|² / N²
//!   h(ρ(π))   = 1 − tr(ρ(π)²)  = 1 − Σ_B |B|²/N²  =  h(π)
//! ```
//!
//! # ★ So this module does *not* provide a new entropy function
//!
//! `h(ρ(π))` is the **same value** as the lattice-side logical entropy. Adding a second
//! function for it would split the definition in two, so what this module contributes is not a
//! *measure* but the **density-matrix view** — construction, purity, Lüders measurement. For
//! entropy, use the existing `Partition` API.
//!
//! # ★ Independence — this module never calls the lattice entropy
//!
//! Since `h(ρ) = h(π)` is arithmetically immediate, computing the trace *via* the lattice
//! entropy would make any "verification" of that identity a tautology. So the `N × N` matrix
//! is actually materialised here and the trace is computed as a sum of products. That
//! independence is itself sealed by a test
//! (`quantum_logical_entropy_tests::f_matrix_path_is_independent_of_the_lattice_entropy`).
//!
//! # Exact arithmetic
//!
//! Integers throughout (numerator plus a common denominator); no floating point. An entry is
//! `num[j*dim+k] / den`, with `den = N` for a partition density matrix and `den = 2N` for an
//! equal mixture.
//!
//! # Purity — where the correspondence is easy to get backwards
//!
//! `tr(ρ²) = 1 ⟺ Σ|B|² = N² ⟺ one block ⟺ **indiscrete**`. So among partition density
//! matrices **only `Top` (indiscrete) is pure**; everything else is mixed, and `Bottom`
//! (discrete) is `I/N`, the maximally mixed one. ★ Pure means maximally *indefinite* — zero
//! distinctions — and measurement (Lüders) refines, creating distinctions. That refinement is
//! [`Partition::differentiate`](crate::Partition::differentiate).

use crate::Partition;

/// The density matrix of a partition (or a mixture) — entries are `num[j*dim+k] / den`, all integers.
#[derive(Clone, Debug)]
pub struct DensityMatrix {
    dim: usize,
    num: Vec<u64>,
    den: u64,
}

/// Exact rational comparison: `a/b == c/d ⟺ a·d == c·b`. No floating point.
#[inline]
fn rat_eq(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0 as u128 * b.1 as u128 == b.0 as u128 * a.1 as u128
}

impl PartialEq for DensityMatrix {
    /// **Rational equality** — equal values compare equal even with different denominators.
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
            && self
                .num
                .iter()
                .zip(&other.num)
                .all(|(&a, &b)| rat_eq((a, self.den), (b, other.den)))
    }
}

impl DensityMatrix {
    /// The definition verbatim: `1/N` within a block, `0` otherwise.
    pub fn of_partition(p: &Partition) -> Self {
        let ids = p.get_ids();
        let n = ids.len();
        let mut num = vec![0u64; n * n];
        for j in 0..n {
            for k in 0..n {
                if ids[j] == ids[k] {
                    num[j * n + k] = 1;
                }
            }
        }
        Self { dim: n, num, den: n as u64 }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// An entry, as `(numerator, denominator)`.
    pub fn entry(&self, j: usize, k: usize) -> (u64, u64) {
        (self.num[j * self.dim + k], self.den)
    }

    /// `tr(ρ)` — must be `1` for a density matrix.
    pub fn trace(&self) -> (u64, u64) {
        let s: u64 = (0..self.dim).map(|i| self.num[i * self.dim + i]).sum();
        (s, self.den)
    }

    /// `tr(ρ²) = Σ_{j,k} ρ_jk · ρ_kj` — a sum of products over the materialised matrix.
    ///
    /// It never calls the lattice entropy (see the module's independence section).
    pub fn trace_of_square(&self) -> (u64, u64) {
        let mut acc: u128 = 0;
        for j in 0..self.dim {
            for k in 0..self.dim {
                acc += self.num[j * self.dim + k] as u128 * self.num[k * self.dim + j] as u128;
            }
        }
        let den = self.den as u128 * self.den as u128;
        // u64 suffices up to N = 6; an overflow here should surface immediately.
        (acc as u64, den as u64)
    }

    pub fn is_symmetric(&self) -> bool {
        (0..self.dim)
            .all(|j| (0..self.dim).all(|k| self.num[j * self.dim + k] == self.num[k * self.dim + j]))
    }

    /// Is `ρ == I/N`, the maximally mixed state of the discrete partition?
    pub fn is_identity_over_n(&self) -> bool {
        self.den == self.dim as u64
            && (0..self.dim).all(|j| {
                (0..self.dim).all(|k| self.num[j * self.dim + k] == u64::from(j == k))
            })
    }

    /// **Positive-semidefinite decomposition** — `ρ == Σ_B (|B|/N) |b⟩⟨b|` with `|b⟩ = (1/√|B|) Σ_{u∈B} |u⟩`.
    ///
    /// The square roots cancel in the outer product: entries of `|b⟩⟨b|` are `1/|B|` inside a block,
    /// so `(|B|/N)·(1/|B|) = 1/N`. The check is therefore exact integer arithmetic, and the mere
    /// existence of the decomposition shows `ρ` is positive semidefinite.
    pub fn equals_block_spectral_decomposition(&self, p: &Partition) -> bool {
        let ids = p.get_ids();
        let n = ids.len();
        if n != self.dim {
            return false;
        }
        // Σ_B (|B|/N)·(1/|B|)·[j,k ∈ B] = (1/N)·[j ~ k]
        for j in 0..n {
            for k in 0..n {
                let want = if ids[j] == ids[k] { (1u64, n as u64) } else { (0u64, 1u64) };
                if !rat_eq(self.entry(j, k), want) {
                    return false;
                }
            }
        }
        true
    }

    /// Rational equality — same verdict as [`PartialEq`], independent of the denominator used.
    pub fn equals_mixture(&self, other: &Self) -> bool {
        self == other
    }
}

/// **Lüders mixture** `Σ_C P_C ρ P_C`, where `P_C` is the 0/1 diagonal projector onto block `C` of σ.
///
/// `(P_C ρ P_C)_jk = ρ_jk` for `j,k ∈ C` and 0 elsewhere, so summing over blocks keeps `ρ_jk`
/// **only when `j` and `k` also share a block in σ** — the intersection of the two equivalences.
///
/// ⟹ `luders_mixture(ρ(π), σ) == ρ(π ∧ σ)`, where `∧` is this crate's `Meet` =
/// [`Partition::differentiate`](crate::Partition::differentiate). Measurement *is* refinement.
/// (The identity is not assumed: `d_luders_mixture_equals_meet` checks it exhaustively.)
pub fn luders_mixture(rho: &DensityMatrix, sigma: &Partition) -> DensityMatrix {
    let sids = sigma.get_ids();
    let n = rho.dim;
    assert_eq!(sids.len(), n, "sigma has a different universe size than rho");
    let mut num = vec![0u64; n * n];
    for j in 0..n {
        for k in 0..n {
            if sids[j] == sids[k] {
                num[j * n + k] = rho.num[j * n + k];
            }
        }
    }
    DensityMatrix { dim: n, num, den: rho.den }
}

/// Equal mixture `(ρ₁ + ρ₂)/2` — the common denominator is doubled to stay integral.
///
/// ⚠ The result is generally **not the density matrix of any partition**: entries lie in
/// `{0, 1/2N, 1/N}`, and `1/2N` appears exactly when only one side relates the pair, so for
/// `π ≠ σ` it falls outside the image by construction.
pub fn equal_mixture(a: &DensityMatrix, b: &DensityMatrix) -> DensityMatrix {
    assert_eq!(a.dim, b.dim, "dimension mismatch");
    let (da, db) = (a.den, b.den);
    let den = 2 * da * db;
    let num = a
        .num
        .iter()
        .zip(&b.num)
        .map(|(&x, &y)| x * db + y * da)
        .collect();
    DensityMatrix { dim: a.dim, num, den }
}
