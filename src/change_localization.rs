//! **Change localization** — a *per-cell* decomposition of `d_⊕` between two partitions.
//!
//! `cᵢ = |Pblock(i)| + |Qblock(i)| − 2·|Pblock(i) ∩ Qblock(i)|` counts how many of the
//! distinction relations involving cell `i` *flipped* between `P` and `Q`. It is
//! label-invariant, and the decomposition is exact:
//!
//! ```text
//!   Σ cᵢ = |dit(P) △ dit(Q)| = N² · d_⊕(P, Q)
//! ```
//!
//! A high `cᵢ` marks a region that changed; if `P == Q` every `cᵢ` is zero. This is the
//! *local* dual of the global ditset-XOR distance.

use crate::unified::Partition;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::FloatExt;
use crate::HashMap;
#[allow(unused_imports)]
use alloc::borrow::ToOwned;
#[allow(unused_imports)]
use alloc::boxed::Box;
#[allow(unused_imports)]
use alloc::string::{String, ToString};
#[allow(unused_imports)]
use alloc::vec::Vec;

/// Per-cell change amounts `cᵢ` for two partitions, with `Σcᵢ/N² = d_⊕`. Assumes a shared universe (uses `min(len)`).
pub fn change_localization(p: &Partition, q: &Partition) -> Vec<f64> {
    let pi = p.get_ids();
    let qi = q.get_ids();
    let n = pi.len().min(qi.len());
    let mut psize: HashMap<u32, usize> = HashMap::new();
    let mut qsize: HashMap<u32, usize> = HashMap::new();
    let mut inter: HashMap<(u32, u32), usize> = HashMap::new();
    for i in 0..n {
        *psize.entry(pi[i]).or_insert(0) += 1;
        *qsize.entry(qi[i]).or_insert(0) += 1;
        *inter.entry((pi[i], qi[i])).or_insert(0) += 1;
    }
    (0..n)
        .map(|i| {
            let pa = psize[&pi[i]];
            let qb = qsize[&qi[i]];
            let ab = inter[&(pi[i], qi[i])];
            (pa + qb - 2 * ab) as f64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_change_localization_decomposes_dxor() {
        // Exact decomposition: Σcᵢ/N² = ditset_xor_distance(P, Q).
        let p = Partition::from_ids(vec![0, 0, 1, 1]);
        let q = Partition::from_ids(vec![0, 1, 0, 1]);
        let c = change_localization(&p, &q);
        let n = 4.0f64;
        let sum: f64 = c.iter().sum();
        assert!(
            (sum / (n * n) - p.ditset_xor_distance(&q)).abs() < 1e-9,
            "Σcᵢ/N²=d_⊕"
        );
    }

    #[test]
    fn t_change_localization_zero_on_identity() {
        let p = Partition::from_ids(vec![0, 0, 1, 1]);
        assert!(
            change_localization(&p, &p).iter().all(|&c| c == 0.0),
            "P == Q implies every cell is zero"
        );
    }
}
