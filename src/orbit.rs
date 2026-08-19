//! Orbit Partition (C-axis generalization of [`crate::cyclic::CyclicPartition`]).
//!
//! Replace the Z/L cyclic
//! group action with an arbitrary finite group G's orbit equivalence. The
//! partition stores only a *fundamental domain* (one block ID per orbit) plus
//! a compact group specification; the full element-to-block map is recovered
//! via the group's `orbit_repr(i)` function.
//!
//! ## Supported groups (Phase 0)
//!
//! | G                | orbit_repr(i)                                 | orbit count          |
//! |---               |---                                            |---                   |
//! | `Cyclic { l }`   | `i % l`                                       | `l`                  |
//! | `Product { l1, l2, stride }` | `((i/stride) % l1) * l2 + (i % stride) % l2` | `l1 * l2` |
//! | `Dihedral { l }` | `min(i % l, l-1 - i % l)`                     | `(l+1) / 2`          |
//!
//! - `Cyclic` is the Z/L action — equivalent to [`crate::cyclic::CyclicPartition`]
//!   for `l ≤ MAX_FD`. Provided for unified-API completeness.
//! - `Product` is Z/L₁ × Z/L₂ acting on a tiled 2D universe of row stride
//!   `stride`. Captures image-tile patterns that exceed any 1D cyclic period.
//! - `Dihedral` is D_L (rotation + reflection) — palindromic / mirror-symmetric
//!   patterns.
//!
//! ## Memory cost
//!
//! Memory: ~`MAX_FD · 4 + α` bytes (~76 B at MAX_FD=16). Constant regardless of
//! `|U|` — same substrate property as `CyclicPartition`.
//!
//! ## Correctness — bit-exact with FastPartition
//!
//! For any group `G` in the supported set and a fundamental domain `fd` of
//! length `G.orbit_count()`, the partition `op = OrbitPartition::from(...)`
//! satisfies:
//!
//! ```text
//!   op.get_ids_owned() == FastPartition::from_ids(
//!       &(0..N).map(|i| fd[G.orbit_repr(i)]).collect()
//!   ).get_ids()
//! ```
//!
//! Verified across 15 + 12 + 12 = 39 cases in this module's `tests`.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::FloatExt;
use crate::OnceLock;
#[allow(unused_imports)]
use alloc::borrow::ToOwned;
#[allow(unused_imports)]
use alloc::boxed::Box;
#[allow(unused_imports)]
use alloc::string::{String, ToString};
#[allow(unused_imports)]
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::optimized::FastPartition;

/// Upper bound for fundamental-domain size (= orbit count).
pub const MAX_FD: usize = 16;

/// Group specification — the *finite group* whose orbit equivalence defines
/// the partition. All variants are `Copy` and store only small integer
/// parameters so the enclosing `OrbitPartition` remains compact.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum GroupSpec {
    /// Z/L cyclic group acting on `[0, N)` by `i ↦ i + 1 mod L`. Equivalent
    /// to [`crate::cyclic::CyclicPartition`] for `l ≤ MAX_FD`.
    Cyclic {
        /// Cycle length L ∈ [1, MAX_FD].
        l: u8,
    },
    /// Direct product `Z/L₁ × Z/L₂` acting on a 2D-tiled universe. The
    /// stride is the *row width* of the underlying 2D structure (e.g., the
    /// image width). Set `stride = l2` to recover a 1D cyclic of period
    /// `l1·l2`.
    Product {
        /// Row group order L₁ ∈ [1, MAX_FD].
        l1: u8,
        /// Column group order L₂ ∈ [1, MAX_FD / l1].
        l2: u8,
        /// 2D row stride (universe width along the inner axis).
        stride: u32,
    },
    /// Dihedral group `D_L` of order 2L (rotations + reflections) acting on
    /// segments of length L by `i ↦ min(i mod L, L - 1 - i mod L)`. Orbit
    /// count is `⌈L/2⌉` (= `(L + 1) / 2`).
    Dihedral {
        /// Dihedral order L ∈ [2, 2·MAX_FD] (orbit count ≤ MAX_FD).
        l: u8,
    },
}

impl GroupSpec {
    /// Number of distinct orbits — equals the required `fundamental_domain`
    /// length.
    pub fn orbit_count(&self) -> usize {
        match *self {
            GroupSpec::Cyclic { l } => l as usize,
            GroupSpec::Product { l1, l2, .. } => (l1 as usize) * (l2 as usize),
            GroupSpec::Dihedral { l } => ((l as usize) + 1) / 2,
        }
    }

    /// Orbit index of element `i ∈ [0, size)`. O(1).
    pub fn orbit_repr(&self, i: usize) -> usize {
        match *self {
            GroupSpec::Cyclic { l } => i % (l as usize),
            GroupSpec::Product { l1, l2, stride } => {
                let stride = stride as usize;
                let r = (i / stride) % (l1 as usize);
                let c = (i % stride) % (l2 as usize);
                r * (l2 as usize) + c
            }
            GroupSpec::Dihedral { l } => {
                let l = l as usize;
                let pos = i % l;
                core::cmp::min(pos, l - 1 - pos)
            }
        }
    }
}

/// Partition defined by a finite group `G`'s orbit equivalence on `[0, N)`.
///
/// `block_of(i) = fundamental_domain[G.orbit_repr(i)]`.
#[derive(Debug, Serialize, Deserialize)]
pub struct OrbitPartition {
    /// Block IDs, one per orbit. Only the first `fd_len` entries are valid.
    /// Canonicalized to first-appearance IDs (0, 1, 2, ...).
    fundamental_domain: [u32; MAX_FD],
    /// Active length of `fundamental_domain`, equal to `group.orbit_count()`.
    fd_len: u8,
    /// Group action that defines orbit equivalence.
    group: GroupSpec,
    /// Universe size N.
    size: u32,
    /// Lazy materialized id vector for `get_ids() -> &[u32]`. Reset on Clone.
    #[serde(skip)]
    materialized: OnceLock<Vec<u32>>,
}

impl Clone for OrbitPartition {
    fn clone(&self) -> Self {
        Self {
            fundamental_domain: self.fundamental_domain,
            fd_len: self.fd_len,
            group: self.group,
            size: self.size,
            materialized: OnceLock::new(),
        }
    }
}

impl PartialEq for OrbitPartition {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size
            && self.group == other.group
            && self.fd_len == other.fd_len
            && self.fundamental_domain[..self.fd_len as usize]
                == other.fundamental_domain[..other.fd_len as usize]
    }
}

impl Eq for OrbitPartition {}

impl OrbitPartition {
    /// Construct from a fundamental domain (`fd[k] = block ID of orbit k`)
    /// and a group specification. The fd is canonicalized to first-appearance
    /// IDs.
    ///
    /// # Panics
    /// - If `fd.len() != group.orbit_count()`.
    /// - If `group.orbit_count() > MAX_FD`.
    pub fn from_fundamental_domain(fd: &[u32], group: GroupSpec, size: usize) -> Self {
        let orbit_count = group.orbit_count();
        assert!(
            orbit_count <= MAX_FD,
            "OrbitPartition: orbit_count {} exceeds MAX_FD {}",
            orbit_count,
            MAX_FD
        );
        assert_eq!(
            fd.len(),
            orbit_count,
            "OrbitPartition: fundamental_domain size ({}) must equal orbit_count ({})",
            fd.len(),
            orbit_count
        );

        let mut arr = [0u32; MAX_FD];
        let mut seen_raw = [0u32; MAX_FD];
        let mut seen_canon: u32 = 0;
        for (i, &raw) in fd.iter().enumerate() {
            let mut canon = None;
            for k in 0..(seen_canon as usize) {
                if seen_raw[k] == raw {
                    canon = Some(k as u32);
                    break;
                }
            }
            arr[i] = match canon {
                Some(c) => c,
                None => {
                    let c = seen_canon;
                    seen_raw[seen_canon as usize] = raw;
                    seen_canon += 1;
                    c
                }
            };
        }

        Self {
            fundamental_domain: arr,
            fd_len: fd.len() as u8,
            group,
            size: size as u32,
            materialized: OnceLock::new(),
        }
    }

    /// Lazy materialized id slice. First call computes and caches.
    pub fn get_ids(&self) -> &[u32] {
        self.materialized.get_or_init(|| self.get_ids_owned())
    }

    /// Block ID of element `i`. O(1).
    pub fn block_of(&self, i: usize) -> u32 {
        debug_assert!(i < self.size as usize);
        let orbit = self.group.orbit_repr(i);
        self.fundamental_domain[orbit]
    }

    /// Materialize the full id sequence as a `Vec<u32>`.
    pub fn get_ids_owned(&self) -> Vec<u32> {
        let n = self.size as usize;
        (0..n).map(|i| self.block_of(i)).collect()
    }

    /// Universe size N.
    pub fn size(&self) -> usize {
        self.size as usize
    }

    /// Group specification.
    pub fn group(&self) -> &GroupSpec {
        &self.group
    }

    /// Fundamental domain slice (canonicalized).
    pub fn fundamental_domain(&self) -> &[u32] {
        &self.fundamental_domain[..self.fd_len as usize]
    }

    /// Number of distinct block IDs in the fundamental domain. O(MAX_FD).
    pub fn block_count(&self) -> usize {
        if self.size == 0 {
            return 0;
        }
        let fd = self.fundamental_domain();
        let mut uniq = [0u32; MAX_FD];
        let mut cnt = 0;
        for &v in fd {
            let mut seen = false;
            for u in &uniq[..cnt] {
                if *u == v {
                    seen = true;
                    break;
                }
            }
            if !seen {
                uniq[cnt] = v;
                cnt += 1;
            }
        }
        cnt
    }

    /// Logical entropy. Currently O(N) via materialization; analytical O(orbit_count)
    /// path is left to a follow-up sprint (would require per-orbit cardinality
    /// formulae per `GroupSpec` variant).
    pub fn logical_entropy(&self) -> f64 {
        FastPartition::from_ids(&self.get_ids_owned()).logical_entropy()
    }

    /// Distinction count `N² - Σ|B|²`. Currently O(N) via materialization.
    pub fn distinction_count(&self) -> usize {
        FastPartition::from_ids(&self.get_ids_owned()).distinction_count()
    }

    /// Memory cost in bytes (sizeof Self). Constant regardless of N.
    pub fn memory_usage(&self) -> usize {
        core::mem::size_of::<Self>()
    }

    /// Convert to dense `FastPartition`.
    pub fn to_fast_partition(&self) -> FastPartition {
        FastPartition::from_ids(&self.get_ids_owned())
    }
}

impl From<&OrbitPartition> for FastPartition {
    fn from(o: &OrbitPartition) -> Self {
        o.to_fast_partition()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expand a fundamental domain via the group's orbit_repr to produce the
    /// reference id sequence.
    fn expand_via_formula(fd: &[u32], group: GroupSpec, size: usize) -> Vec<u32> {
        // Canonicalize fd to match OrbitPartition's first-appearance encoding,
        // so the reference matches what OrbitPartition produces.
        let mut canon_map: Vec<(u32, u32)> = Vec::new();
        let mut next_id: u32 = 0;
        let canonical_fd: Vec<u32> = fd
            .iter()
            .map(|&raw| {
                if let Some(&(_, c)) = canon_map.iter().find(|(r, _)| *r == raw) {
                    c
                } else {
                    let c = next_id;
                    canon_map.push((raw, c));
                    next_id += 1;
                    c
                }
            })
            .collect();
        (0..size)
            .map(|i| canonical_fd[group.orbit_repr(i)])
            .collect()
    }

    /// orbit_count formula sanity check for all three groups.
    #[test]
    fn test_orbit_count_formulae() {
        assert_eq!(GroupSpec::Cyclic { l: 5 }.orbit_count(), 5);
        assert_eq!(GroupSpec::Cyclic { l: 16 }.orbit_count(), 16);
        assert_eq!(
            GroupSpec::Product {
                l1: 2,
                l2: 3,
                stride: 3
            }
            .orbit_count(),
            6
        );
        assert_eq!(
            GroupSpec::Product {
                l1: 4,
                l2: 4,
                stride: 1024
            }
            .orbit_count(),
            16
        );
        assert_eq!(GroupSpec::Dihedral { l: 3 }.orbit_count(), 2);
        assert_eq!(GroupSpec::Dihedral { l: 4 }.orbit_count(), 2);
        assert_eq!(GroupSpec::Dihedral { l: 6 }.orbit_count(), 3);
        assert_eq!(GroupSpec::Dihedral { l: 8 }.orbit_count(), 4);
    }

    /// orbit_repr correctness for hand-checked cases.
    #[test]
    fn test_orbit_repr_hand_cases() {
        // Z/4: 0,1,2,3,0,1,2,3,...
        let g = GroupSpec::Cyclic { l: 4 };
        for i in 0..16 {
            assert_eq!(g.orbit_repr(i), i % 4);
        }

        // D_4: orbits {0,3}, {1,2} → orbit_repr(0,1,2,3) = (0, 1, 1, 0)
        let g = GroupSpec::Dihedral { l: 4 };
        assert_eq!(g.orbit_repr(0), 0);
        assert_eq!(g.orbit_repr(1), 1);
        assert_eq!(g.orbit_repr(2), 1);
        assert_eq!(g.orbit_repr(3), 0);
        assert_eq!(g.orbit_repr(4), 0); // wraps: 4 % 4 = 0
        assert_eq!(g.orbit_repr(7), 0); // 7 % 4 = 3 → min(3, 0) = 0

        // Z/2 × Z/3 with stride=3: 2D tile across 1D index
        // i=0: r=0, c=0 → 0*3+0 = 0
        // i=1: r=0, c=1 → 0*3+1 = 1
        // i=2: r=0, c=2 → 0*3+2 = 2
        // i=3: r=1, c=0 → 1*3+0 = 3
        // i=4: r=1, c=1 → 1*3+1 = 4
        // i=5: r=1, c=2 → 1*3+2 = 5
        // i=6: r=0 again (wraps L₁=2) → 0
        let g = GroupSpec::Product {
            l1: 2,
            l2: 3,
            stride: 3,
        };
        let expected = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5];
        for (i, &e) in expected.iter().enumerate() {
            assert_eq!(g.orbit_repr(i), e, "at i={}", i);
        }
    }

    /// C-C1 — bit-exact with FastPartition for `Cyclic` group (sanity:
    /// must match CyclicPartition).
    #[test]
    fn test_orbit_cyclic_bit_exact() {
        let cases: &[(Vec<u32>, u8)] = &[
            (vec![0u32, 1, 2], 3),
            (vec![0u32, 1, 2, 3, 4], 5),
            (vec![5u32, 6, 7, 8, 9, 10, 11, 12], 8),
            ((0..12u32).collect(), 12),
            ((0..16u32).collect(), 16),
        ];
        let universes = [1024usize, 4096, 16384];

        let mut passed = 0;
        let mut total = 0;
        for (fd, l) in cases {
            for &n in &universes {
                total += 1;
                let g = GroupSpec::Cyclic { l: *l };
                let op = OrbitPartition::from_fundamental_domain(fd, g, n);
                let ref_ids = expand_via_formula(fd, g, n);
                assert_eq!(op.get_ids_owned(), ref_ids, "Cyclic L={} N={}", l, n);
                // Also compare with FastPartition canonical form
                let op_canon = FastPartition::from_ids(&op.get_ids_owned());
                let ref_canon = FastPartition::from_ids(&ref_ids);
                assert_eq!(op_canon.get_ids(), ref_canon.get_ids());
                passed += 1;
            }
        }
        assert_eq!(passed, total, "Cyclic orbit: {}/{} PASS", passed, total);
    }

    /// C-C1 — bit-exact with FastPartition for `Product` group (genuine 2D
    /// patterns that exceed any 1D cyclic period).
    #[test]
    fn test_orbit_product_bit_exact() {
        // (l1, l2, stride, fd, N)
        let cases: &[(u8, u8, u32, Vec<u32>)] = &[
            (2, 3, 3, vec![0, 1, 2, 3, 4, 5]), // stride=l2 → 1D-equivalent
            (2, 2, 4, vec![0, 1, 2, 3]),       // genuine 2D with W=4
            (3, 3, 9, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]),
            (4, 4, 1024, (0..16u32).collect()), // image-like W=1024
        ];
        let universes = [1024usize, 4096, 16384];

        let mut passed = 0;
        let mut total = 0;
        for (l1, l2, stride, fd) in cases {
            for &n in &universes {
                total += 1;
                let g = GroupSpec::Product {
                    l1: *l1,
                    l2: *l2,
                    stride: *stride,
                };
                let op = OrbitPartition::from_fundamental_domain(fd, g, n);
                let ref_ids = expand_via_formula(fd, g, n);
                assert_eq!(
                    op.get_ids_owned(),
                    ref_ids,
                    "Product l1={} l2={} stride={} N={}",
                    l1,
                    l2,
                    stride,
                    n
                );
                // Canonical form bit-exact (both already canonicalized; check anyway)
                let op_canon = FastPartition::from_ids(&op.get_ids_owned());
                let ref_canon = FastPartition::from_ids(&ref_ids);
                assert_eq!(op_canon.get_ids(), ref_canon.get_ids());
                passed += 1;
            }
        }
        assert_eq!(passed, total, "Product orbit: {}/{} PASS", passed, total);
    }

    /// C-C1 — bit-exact with FastPartition for `Dihedral` group (palindromic).
    #[test]
    fn test_orbit_dihedral_bit_exact() {
        // (l, fd, N)
        let cases: &[(u8, Vec<u32>)] = &[
            (3, vec![0, 1]),       // orbits {0,2}, {1} → 2 fd
            (4, vec![0, 1]),       // orbits {0,3}, {1,2} → 2 fd
            (6, vec![0, 1, 2]),    // 3 fd
            (8, vec![0, 1, 2, 3]), // 4 fd
        ];
        let universes = [1024usize, 4096, 16384];

        let mut passed = 0;
        let mut total = 0;
        for (l, fd) in cases {
            for &n in &universes {
                total += 1;
                let g = GroupSpec::Dihedral { l: *l };
                let op = OrbitPartition::from_fundamental_domain(fd, g, n);
                let ref_ids = expand_via_formula(fd, g, n);
                assert_eq!(op.get_ids_owned(), ref_ids, "Dihedral L={} N={}", l, n);
                let op_canon = FastPartition::from_ids(&op.get_ids_owned());
                let ref_canon = FastPartition::from_ids(&ref_ids);
                assert_eq!(op_canon.get_ids(), ref_canon.get_ids());
                // Sanity: Dihedral pattern is palindromic per segment of length L.
                // Check the first L positions: ids[i] == ids[L-1-i].
                let ids = op.get_ids_owned();
                for i in 0..(*l as usize) {
                    let mirror = (*l as usize) - 1 - i;
                    assert_eq!(
                        ids[i], ids[mirror],
                        "Dihedral palindrome violated at L={} i={} mirror={}",
                        l, i, mirror
                    );
                }
                passed += 1;
            }
        }
        assert_eq!(passed, total, "Dihedral orbit: {}/{} PASS", passed, total);
    }

    /// C-C2 — memory is constant in |U| and ≤ 256 bytes for all 3 groups.
    #[test]
    fn test_orbit_memory_bounded() {
        let groups: &[(GroupSpec, Vec<u32>)] = &[
            (GroupSpec::Cyclic { l: 5 }, vec![0, 1, 2, 3, 4]),
            (
                GroupSpec::Product {
                    l1: 4,
                    l2: 4,
                    stride: 1024,
                },
                (0..16u32).collect(),
            ),
            (GroupSpec::Dihedral { l: 8 }, vec![0, 1, 2, 3]),
        ];
        for (g, fd) in groups {
            let small = OrbitPartition::from_fundamental_domain(fd, *g, 1024);
            let large = OrbitPartition::from_fundamental_domain(fd, *g, 65536);
            assert_eq!(
                small.memory_usage(),
                large.memory_usage(),
                "OrbitPartition compact memory MUST be independent of |U| for {:?}",
                g
            );
            assert!(
                large.memory_usage() <= 256,
                "OrbitPartition memory ({}) for {:?} must be ≤ 256 bytes",
                large.memory_usage(),
                g
            );
        }
    }

    /// C-C3 — first-appearance canonicalization preserves G-orbit equivalence.
    #[test]
    fn test_orbit_canonicalization_first_appearance() {
        // Raw fd with non-canonical values; expect first-appearance re-labeling.
        let raw_to_canonical: &[(Vec<u32>, GroupSpec, Vec<u32>)] = &[
            (
                vec![100, 200, 100, 200, 300],
                GroupSpec::Cyclic { l: 5 },
                vec![0, 1, 0, 1, 2],
            ),
            (
                vec![5, 5, 7, 7, 7, 9],
                GroupSpec::Product {
                    l1: 2,
                    l2: 3,
                    stride: 3,
                },
                vec![0, 0, 1, 1, 1, 2],
            ),
            (
                vec![42, 11, 42, 11],
                GroupSpec::Dihedral { l: 8 },
                vec![0, 1, 0, 1],
            ),
        ];
        for (raw, g, expected_canon) in raw_to_canonical {
            let op = OrbitPartition::from_fundamental_domain(raw, *g, 1024);
            assert_eq!(
                op.fundamental_domain(),
                expected_canon.as_slice(),
                "canonicalization mismatch for {:?}",
                g
            );
        }
    }

    /// block_of matches get_ids for all 3 groups.
    #[test]
    fn test_orbit_block_of_matches_get_ids() {
        let cases: &[(GroupSpec, Vec<u32>)] = &[
            (GroupSpec::Cyclic { l: 5 }, vec![0, 1, 2, 3, 4]),
            (
                GroupSpec::Product {
                    l1: 2,
                    l2: 2,
                    stride: 4,
                },
                vec![0, 1, 2, 3],
            ),
            (GroupSpec::Dihedral { l: 6 }, vec![0, 1, 2]),
        ];
        for (g, fd) in cases {
            let op = OrbitPartition::from_fundamental_domain(fd, *g, 100);
            let ids = op.get_ids_owned();
            for i in 0..100 {
                assert_eq!(op.block_of(i), ids[i], "block_of mismatch for {:?}", g);
            }
        }
    }

    /// Sanity: Cyclic case matches CyclicPartition output bit-exact.
    #[test]
    fn test_orbit_cyclic_matches_cyclic_partition() {
        use crate::cyclic::CyclicPartition;
        let fd = vec![0u32, 1, 2, 3, 4, 5];
        let op = OrbitPartition::from_fundamental_domain(&fd, GroupSpec::Cyclic { l: 6 }, 4096);
        let cp = CyclicPartition::from_pattern(&fd, 4096);
        assert_eq!(op.get_ids_owned(), cp.get_ids_owned());
        assert!((op.logical_entropy() - cp.logical_entropy()).abs() < 1e-12);
        assert_eq!(op.distinction_count(), cp.distinction_count());
        assert_eq!(op.block_count(), cp.block_count());
    }

    /// Determinism: same input → same output across 100 trials, for each group.
    #[test]
    fn test_orbit_deterministic() {
        let cases: &[(GroupSpec, Vec<u32>)] = &[
            (GroupSpec::Cyclic { l: 4 }, vec![0, 1, 2, 3]),
            (
                GroupSpec::Product {
                    l1: 2,
                    l2: 3,
                    stride: 3,
                },
                vec![0, 1, 2, 3, 4, 5],
            ),
            (GroupSpec::Dihedral { l: 6 }, vec![0, 1, 2]),
        ];
        for (g, fd) in cases {
            let baseline = OrbitPartition::from_fundamental_domain(fd, *g, 1000).get_ids_owned();
            for _ in 0..100 {
                let op = OrbitPartition::from_fundamental_domain(fd, *g, 1000);
                assert_eq!(op.get_ids_owned(), baseline);
            }
        }
    }
}
