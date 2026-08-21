//! # Partition Lattice
//!
//! A high-performance library for partition lattice algebra and logical information theory.
//!
//! ## Core Concepts
//!
//! A **partition** of a finite set $U$ is a collection of disjoint, non-empty subsets (blocks)
//! whose union is $U$. Partitions form a **lattice** under the refinement order, with:
//!
//! - **Meet (∧)**: Common refinement (finer) — `differentiate()`
//! - **Join (∨)**: Common coarsening (coarser) — `integrate()`
//! - **Logical entropy**: $h(\pi) = 1 - \sum p_B^2$ — distinction probability
//!
//! ## Architecture
//!
//! Multiple backends are auto-selected based on input characteristics (N, K):
//!
//! | Backend | Storage | Best for |
//! |---------|---------|----------|
//! | `PartitionLattice` | `Vec<u32>` block IDs | Small N, reference impl |
//! | `FastPartition` | `Vec<BitVec<u64>>` | SIMD-friendly, K ≤ 512 |
//! | `SparsePartition` | `RoaringBitmap` | Large N, large K |
//!
//! ## Quick Start
//!
//! ```rust
//! use partition_lattice::{Partition, PartitionOps};
//!
//! let pi = Partition::from_ids(vec![0, 0, 1, 1]);
//! let sigma = Partition::from_ids(vec![0, 1, 0, 1]);
//!
//! // Meet: common refinement
//! let meet = pi.differentiate(&sigma);
//! assert_eq!(meet.block_count(), 4);
//!
//! // Logical entropy
//! let h = pi.logical_entropy();
//! assert!((h - 0.5).abs() < 1e-10);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
// In the no_std (Lattice-only) profile the `_` fallback arms of the multi-backend
// match dispatches are unreachable (only the Lattice variant exists), and a few
// shim helpers (par-shim / OnceLock) are unused — both are legitimately dead only
// under this profile, so allow them here rather than gate every single site.
#![cfg_attr(not(feature = "std"), allow(unreachable_patterns, dead_code))]

#[macro_use]
extern crate alloc;

#[cfg(not(feature = "std"))]
pub(crate) use hashbrown::{HashMap, HashSet};
/// Crate-wide `HashMap`/`HashSet` aliases: `std::collections` under `std`,
/// `hashbrown` (no_std) otherwise. Same API; hashbrown is the upstream of the
/// std hashmap implementation, so behavior matches for our label-blind usage.
#[cfg(feature = "std")]
pub(crate) use std::collections::{HashMap, HashSet};

#[cfg(not(feature = "std"))]
pub(crate) use once_shim::OnceLock;
/// Crate-wide `OnceLock`: `std::sync::OnceLock` under `std`, a `spin::Once`
/// wrapper (same `new()` / `get_or_init()` / `get()` surface) otherwise. Used for
/// lazily-built lookup tables — same values, spin-lock init on no_std.
#[cfg(feature = "std")]
pub(crate) use std::sync::OnceLock;
#[cfg(not(feature = "std"))]
mod once_shim {
    pub struct OnceLock<T>(spin::Once<T>);
    impl<T> OnceLock<T> {
        pub const fn new() -> Self {
            Self(spin::Once::new())
        }
        #[inline]
        pub fn get_or_init<F: FnOnce() -> T>(&self, f: F) -> &T {
            self.0.call_once(f)
        }
        #[inline]
        #[allow(dead_code)]
        pub fn get(&self) -> Option<&T> {
            self.0.get()
        }
        /// `std::sync::OnceLock::set` surface. On no_std this is only reached by
        /// registration paths that no MCU deployment exercises (hook init is a
        /// std-side/distributed concern), so the check-then-init race is benign.
        #[inline]
        #[allow(dead_code)]
        pub fn set(&self, value: T) -> Result<(), T> {
            if self.0.get().is_some() {
                return Err(value);
            }
            self.0.call_once(|| value);
            Ok(())
        }
    }
    impl<T> Default for OnceLock<T> {
        fn default() -> Self {
            Self::new()
        }
    }
    impl<T> core::fmt::Debug for OnceLock<T> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("OnceLock").finish_non_exhaustive()
        }
    }
}

/// Floating-point transcendental/root methods for no_std (`sqrt`/`cos`/…), backed
/// by `libm` (the same IEEE routines std uses). Imported only under
/// `not(feature = "std")` where `f64` has no inherent methods; under `std` the
/// inherent methods win. Bit-identical to std for our usage.
#[cfg(not(feature = "std"))]
#[allow(dead_code)]
pub(crate) trait FloatExt {
    fn sqrt(self) -> Self;
    fn cos(self) -> Self;
    fn sin(self) -> Self;
    fn ceil(self) -> Self;
    fn floor(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, n: Self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
}
#[cfg(not(feature = "std"))]
#[allow(dead_code)]
impl FloatExt for f64 {
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline]
    fn cos(self) -> Self {
        libm::cos(self)
    }
    #[inline]
    fn sin(self) -> Self {
        libm::sin(self)
    }
    #[inline]
    fn ceil(self) -> Self {
        libm::ceil(self)
    }
    #[inline]
    fn floor(self) -> Self {
        libm::floor(self)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        libm::pow(self, n as f64)
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        libm::pow(self, n)
    }
    #[inline]
    fn exp(self) -> Self {
        libm::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        libm::log(self)
    }
}

#[allow(dead_code)]
pub(crate) mod partition;

/// Sequential fallbacks for rayon's parallel adapters when the `parallel` feature
/// is **off** (no_std/embedded builds). Every rayon hot path in this crate is a
/// `par_iter`/`into_par_iter` → `map`/`flat_map`/`filter_map`/`collect`/`sum`
/// chain whose std-iterator equivalent is order-preserving, so dropping rayon is
/// **bit-exact** — only the threading is removed. The small-N (N<128 Lattice) hot
/// path never engages rayon anyway (`PARALLEL_THRESHOLD = 512`), so this shim is a
/// no-op on the embedded path. `use crate::par_shim::*` mirrors `use rayon::prelude::*`.
#[cfg(not(feature = "parallel"))]
pub(crate) mod par_shim {
    /// `.par_iter()` → `.iter()` on slices (and `Vec`/arrays via deref).
    pub trait ParRefShim<T> {
        fn par_iter(&self) -> core::slice::Iter<'_, T>;
    }
    impl<T> ParRefShim<T> for [T] {
        #[inline]
        fn par_iter(&self) -> core::slice::Iter<'_, T> {
            self.iter()
        }
    }
    /// `.into_par_iter()` → `.into_iter()` for any `IntoIterator` (Range, Vec, …).
    pub trait IntoParShim: IntoIterator + Sized {
        #[inline]
        fn into_par_iter(self) -> <Self as IntoIterator>::IntoIter {
            self.into_iter()
        }
    }
    impl<T: IntoIterator> IntoParShim for T {}
}

#[cfg(feature = "research")]
pub mod ablation;
pub mod acceleration;
pub mod algorithms;
pub mod change_localization;
pub mod cover;
pub mod curvature;
#[cfg(feature = "research")]
pub mod pcc_arms;
// Dense backends (Fast/Cyclic/WideCyclic/FactorizedFast) rest on `optimized`
// (`BitVec<u64>`, which needs 64-bit atomics unavailable on 32-bit RISC-V). They
// are `std`-gated so the `no_std` profile is Lattice-only — target-linkable for
// embedded (ESP32) builds. Results are identical (same partitions); only the
// large-N memory optimization is dropped.
#[cfg(feature = "std")]
pub mod cyclic;
#[cfg(feature = "std")]
pub mod cyclic_residual;
#[cfg(feature = "std")]
pub mod cyclic_wide;
pub mod delta_refine;
#[cfg(feature = "research")]
pub mod derivation;
#[cfg(feature = "research")]
pub mod dof_ledger;
#[cfg(feature = "std")]
pub mod density;
#[cfg(feature = "std")]
pub mod dsd;
pub mod dispatch;
#[cfg(feature = "research")]
pub mod encoder_card;
pub mod display;
#[cfg(feature = "std")]
pub mod distributed;
pub mod expressiveness;
#[cfg(feature = "std")]
pub mod factorized;
#[cfg(feature = "research")]
pub mod identifiability;
pub mod incidence;
pub(crate) mod join;
#[cfg(feature = "research")]
pub mod interval;
pub mod lbpe;
#[cfg(feature = "research")]
pub mod leakage;
pub mod liveness;
#[cfg(feature = "research")]
pub mod loss_ledger;
#[cfg(feature = "research")]
pub mod observable_spec;
#[allow(dead_code)]
#[doc(hidden)]
#[cfg(feature = "std")]
pub mod optimized;
#[cfg(feature = "research")]
pub mod registration;
#[cfg(feature = "std")]
pub mod orbit;
pub mod endomap;
pub mod map;
pub mod pure_pcc;
pub mod simplicial;
#[cfg(feature = "std")]
pub mod sparse;
pub mod subset;
#[cfg(feature = "std")]
pub mod streaming;
#[cfg(feature = "research")]
pub mod supply;
pub mod tropical;
pub mod unified;

pub use acceleration::{AccelPolicy, AccelerationBackend, CpuBackend};
pub use change_localization::change_localization;
pub use curvature::discrete_curvature;
#[cfg(feature = "std")]
pub use cyclic::{cyclic_refine, CyclicPartition, MAX_CYCLE};
#[cfg(feature = "std")]
pub use cyclic_residual::CyclicWithResidual;
#[cfg(feature = "std")]
pub use cyclic_wide::{WideCyclicPartition, MAX_CYCLE_WIDE};
#[cfg(feature = "std")]
pub use factorized::{FactorizedFastPartition, DEFAULT_FACTOR, MIN_UNIVERSE_FOR_FACTORIZATION};
#[cfg(feature = "std")]
pub use orbit::{GroupSpec, OrbitPartition, MAX_FD};
pub use partition::{get_logical_ops, reset_logical_ops};
pub use pure_pcc::{antichain_select, new_distinction_count, pure_pcc_step, PureStep};
pub use unified::{DispatchAdvice, PartitionOp};
pub use unified::{Partition, PartitionOps};

/// Block ID type for partition element-to-block mappings.
/// u32 provides 4.3B range (max practical block count = 512, MAX_ELEMENTWISE_BLOCKS).
/// 50% memory reduction vs usize on 64-bit platforms (4B vs 8B per element).
pub type BlockId = u32;

/// SIMD-optimized bit operations (benchmark/external access).
#[cfg(feature = "std")]
pub mod simd_ops {
    pub use crate::optimized::simd_ops::*;
}
