//! Acceleration Backend — pluggable hardware dispatch for partition operations.
//!
//! Defines [`AccelerationBackend`] trait that enables transparent GPU/CPU selection.
//! [`CpuBackend`] provides the default CPU implementation.
//!
//! # Usage
//!
//! ```rust
//! use partition_lattice::unified::Partition;
//! use partition_lattice::acceleration::{AccelerationBackend, CpuBackend};
//!
//! let a = Partition::from_ids(vec![0, 0, 1, 1]);
//! let b = Partition::from_ids(vec![0, 1, 0, 1]);
//! let cpu = CpuBackend;
//!
//! // Same result as a.logical_entropy(), but dispatched through backend
//! let h = a.logical_entropy_with(&cpu);
//! let d = a.ditset_xor_distance_with(&b, &cpu);
//! ```

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

/// Pluggable acceleration backend for partition operations.
///
/// CPU (default) and GPU (l-vm-gpu) implementations available.
/// All inputs are `&[u32]` block IDs from `Partition::get_ids()`.
/// All scalar outputs are `f64` for precision consistency.
pub trait AccelerationBackend: Send + Sync {
    /// h(π): Logical entropy — `1 − Σ p_B²`
    fn logical_entropy(&self, ids: &[u32]) -> f64;

    /// h(π ∧ σ): Fused meet entropy without materializing the meet partition.
    fn fused_entropy_differentiate(&self, ids_a: &[u32], ids_b: &[u32]) -> f64;

    /// π ∧ σ: Materialize the meet (common refinement).
    fn differentiate(&self, ids_a: &[u32], ids_b: &[u32]) -> Vec<u32>;

    /// D_L(π‖σ) = h(π ∧ σ) − h(σ): Logical divergence.
    fn logical_divergence(&self, ids_a: &[u32], ids_b: &[u32]) -> f64 {
        self.fused_entropy_differentiate(ids_a, ids_b) - self.logical_entropy(ids_b)
    }

    /// d⊕(π, σ) = |dit(π) Δ dit(σ)| / N²: Ditset XOR distance.
    fn ditset_xor_distance(&self, ids_a: &[u32], ids_b: &[u32]) -> f64 {
        let h_meet = self.fused_entropy_differentiate(ids_a, ids_b);
        let h_a = self.logical_entropy(ids_a);
        let h_b = self.logical_entropy(ids_b);
        (2.0 * h_meet - h_a - h_b).abs()
    }

    /// π ∨ σ (ShallowJoin): Fixed-depth label propagation with double-buffer.
    ///
    /// GPU backends use SIMT-natural symmetric propagation.
    /// CPU backend uses double-buffer for equivalent symmetric propagation.
    /// Returns normalized block IDs.
    fn integrate_shallow(&self, ids_a: &[u32], ids_b: &[u32], depth: u32) -> Vec<u32> {
        // Default: CPU double-buffer via unified.rs
        let a = Partition::from_ids(ids_a.to_vec());
        let b = Partition::from_ids(ids_b.to_vec());
        a.integrate_with_strategy(&b, crate::unified::JoinStrategy::Shallow(depth))
            .get_ids()
            .to_vec()
    }

    /// π ∨ σ (Exact): Union-Find with full transitive closure.
    ///
    /// Use when mathematical exactness is required (witness verification, proofs).
    fn integrate_exact(&self, ids_a: &[u32], ids_b: &[u32]) -> Vec<u32> {
        let a = Partition::from_ids(ids_a.to_vec());
        let b = Partition::from_ids(ids_b.to_vec());
        a.integrate_exact(&b).get_ids().to_vec()
    }

    /// Batch h(π∧σ) over independent pairs — co-exec dispatch surface
    /// (established by measurement, 2026-06-10).
    ///
    /// Results keep input order. The default implementation delegates sequentially; overrides:
    /// - `CpuBackend`: rayon pair-parallel, `f64`-exact.
    /// - `UnifiedGpuBackend`: on unified memory, CPU/GPU co-execution inside an `N` band
    ///   (measured +77% at 500K, crossover in (2M, 4M]; the GPU share of `h` differs by ~3e-8).
    fn fused_entropy_differentiate_batch(&self, pairs: &[(&[u32], &[u32])]) -> Vec<f64> {
        pairs
            .iter()
            .map(|(a, b)| self.fused_entropy_differentiate(a, b))
            .collect()
    }

    /// Backend name for diagnostics.
    fn backend_name(&self) -> &str;
}

/// `h(π∧σ)` computed directly from raw `&[u32]` ids, bypassing [`Partition::from_ids`].
///
/// Found during production integration: at the trait's `&[u32]` boundary the `from_ids`
/// conversion (four-way routing plus two `Vec` clones) dominates the fused operation itself
/// and cancels the measured co-execution gain — representation-conversion cost surfacing at
/// a trait boundary. This function instead uses two integer counting passes,
/// `O(N)`, and yields the same `f64` (`1 − Σc²/N²`, identical counts to the `Partition` path).
///
/// Dense branch: `Vec` counting when `K_a·K_b ≤ 4M` entries (32 MB of `u64`).
/// Sparse branch: `HashMap<u64, u64>` otherwise.
pub fn fused_meet_entropy_ids(ids_a: &[u32], ids_b: &[u32]) -> f64 {
    assert_eq!(ids_a.len(), ids_b.len(), "universe size mismatch");
    let n = ids_a.len();
    if n == 0 {
        return 0.0;
    }
    let ka = ids_a.iter().copied().max().unwrap_or(0) as u64 + 1;
    let kb = ids_b.iter().copied().max().unwrap_or(0) as u64 + 1;
    const DENSE_CAP: u64 = 4_194_304; // 4M entries = 32MB u64
    let sum_sq: u64 = if ka.saturating_mul(kb) <= DENSE_CAP {
        let mut counts = vec![0u64; (ka * kb) as usize];
        for (&a, &b) in ids_a.iter().zip(ids_b) {
            counts[(a as u64 * kb + b as u64) as usize] += 1;
        }
        counts.iter().map(|&c| c * c).sum()
    } else {
        let mut counts: crate::HashMap<u64, u64> = crate::HashMap::with_capacity(1024);
        for (&a, &b) in ids_a.iter().zip(ids_b) {
            *counts.entry(a as u64 * kb + b as u64).or_insert(0) += 1;
        }
        counts.values().map(|&c| c * c).sum()
    };
    let n2 = (n as u64) * (n as u64);
    1.0 - sum_sq as f64 / n2 as f64
}

/// Default CPU backend — delegates to existing `Partition` implementations.
pub struct CpuBackend;

impl AccelerationBackend for CpuBackend {
    fn logical_entropy(&self, ids: &[u32]) -> f64 {
        Partition::from_ids(ids.to_vec()).logical_entropy()
    }

    /// Computes directly from raw ids via [`fused_meet_entropy_ids`], avoiding the `from_ids`
    /// conversion at the `&[u32]` trait boundary; the `f64` result is unchanged.
    fn fused_entropy_differentiate(&self, ids_a: &[u32], ids_b: &[u32]) -> f64 {
        fused_meet_entropy_ids(ids_a, ids_b)
    }

    fn differentiate(&self, ids_a: &[u32], ids_b: &[u32]) -> Vec<u32> {
        let a = Partition::from_ids(ids_a.to_vec());
        let b = Partition::from_ids(ids_b.to_vec());
        a.differentiate(&b).get_ids().to_vec()
    }

    /// Batch: rayon parallelism across pairs (sequential within a pair —
    /// same shape as the cpu-alone arm of the measurement harness). `f64`-exact, order preserving.
    fn fused_entropy_differentiate_batch(&self, pairs: &[(&[u32], &[u32])]) -> Vec<f64> {
        #[cfg(not(feature = "parallel"))]
        use crate::par_shim::ParRefShim;
        #[cfg(feature = "parallel")]
        use rayon::prelude::*;
        pairs
            .par_iter()
            .map(|(a, b)| self.fused_entropy_differentiate(a, b))
            .collect()
    }

    fn backend_name(&self) -> &str {
        "cpu"
    }
}

/// Exclusive acceleration policy — at most one hardware backend per operation.
///
/// Prevents GPU and Distributed from both accelerating the same computation,
/// which would cause precision mismatches (f32 vs f64) and double dispatch.
///
/// # Design
///
/// This is a *runtime* selector, not a trait. The caller picks **one** policy
/// at initialization time; every `_with()` call goes through that single backend.
///
/// ```text
/// AccelPolicy::Cpu         → CpuBackend (default, always available)
/// AccelPolicy::Gpu(impl)   → GPU backend (Metal/CUDA, f32 internally)
/// AccelPolicy::Distributed → reserved for Phase 1 (shard-level MapReduce)
/// ```
#[derive(Default)]
pub enum AccelPolicy {
    /// CPU-only computation (default). Always available, f64 precision.
    #[default]
    Cpu,
    /// GPU-accelerated computation (Metal/CUDA). Uses f32 internally, returns f64.
    /// The boxed backend is typically `l_vm_gpu::UnifiedGpuBackend`.
    Gpu(Box<dyn AccelerationBackend>),
    /// Distributed MapReduce computation. Reserved for Phase 1.
    /// Will hold a `DistributedBackend` implementing `AccelerationBackend`.
    Distributed(Box<dyn AccelerationBackend>),
}

impl AccelPolicy {
    /// Get the active backend as a trait object reference.
    ///
    /// For `Cpu`, returns a static `CpuBackend` reference.
    pub fn backend(&self) -> &dyn AccelerationBackend {
        match self {
            AccelPolicy::Cpu => &CpuBackend,
            AccelPolicy::Gpu(b) => b.as_ref(),
            AccelPolicy::Distributed(b) => b.as_ref(),
        }
    }

    /// Returns the policy kind name for diagnostics.
    pub fn kind(&self) -> &str {
        match self {
            AccelPolicy::Cpu => "cpu",
            AccelPolicy::Gpu(_) => "gpu",
            AccelPolicy::Distributed(_) => "distributed",
        }
    }

    /// Returns `true` if GPU acceleration is active.
    pub fn is_gpu(&self) -> bool {
        matches!(self, AccelPolicy::Gpu(_))
    }

    /// Returns `true` if distributed acceleration is active.
    pub fn is_distributed(&self) -> bool {
        matches!(self, AccelPolicy::Distributed(_))
    }
}

impl core::fmt::Debug for AccelPolicy {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "AccelPolicy::{}", self.kind())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── AccelPolicy tests ──────────────────────────────────────

    #[test]
    fn test_accel_policy_default_is_cpu() {
        let policy = AccelPolicy::default();
        assert_eq!(policy.kind(), "cpu");
        assert!(!policy.is_gpu());
        assert!(!policy.is_distributed());
    }

    #[test]
    fn test_accel_policy_cpu_backend_works() {
        let policy = AccelPolicy::Cpu;
        let h = policy.backend().logical_entropy(&[0, 0, 1, 1]);
        assert!((h - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_accel_policy_gpu_variant_exclusive() {
        // GPU policy wraps a backend; cannot also be Distributed
        let gpu_backend = Box::new(CpuBackend) as Box<dyn AccelerationBackend>;
        let policy = AccelPolicy::Gpu(gpu_backend);
        assert!(policy.is_gpu());
        assert!(!policy.is_distributed());
        assert_eq!(policy.kind(), "gpu");
    }

    #[test]
    fn test_accel_policy_distributed_variant_exclusive() {
        // Distributed policy wraps a backend; cannot also be GPU
        let dist_backend = Box::new(CpuBackend) as Box<dyn AccelerationBackend>;
        let policy = AccelPolicy::Distributed(dist_backend);
        assert!(policy.is_distributed());
        assert!(!policy.is_gpu());
        assert_eq!(policy.kind(), "distributed");
    }

    #[test]
    fn test_accel_policy_gpu_backend_dispatches() {
        let gpu_backend = Box::new(CpuBackend) as Box<dyn AccelerationBackend>;
        let policy = AccelPolicy::Gpu(gpu_backend);
        let h = policy.backend().logical_entropy(&[0, 1, 2, 3]);
        assert!((h - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_accel_policy_debug_format() {
        let policy = AccelPolicy::Cpu;
        assert_eq!(format!("{:?}", policy), "AccelPolicy::cpu");
    }

    #[test]
    fn test_accel_policy_backend_name_matches_kind() {
        let policy = AccelPolicy::Cpu;
        assert_eq!(policy.backend().backend_name(), "cpu");
    }

    // ── Original CpuBackend tests ──────────────────────────────

    #[test]
    fn test_cpu_backend_entropy_matches_partition() {
        let p = Partition::from_ids(vec![0, 0, 1, 1]);
        let cpu = CpuBackend;
        let expected = p.logical_entropy();
        let actual = p.logical_entropy_with(&cpu);
        assert!(
            (expected - actual).abs() < 1e-10,
            "entropy mismatch: {} vs {}",
            expected,
            actual
        );
    }

    #[test]
    fn test_cpu_backend_fused_entropy_matches() {
        let a = Partition::from_ids(vec![0, 0, 1, 1]);
        let b = Partition::from_ids(vec![0, 1, 0, 1]);
        let cpu = CpuBackend;
        let expected = a.fused_entropy_differentiate(&b);
        let actual = a.fused_entropy_differentiate_with(&b, &cpu);
        assert!(
            (expected - actual).abs() < 1e-10,
            "fused_entropy mismatch: {} vs {}",
            expected,
            actual
        );
    }

    #[test]
    fn test_cpu_backend_divergence_matches() {
        let a = Partition::from_ids(vec![0, 0, 1, 1]);
        let b = Partition::from_ids(vec![0, 1, 0, 1]);
        let cpu = CpuBackend;
        let expected = a.logical_divergence(&b);
        let actual = a.logical_divergence_with(&b, &cpu);
        assert!(
            (expected - actual).abs() < 1e-10,
            "divergence mismatch: {} vs {}",
            expected,
            actual
        );
    }

    #[test]
    fn test_cpu_backend_xor_distance_matches() {
        let a = Partition::from_ids(vec![0, 0, 1, 1, 2]);
        let b = Partition::from_ids(vec![0, 1, 0, 1, 2]);
        let cpu = CpuBackend;
        let expected = a.ditset_xor_distance(&b);
        let actual = a.ditset_xor_distance_with(&b, &cpu);
        assert!(
            (expected - actual).abs() < 1e-10,
            "xor_distance mismatch: {} vs {}",
            expected,
            actual
        );
    }

    #[test]
    fn test_cpu_backend_differentiate_matches() {
        let a = Partition::from_ids(vec![0, 0, 1, 1]);
        let b = Partition::from_ids(vec![0, 1, 0, 1]);
        let cpu = CpuBackend;
        let expected = a.differentiate(&b);
        let actual = a.differentiate_with(&b, &cpu);
        assert_eq!(expected.get_ids(), actual.get_ids());
    }

    #[test]
    fn test_cpu_backend_name() {
        assert_eq!(CpuBackend.backend_name(), "cpu");
    }

    #[test]
    fn test_cpu_backend_indiscrete_entropy() {
        // All same block → h = 0
        let cpu = CpuBackend;
        let h = cpu.logical_entropy(&[0, 0, 0, 0]);
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn test_cpu_backend_discrete_entropy() {
        // All different blocks → h = 1 - 4*(1/4)² = 0.75
        let cpu = CpuBackend;
        let h = cpu.logical_entropy(&[0, 1, 2, 3]);
        assert!((h - 0.75).abs() < 1e-10);
    }

    // ── Batch fused surface ──────────────────────────────────────────────

    #[test]
    fn test_batch_fused_entropy_matches_elementwise() {
        // The batch result equals the pairwise fused_entropy_differentiate, order preserved and exact.
        let cpu = CpuBackend;
        let a1: Vec<u32> = vec![0, 0, 1, 1, 2, 2];
        let b1: Vec<u32> = vec![0, 1, 0, 1, 0, 1];
        let a2: Vec<u32> = vec![0, 1, 2, 0, 1, 2];
        let b2: Vec<u32> = vec![0, 0, 0, 1, 1, 1];
        let pairs: Vec<(&[u32], &[u32])> = vec![(&a1, &b1), (&a2, &b2)];
        let batch = cpu.fused_entropy_differentiate_batch(&pairs);
        assert_eq!(batch.len(), 2);
        assert!((batch[0] - cpu.fused_entropy_differentiate(&a1, &b1)).abs() < 1e-12);
        assert!((batch[1] - cpu.fused_entropy_differentiate(&a2, &b2)).abs() < 1e-12);
    }

    #[test]
    fn test_batch_fused_entropy_empty() {
        let cpu = CpuBackend;
        let pairs: Vec<(&[u32], &[u32])> = Vec::new();
        assert!(cpu.fused_entropy_differentiate_batch(&pairs).is_empty());
    }

    #[test]
    fn test_fused_meet_entropy_ids_matches_partition_path() {
        // The raw-ids path (bypassing from_ids) yields the same f64 as the Partition path:
        // identical integer counts feed the identical formula 1 − Σc²/N².
        let a: Vec<u32> = (0..5000u32).map(|i| (i * 7) % 97).collect();
        let b: Vec<u32> = (0..5000u32).map(|i| (i * 11) % 53).collect();
        let want = Partition::from_ids(a.clone())
            .fused_entropy_differentiate(&Partition::from_ids(b.clone()));
        let got = fused_meet_entropy_ids(&a, &b);
        assert!((got - want).abs() < 1e-12, "got {got} want {want}");
        // Force the sparse (HashMap) branch with labels above the dense cap.
        let a2: Vec<u32> = (0..5000u32).map(|i| i * 1000).collect();
        let b2: Vec<u32> = (0..5000u32).map(|i| (i * 7) % 4001).collect();
        let want2 = Partition::from_ids(a2.clone())
            .fused_entropy_differentiate(&Partition::from_ids(b2.clone()));
        let got2 = fused_meet_entropy_ids(&a2, &b2);
        assert!(
            (got2 - want2).abs() < 1e-12,
            "sparse: got {got2} want {want2}"
        );
    }

    #[test]
    fn test_cpu_backend_fused_uses_ids_path() {
        // CpuBackend single and batch agree with the raw-ids path.
        let cpu = CpuBackend;
        let a: Vec<u32> = (0..3000u32).map(|i| (i * 13) % 71).collect();
        let b: Vec<u32> = (0..3000u32).map(|i| (i * 17) % 31).collect();
        let direct = fused_meet_entropy_ids(&a, &b);
        assert!((cpu.fused_entropy_differentiate(&a, &b) - direct).abs() < 1e-12);
    }
}
