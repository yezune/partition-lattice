//! Distributed dispatch hook — zero-overhead callback registration.
//!
//! Allows external crates (e.g., `l-vm-distributed`) to register handlers
//! that intercept `Partition::logical_entropy()` and `fused_entropy_differentiate()`
//! for large partitions, routing them to distributed MapReduce execution.
//!
//! # Design
//!
//! Uses `OnceLock<Box<dyn Fn>>` for zero-cost when no handler is registered:
//! - `OnceLock::get()` is a single atomic load (relaxed ordering).
//! - When no handler → returns `None` → local fallback with zero overhead.
//! - When handler registered → calls into distributed runtime.
//!
//! # Avoiding Circular Dependencies
//!
//! `l-vm-logic` defines the hook points; `l-vm-distributed` implements them.
//! No dependency from `l-vm-logic` → `l-vm-distributed` is needed.
//!
//! # Coexistence with AccelPolicy
//!
//! This transparent dispatch operates on the **default API path** only
//! (`Partition::logical_entropy()`, `fused_entropy_differentiate()`, etc.).
//! It fulfills FR-DA-02 from DISTRIBUTED_AGENT_REQUIREMENTS.md:
//! `monolithic → streaming → distributed` automatic scaling.
//!
//! The **explicit `_with()` path** (`Partition::logical_entropy_with(backend)`)
//! bypasses these hooks entirely — the caller-supplied `AccelerationBackend`
//! handles the computation directly. This structural separation ensures
//! GPU and Distributed never operate on the same computation simultaneously.
//!
//! ```text
//! logical_entropy()       → dispatch hook → distributed (if N ≥ 10M)
//! logical_entropy_with()  → AccelPolicy backend (GPU or CPU) — no hook
//! ```

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

use crate::Partition;

/// Handler function for distributed entropy computation.
///
/// Returns `Some(result)` if distributed execution succeeded,
/// `None` to fall back to local computation.
pub type EntropyHandler = Box<dyn Fn(&Partition) -> Option<f64> + Send + Sync>;

/// Handler function for distributed meet-entropy computation.
pub type RefineEntropyHandler = Box<dyn Fn(&Partition, &Partition) -> Option<f64> + Send + Sync>;

static ENTROPY_HOOK: OnceLock<EntropyHandler> = OnceLock::new();
static MEET_ENTROPY_HOOK: OnceLock<RefineEntropyHandler> = OnceLock::new();

/// Register a distributed entropy handler.
///
/// Called once at process startup by `l-vm-distributed::auto_init_from_env()`.
/// Returns `Err` if a handler was already registered.
pub fn register_entropy_handler(f: EntropyHandler) -> Result<(), EntropyHandler> {
    ENTROPY_HOOK.set(f)
}

/// Register a distributed meet-entropy handler.
pub fn register_refine_entropy_handler(f: RefineEntropyHandler) -> Result<(), RefineEntropyHandler> {
    MEET_ENTROPY_HOOK.set(f)
}

/// Try dispatching entropy computation to registered handler.
///
/// Returns `None` if no handler registered or handler declines (returns `None`).
pub(crate) fn try_distributed_entropy(p: &Partition) -> Option<f64> {
    ENTROPY_HOOK.get().and_then(|f| f(p))
}

/// Try dispatching meet-entropy computation to registered handler.
pub(crate) fn try_distributed_meet_entropy(p: &Partition, q: &Partition) -> Option<f64> {
    MEET_ENTROPY_HOOK.get().and_then(|f| f(p, q))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_no_handler_returns_none() {
        // OnceLock is per-process, and no handler is set in l-vm-logic tests.
        // This verifies the default no-op path.
        let pi = Partition::from_ids(vec![0, 0, 1, 1]);
        assert!(try_distributed_entropy(&pi).is_none());
    }

    #[test]
    fn test_dispatch_no_handler_meet_returns_none() {
        let pi = Partition::from_ids(vec![0, 0, 1, 1]);
        let sigma = Partition::from_ids(vec![0, 1, 0, 1]);
        assert!(try_distributed_meet_entropy(&pi, &sigma).is_none());
    }

    #[test]
    fn test_entropy_local_fallback_when_no_handler() {
        // Verify that logical_entropy() works normally without any handler
        let pi = Partition::from_ids(vec![0, 0, 1, 1]);
        let h = pi.logical_entropy();
        assert!((h - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_meet_entropy_local_fallback_when_no_handler() {
        let pi = Partition::from_ids(vec![0, 0, 1, 1]);
        let sigma = Partition::from_ids(vec![0, 1, 0, 1]);
        let h = pi.fused_entropy_differentiate(&sigma);
        assert!(h > 0.0);
    }

    // ── Dispatch-hook behaviour ─────────────────────────────

    #[test]
    fn test_hexd_with_path_bypasses_dispatch_hook() {
        // The _with(backend) path bypasses the dispatch hook:
        // it hands the raw &[u32] straight to the backend and therefore never goes
        // through Partition::logical_entropy(), which is what triggers the hook.
        use crate::acceleration::CpuBackend;

        let pi = Partition::from_ids(vec![0, 0, 1, 1, 2, 2]);
        let cpu = CpuBackend;

        // _with() path: calls backend.logical_entropy(ids) directly.
        let h_with = pi.logical_entropy_with(&cpu);
        // Default path: consults the dispatch hook, then computes locally.
        let h_default = pi.logical_entropy();

        assert!(
            (h_with - h_default).abs() < 1e-10,
            "E2: _with() and default should produce same result"
        );
    }

    #[test]
    fn test_hexd_accel_policy_exclusivity() {
        // AccelPolicy must not enable GPU and distributed execution at the same time.
        use crate::acceleration::{AccelPolicy, AccelerationBackend, CpuBackend};

        let gpu_backend = Box::new(CpuBackend) as Box<dyn AccelerationBackend>;
        let policy = AccelPolicy::Gpu(gpu_backend);

        // With GPU active, is_distributed() is false.
        assert!(policy.is_gpu(), "E3: GPU should be active");
        assert!(
            !policy.is_distributed(),
            "E3: Distributed must be inactive when GPU is active"
        );

        let dist_backend = Box::new(CpuBackend) as Box<dyn AccelerationBackend>;
        let policy2 = AccelPolicy::Distributed(dist_backend);

        assert!(
            policy2.is_distributed(),
            "E3: Distributed should be active"
        );
        assert!(
            !policy2.is_gpu(),
            "E3: GPU must be inactive when Distributed is active"
        );
    }

    #[test]
    fn test_hexd_integrate_with_bypasses_hook() {
        // integrate_with() bypasses the dispatch hook as well.
        use crate::acceleration::CpuBackend;

        let pi = Partition::from_ids(vec![0, 0, 1, 1]);
        let sigma = Partition::from_ids(vec![0, 1, 0, 1]);
        let cpu = CpuBackend;

        // _with() path: backend -> ShallowJoin, falling back to exact when size() < 128.
        let join_with = pi.integrate_with(&sigma, &cpu);
        // Default path: ExactJoin, not ShallowJoin, because integrate() delegates to integrate_exact().
        let join_default = pi.integrate(&sigma);

        // Both paths must complete without panicking.
        assert!(join_with.block_count() > 0);
        assert!(join_default.block_count() > 0);
    }
}
