//! **Liveness precondition** — a structural prerequisite for a measurement to mean anything.
//!
//! If a system's trajectory is *constant*, that trajectory produces zero distinctions —
//! `h(π_traj) = 0`. Any indicator computed on top of it measures **degeneracy, not ability**.
//! So the non-degeneracy of the trajectory is checked *before* the indicator is computed.
//!
//! **Lattice-native formulation:**
//! - The trajectory of a channel `c` becomes a partition `π_c` (tick indices grouped by equal value).
//! - The joint state partition is `Meet_c(π_c)` — two ticks share a block iff they agree on
//!   *every* channel.
//! - **Dead ⟺ `h(π_joint) = 0`** ⟺ at most one distinct state.
//!
//! **Scope.** This gate catches *under-variation* (nothing moves) only. The opposite
//! degeneracy — *over-variation*, where every state is unique so nothing ever recurs — is a
//! **separate** condition and is deliberately not folded in: merging them would require a
//! threshold and would blur two different failures into one verdict.
//!
//! The carrier does not matter — a body, a corpus, a policy, an instrument. The predicate is one.

use crate::unified::Partition;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::FloatExt;
use crate::HashMap;
#[allow(unused_imports)]
use alloc::vec::Vec;

/// The liveness report for a trajectory. [`Self::verdict`] gives a deterministic three-valued label.
#[derive(Clone, Debug, PartialEq)]
pub struct LivenessReport {
    /// Number of distinct joint states — the block count of `Meet_c(π_c)`.
    pub distinct_states: usize,
    /// How many channels actually vary along the trajectory (those with `h(π_c) > 0`).
    ///
    /// ⚠ **A binary count is non-injective**: a channel with `h = 0.5` and one with `h = 0.001`
    /// both count as "varying", so **near-degeneracy** (999 of 1000 ticks on one value) is invisible.
    /// To see it, read the `channel_h` **profile** below; a scalar cannot determine the partition.
    pub varying_channels: usize,
    /// Total number of channels.
    pub total_channels: usize,
    /// Logical entropy of the joint partition, `h(Meet_c(π_c))`.
    pub joint_h: f64,
    /// **Per-channel** logical entropy profile `[h(π_c)]`, in input order.
    ///
    /// This exposes the **near-degeneracy** that `varying_channels` cannot distinguish.
    /// ⚠ No threshold is applied here. [`Self::is_live`] uses only structural boundaries (0 or 1);
    /// "how constant counts as dead" is for the *reader* to decide. Putting a τ in the gate would
    /// introduce a tuned constant, which this predicate deliberately avoids.
    ///
    /// Observed in practice: a joint gate passed a run whose load-bearing axis had collapsed to a
    /// single value 99.9% of the time — which is why this field exists.
    pub channel_h: alloc::vec::Vec<f64>,
}

/// The three-valued verdict. No thresholds — every boundary is structural (0 or 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LivenessVerdict {
    /// The trajectory is constant — zero distinctions. Any verdict on top of it is **undetermined**.
    Dead,
    /// Non-degenerate but without a positive control — usable, not certified.
    AliveUncertified,
    /// Non-degenerate *and* accompanied by a positive control in the same run.
    AliveCertified,
}

impl LivenessVerdict {
    /// Integer code for reporting (0 = Dead, 1 = AliveUncertified, 2 = AliveCertified).
    pub fn code(self) -> u64 {
        match self {
            LivenessVerdict::Dead => 0,
            LivenessVerdict::AliveUncertified => 1,
            LivenessVerdict::AliveCertified => 2,
        }
    }
}

impl LivenessReport {
    /// Non-degenerate: some joint distinction exists *and* some channel varies. Structural only.
    pub fn is_live(&self) -> bool {
        self.distinct_states > 1 && self.varying_channels > 0
    }

    /// Three-valued verdict, given whether a positive control accompanied the run.
    pub fn verdict(&self, positive_control: bool) -> LivenessVerdict {
        if !self.is_live() {
            LivenessVerdict::Dead
        } else if positive_control {
            LivenessVerdict::AliveCertified
        } else {
            LivenessVerdict::AliveUncertified
        }
    }
}

/// One channel's value sequence becomes a partition, grouping tick indices by equal value.
/// It is label-blind: only *same or different* is used, never the value itself.
pub fn channel_partition<T: core::hash::Hash + Eq>(values: &[T]) -> Partition {
    let mut seen: HashMap<&T, u32> = HashMap::default();
    let mut ids: Vec<u32> = Vec::with_capacity(values.len());
    for v in values {
        let next = seen.len() as u32;
        ids.push(*seen.entry(v).or_insert(next));
    }
    Partition::from_ids(ids)
}

/// Liveness report for a multi-channel trajectory; the joint partition is `Meet_c(π_c)`.
///
/// Channels of differing length are truncated to the shortest (a shared tick axis is assumed).
/// No channels or no ticks → `Dead`: there is nothing to measure.
pub fn assess<T: core::hash::Hash + Eq>(channels: &[&[T]]) -> LivenessReport {
    let total_channels = channels.len();
    let ticks = channels.iter().map(|c| c.len()).min().unwrap_or(0);
    if total_channels == 0 || ticks == 0 {
        return LivenessReport {
            distinct_states: 0,
            varying_channels: 0,
            total_channels,
            joint_h: 0.0,
            channel_h: alloc::vec::Vec::new(),
        };
    }

    let mut varying_channels = 0usize;
    let mut channel_h: alloc::vec::Vec<f64> = alloc::vec::Vec::with_capacity(total_channels);
    let mut joint: Option<Partition> = None;
    for ch in channels {
        let p = channel_partition(&ch[..ticks]);
        channel_h.push(p.logical_entropy());
        if p.dit_count() > 0 {
            varying_channels += 1;
        }
        joint = Some(match joint {
            None => p,
            Some(acc) => acc.differentiate(&p), // Meet: joint state identity
        });
    }
    let joint = joint.expect("total_channels > 0");

    LivenessReport {
        distinct_states: joint.block_count(),
        varying_channels,
        total_channels,
        joint_h: joint.logical_entropy(),
        channel_h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_liveness_channel_partition_groups_equal_values() {
        // Label-blind: only sameness matters, not what the values are.
        let p = channel_partition(&[7u32, 7, 9, 9]);
        let q = channel_partition(&["a", "a", "b", "b"]);
        assert_eq!(p.block_count(), 2);
        assert_eq!(p.get_ids(), q.get_ids(), "labels differ but the distinction structure is the same");
    }

    #[test]
    fn test_liveness_constant_trajectory_is_dead() {
        // A constant trajectory: nothing varies across the window.
        let root = [3u32, 3, 3, 3, 3];
        let r = assess(&[&root[..]]);
        assert_eq!(r.distinct_states, 1);
        assert_eq!(r.varying_channels, 0);
        assert_eq!(r.joint_h, 0.0, "a constant trajectory has zero dits");
        assert!(!r.is_live());
        assert_eq!(r.verdict(true), LivenessVerdict::Dead, "a positive control cannot revive a dead trajectory");
    }

    #[test]
    fn test_liveness_varying_trajectory_is_live() {
        let root = [0u32, 1, 2, 1];
        let r = assess(&[&root[..]]);
        assert!(r.distinct_states > 1);
        assert_eq!(r.varying_channels, 1);
        assert!(r.joint_h > 0.0);
        assert_eq!(r.verdict(false), LivenessVerdict::AliveUncertified);
        assert_eq!(r.verdict(true), LivenessVerdict::AliveCertified);
    }

    #[test]
    fn test_liveness_joint_is_meet_of_channels() {
        // Joint state identity is the Meet of the channel partitions.
        let a = [0u32, 0, 1, 1];
        let b = [0u32, 1, 0, 1];
        let r = assess(&[&a[..], &b[..]]);
        let expected = channel_partition(&a).differentiate(&channel_partition(&b));
        assert_eq!(r.distinct_states, expected.block_count());
        assert_eq!(r.joint_h, expected.logical_entropy(), "delegates to Meet, bit-exact");
        assert_eq!(r.distinct_states, 4, "two binary channels cross into 4 joint states");
    }

    #[test]
    fn test_liveness_frozen_channels_counted_but_joint_survives() {
        // Most degrees of freedom are constant, shrinking the effective count, but the system is
        // alive as long as one varies. Death requires *every* channel to be constant.
        let moving = [0u32, 1, 0, 1];
        let frozen = [5u32, 5, 5, 5];
        let r = assess(&[&moving[..], &frozen[..], &frozen[..]]);
        assert_eq!(r.total_channels, 3);
        assert_eq!(r.varying_channels, 1, "constant channels do not count towards the effective degrees of freedom");
        assert!(r.is_live(), "one moving channel keeps the system alive");
    }

    #[test]
    fn test_liveness_all_frozen_is_dead() {
        let f1 = [5u32, 5, 5, 5];
        let f2 = [9u32, 9, 9, 9];
        let r = assess(&[&f1[..], &f2[..]]);
        assert_eq!(r.varying_channels, 0);
        assert!(!r.is_live(), "every channel constant means dead");
    }

    #[test]
    fn test_liveness_empty_input_is_dead() {
        let empty: [&[u32]; 0] = [];
        assert!(!assess(&empty).is_live(), "zero channels leaves nothing to measure");
        let no_ticks: [u32; 0] = [];
        assert!(!assess(&[&no_ticks[..]]).is_live(), "zero ticks leaves nothing to measure");
    }

    #[test]
    fn test_liveness_over_variation_is_not_dead() {
        // Boundary of scope: a trajectory whose states are all unique (no reoccurrence) is still
        // alive. That is a separate degeneracy, and the two must not be conflated.
        let unique = [0u32, 1, 2, 3, 4, 5];
        let r = assess(&[&unique[..]]);
        assert!(r.is_live(), "over-variation is not death; it is a separate axis");
        assert_eq!(r.distinct_states, 6);
    }
}

#[cfg(test)]
mod near_degeneracy_tests {
    use super::*;
    use alloc::vec;

    /// A near-degenerate channel (999 of 1000 ticks on one value) still counts as "varying",
    /// because that field is binary — so a run whose load-bearing axis is effectively constant
    /// can still be certified when the joint distinct count is large.
    ///
    /// This test pins that behaviour; the gate is deliberately left alone (no tuned threshold).
    /// The remedy is the per-channel `h` profile, which makes near-degeneracy visible; where to
    /// draw the line is the reader's decision, not the gate's.
    #[test]
    fn test_liveness_channel_h_profile_exposes_near_degeneracy() {
        let live: Vec<u32> = (0..1000).map(|i| (i % 7) as u32).collect();
        let mut near_dead: Vec<u32> = vec![0u32; 1000];
        near_dead[500] = 1; // 999 of 1000 ticks share one value: near-constant
        let r = assess(&[&live[..], &near_dead[..]]);

        // The existing contract is unchanged: both count as varying channels.
        assert_eq!(r.varying_channels, 2, "a binary count marks even a near-degenerate channel as varying");
        assert!(r.is_live());

        // The profile, however, separates them.
        assert_eq!(r.channel_h.len(), 2, "a per-channel h profile must be present");
        assert!(r.channel_h[0] > 0.5, "evenly varying channel h={}", r.channel_h[0]);
        assert!(r.channel_h[1] < 0.01, "near-degenerate channel h={}", r.channel_h[1]);
        // The scalar count cannot separate them while the profile can.
        assert!(r.channel_h[0] / r.channel_h[1] > 100.0);
    }

    /// A fully constant channel has h = 0 — the boundary is structural.
    #[test]
    fn test_liveness_channel_h_zero_for_constant() {
        let a: Vec<u32> = vec![3u32; 50];
        let b: Vec<u32> = (0..50).map(|i| (i % 2) as u32).collect();
        let r = assess(&[&a[..], &b[..]]);
        assert_eq!(r.channel_h[0], 0.0);
        assert!(r.channel_h[1] > 0.0);
        assert_eq!(r.varying_channels, 1);
    }
}
