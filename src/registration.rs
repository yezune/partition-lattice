//! Registration: whether two instruments name the same individuals the same way.
//! When it fails, combining by Meet inflates apparent certainty.
//!
//! # Why a separate check: coarseness is structurally blind to this
//!
//! Meet is a refinement, so `h(π ∧ σ) >= max(h(π), h(σ))` always holds. Combining two
//! completely unrelated instruments therefore raises `h` and lowers headroom, and so reports
//! reduced uncertainty. This is an inflation failure rather than a degeneracy, and the
//! coarseness axis cannot see it in principle.
//!
//! # What the check does
//!
//! Before combining, it asks whether the two partitions agree better than chance.
//!
//! ```text
//!   observed = d_xor(pi, sigma)                          lower means better agreement
//!   null     = { d_xor(pi, shuffled sigma) }             same coarseness, membership shuffled
//!   observed < null q05                                  registered
//! ```
//!
//! No threshold is invented: the null comes from the data. Because the null uses
//! [`crate::ablation::random_replace`], coarseness is held fixed and the check sees only
//! agreement of membership.
//!
//! # What it does not say
//!
//! - It is not value agreement. Whether two sources assign the same *number* to the same
//!   thing is a separate question needing per-source raw values.
//! - Registration is not causation: agreement can come from a common cause. This is a gate
//!   against inflation, not a proof of identity.

use crate::ablation;
use crate::Partition;

#[derive(Debug, Clone)]
pub struct RegistrationReport {
    /// Observed `d_xor(pi, sigma)`; lower means better agreement.
    pub observed: f64,
    pub null_median: f64,
    /// The null's lower 5% quantile; the observation must fall below it.
    pub null_q05: f64,
    pub draws: usize,
    /// Whether combining is permitted.
    pub registered: bool,
    /// Spread of the null; a width of 0 makes the verdict vacuous.
    pub null_spread: f64,
}

impl RegistrationReport {
    /// A degenerate null means undetermined, which must not be read as registered.
    pub fn determinate(&self) -> bool {
        self.null_spread > 0.0
    }
}

/// `a` and `b` are two instrument partitions over the same universe.
pub fn probe(a: &[u32], b: &[u32], draws: usize, seed: u64) -> RegistrationReport {
    assert_eq!(a.len(), b.len(), "registration is defined only over a shared universe");
    let pa = Partition::from_ids(a.to_vec());
    let pb = Partition::from_ids(b.to_vec());
    let observed = pa.ditset_xor_distance(&pb);

    let mut null: Vec<f64> = (0..draws)
        .map(|d| {
            let shuffled = ablation::random_replace(b, seed ^ (d as u64).wrapping_mul(0x9E37));
            pa.ditset_xor_distance(&Partition::from_ids(shuffled))
        })
        .collect();
    null.sort_by(|x, y| x.partial_cmp(y).unwrap());

    let q = |p: f64| null[((p * draws as f64) as usize).min(draws.saturating_sub(1))];
    let (q05, med) = (q(0.05), q(0.5));
    let spread = null.last().copied().unwrap_or(0.0) - null.first().copied().unwrap_or(0.0);

    RegistrationReport {
        observed,
        null_median: med,
        null_q05: q05,
        draws,
        registered: observed < q05 && spread > 0.0,
        null_spread: spread,
    }
}

pub fn render(name: &str, r: &RegistrationReport) -> String {
    let mut s = String::new();
    s.push_str(&format!("**Registration `{name}`**, {} null draws\n\n", r.draws));
    s.push_str("| Item | Value |\n|---|--:|\n");
    s.push_str(&format!("| observed `d_xor` (lower is better) | {:.6} |\n", r.observed));
    s.push_str(&format!("| null median | {:.6} |\n", r.null_median));
    s.push_str(&format!("| null **q05** (line to clear) | {:.6} |\n", r.null_q05));
    s.push_str(&format!("| null spread | {:.6} |\n", r.null_spread));
    s.push_str(&format!(
        "\n⟹ {}\n",
        if !r.determinate() {
            "**Undetermined**: the null is degenerate (zero spread); do not read this as registered".to_string()
        } else if r.registered {
            "**Registered**: agreement beats chance, so combining by Meet is permitted".to_string()
        } else {
            "**Not registered**: agreement is at chance level, so combining by Meet would inflate certainty".to_string()
        }
    ));
    s.push_str("The null preserves block sizes and shuffles only membership, so coarseness is held fixed and only agreement of membership is measured.\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registration_identical_partitions_are_registered() {
        let a: Vec<u32> = (0..60).map(|i| i % 6).collect();
        let r = probe(&a, &a, 200, 7);
        assert_eq!(r.observed, 0.0);
        assert!(r.registered, "a partition must register against itself: {r:?}");
    }

    #[test]
    fn test_registration_independent_partitions_are_unregistered() {
        // Negative control: two unrelated partitions must not register.
        //
        // An earlier version used `b = (i * 37 + 11) % 6`, but 37 is congruent to 1 mod 6, so that
        //    was merely a relabelling of `a`. Partition distance is label-blind, so it gave
        //    `d_xor = 0.0` and the check correctly reported registration; the control data was wrong.
        //    Now two structurally different partitions are used: residues versus contiguous runs.
        let a: Vec<u32> = (0..90).map(|i| i % 6).collect();
        let b: Vec<u32> = (0..90).map(|i| (i / 15) as u32).collect();
        let r = probe(&a, &b, 300, 3);
        assert!(!r.registered, "unrelated partitions registering would make the check useless: {r:?}");
    }

    #[test]
    fn test_registration_partially_aligned_is_detected() {
        // If b mostly follows a, it must register.
        let a: Vec<u32> = (0..120).map(|i| i % 6).collect();
        let b: Vec<u32> = (0..120).map(|i| if i % 10 == 0 { (i + 3) as u32 % 6 } else { i as u32 % 6 }).collect();
        let r = probe(&a, &b, 300, 11);
        assert!(r.registered, "mostly matching partitions must register: {r:?}");
    }

    #[test]
    fn test_registration_degenerate_null_is_undetermined() {
        // With a single block, shuffling changes nothing, so the null has zero spread and the verdict is undetermined.
        let a = vec![0u32; 40];
        let b = vec![0u32; 40];
        let r = probe(&a, &b, 50, 1);
        assert!(!r.determinate());
        assert!(!r.registered, "a degenerate null must not yield a registered verdict");
        assert!(render("x", &r).contains("Undetermined"));
    }

    #[test]
    fn test_registration_null_preserves_coarseness() {
        // If the null changed coarseness, the check would measure coarseness rather than registration.
        let b: Vec<u32> = (0..80).map(|i| i % 5).collect();
        let pb = Partition::from_ids(b.clone());
        let sh = Partition::from_ids(ablation::random_replace(&b, 42));
        assert!((pb.logical_entropy() - sh.logical_entropy()).abs() < 1e-12);
    }
}
