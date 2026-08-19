//! Leakage check: whether outcomes or labels entered the encoder's design in advance.
//!
//! Failure here means circular validation. The verdict is binary and obtained by perturbation,
//! so no threshold is needed.
//!
//! # Principle
//!
//! Leakage is the structural property that the encoder's output depends on data it should not
//! see. Such a dependency shows up when that data is perturbed:
//!
//! ```text
//!   t0 = fit(all)                       what the pipeline actually does
//!   t1 = fit(all, with held-out perturbed)
//!   t0 != t1  implies the encoder saw the held-out data: leakage
//! ```
//!
//! Two axes are measured separately: held-out leakage (did it see the evaluation data) and
//! target leakage (did it see the value to be predicted). The latter is subtler: fitting only
//! within the training set is still circular if the target is used.
//!
//! # What the check does not do
//!
//! - It does not prove the absence of leakage: a perturbation may happen to leave the output
//!   unchanged, for instance by staying inside the same quantile bin. That is why
//!   [`LeakageProbe::probe_alive`] reports a positive control; with a dead control the
//!   verdict is undetermined.
//! - It sets no threshold: a single differing bit is a dependency.

/// Result of the perturbation probe.
#[derive(Debug, Clone)]
pub struct LeakageProbe {
    /// Whether the encoder's output depends on the held-out values.
    pub held_out_influence: bool,
    /// Whether the encoder's output depends on the target values.
    pub target_influence: bool,
    /// Positive control: does perturbing the *training* values change the output?
    /// If false, the perturbation is powerless and both verdicts above are undetermined.
    pub probe_alive: bool,
}

impl LeakageProbe {
    /// Whether a verdict is possible, i.e. whether the positive control is alive.
    pub fn determinate(&self) -> bool {
        self.probe_alive
    }
    /// No leakage; meaningful only when the positive control is alive.
    pub fn clean(&self) -> Option<bool> {
        if self.probe_alive {
            Some(!self.held_out_influence && !self.target_influence)
        } else {
            None
        }
    }
}

/// How far to perturb, as a multiple of the data range; large enough to move quantile bins.
/// A small perturbation would stay in the same bin and yield a false negative.
const SHIFT_SCALE: f64 = 10.0;

fn range(v: &[f64]) -> f64 {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &x in v.iter().filter(|x| x.is_finite()) {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    if hi > lo { hi - lo } else { 1.0 }
}

fn shifted(v: &[f64], idx: &[usize], delta: f64) -> Vec<f64> {
    let mut out = v.to_vec();
    for &i in idx {
        if i < out.len() {
            out[i] += delta;
        }
    }
    out
}

/// `fit` is the fit the pipeline actually performs: `(values, target) -> encoder output`.
/// The output may be thresholds or block ids; it is only ever compared.
///
/// - `held`: held-out indices
/// - `train`: training indices, used for the positive control
pub fn probe<F, T>(fit: F, values: &[f64], target: &[f64], train: &[usize], held: &[usize]) -> LeakageProbe
where
    F: Fn(&[f64], &[f64]) -> Vec<T>,
    T: PartialEq,
{
    let base = fit(values, target);
    let dv = SHIFT_SCALE * range(values);
    let dt = SHIFT_SCALE * range(target);

    let held_out_influence = fit(&shifted(values, held, dv), target) != base;
    let target_influence = fit(values, &shifted(target, held, dt)) != base
        || fit(values, &shifted(target, train, dt)) != base;
    // Positive control: perturbing training values must change the output, since the encoder fits on them.
    let probe_alive = fit(&shifted(values, train, dv), target) != base;

    LeakageProbe { held_out_influence, target_influence, probe_alive }
}

pub fn render(p: &LeakageProbe) -> String {
    let mut s = String::new();
    s.push_str("| Axis | Result |\n|---|---|\n");
    s.push_str(&format!(
        "| **positive control**: does perturbing training change the output | {} |\n",
        if p.probe_alive { "alive" } else { "**dead: the verdicts below are void**" }
    ));
    s.push_str(&format!(
        "| held-out leakage | {} |\n",
        if p.held_out_influence { "**the encoder saw the evaluation data**" } else { "no dependency" }
    ));
    s.push_str(&format!(
        "| target leakage | {} |\n",
        if p.target_influence { "**the encoder saw the target: circular**" } else { "no dependency" }
    ));
    s.push_str(&format!(
        "\n⟹ {}\n",
        match p.clean() {
            Some(true) => "**No leakage** within the perturbation range".to_string(),
            Some(false) => "**Leakage detected**".to_string(),
            None => "**Undetermined**: the positive control is dead; do not read this as no leakage".to_string(),
        }
    ));
    s.push_str("No leakage is not a proof: a dependency the perturbation failed to move may remain.\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quantile_thresholds(v: &[f64], k: usize) -> Vec<u64> {
        let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (1..k)
            .map(|i| {
                let pos = i as f64 / k as f64 * (s.len() - 1) as f64;
                let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
                (s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)).to_bits()
            })
            .collect()
    }

    fn data() -> (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>) {
        let v: Vec<f64> = (0..60).map(|i| ((i * 17) % 53) as f64).collect();
        let t: Vec<f64> = (0..60).map(|i| ((i * 29) % 41) as f64).collect();
        let train: Vec<usize> = (0..40).collect();
        let held: Vec<usize> = (40..60).collect();
        (v, t, train, held)
    }

    #[test]
    fn test_leakage_clean_encoder_fits_train_only() {
        let (v, t, train, held) = data();
        let tr = train.clone();
        let p = probe(|vals: &[f64], _t: &[f64]| {
            let sub: Vec<f64> = tr.iter().map(|&i| vals[i]).collect();
            quantile_thresholds(&sub, 6)
        }, &v, &t, &train, &held);
        assert!(p.probe_alive, "the positive control must be alive");
        assert_eq!(p.clean(), Some(true), "a training-only fit must be leak-free: {p:?}");
    }

    #[test]
    fn test_leakage_detects_fit_on_all_data() {
        // Positive case: fitting on everything must be detected as held-out leakage.
        let (v, t, train, held) = data();
        let p = probe(|vals: &[f64], _t: &[f64]| quantile_thresholds(vals, 6), &v, &t, &train, &held);
        assert!(p.probe_alive);
        assert!(p.held_out_influence, "fitting on all data is held-out leakage: {p:?}");
        assert_eq!(p.clean(), Some(false));
    }

    #[test]
    fn test_leakage_detects_target_use() {
        // Mixing the target into the encoder is circular.
        let (v, t, train, held) = data();
        let tr = train.clone();
        let p = probe(|vals: &[f64], tgt: &[f64]| {
            let sub: Vec<f64> = tr.iter().map(|&i| vals[i] + tgt[i]).collect();
            quantile_thresholds(&sub, 6)
        }, &v, &t, &train, &held);
        assert!(p.target_influence, "target use must be detected: {p:?}");
    }

    #[test]
    fn test_leakage_dead_probe_is_undetermined_not_clean() {
        // A constant encoder sees nothing, so the verdict must be undetermined rather than leak-free.
        let (v, t, train, held) = data();
        let p = probe(|_v: &[f64], _t: &[f64]| vec![0u64; 5], &v, &t, &train, &held);
        assert!(!p.probe_alive);
        assert_eq!(p.clean(), None, "a dead control must not read as leak-free");
        assert!(render(&p).contains("Undetermined"));
    }
}
