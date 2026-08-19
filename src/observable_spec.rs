//! `ObservableSpec`: the observable and uncertainty clauses of a data contract, as executable checks.
//!
//! Where [`encoder_card`](crate::encoder_card) measures the lattice-side boundary (value to
//! partition), this sits in front of it and asks whether the value world matches the contract.
//!
//! | Field | Here | Verdict |
//! |---|---|---|
//! | `unit` | declaration plus an observed-magnitude witness | reported, so order-of-magnitude errors are visible |
//! | `sampling` | distribution of observation counts | reported |
//! | detection limit | bin width vs reported uncertainty | judged, with no invented threshold |
//! | missing-value policy | declaration vs loader behaviour | judged |
//! | `instrument` (sensor, calibration history) | often unavailable | recorded as blocked, with a reason |
//!
//! # Why the detection limit is a verdict: the threshold comes from the data
//!
//! When a bin is narrower than the instrument can resolve, the distinction encodes noise rather
//! than signal. Both quantities come from the data (the bin width from the encoder, the
//! uncertainty from the source), so there is no threshold left to choose.
//!
//! By contrast `unit` is not judged: any "plausible range" would be an invented gate. Instead
//! the declared unit and the observed median are printed side by side.

/// Observation spec for the value world. Blocked fields are `None`, with a recorded reason.
#[derive(Debug, Clone)]
pub struct ObservableSpec {
    pub measurand: String,
    pub unit: String,
    /// Sensor model and calibration history; structurally unavailable in some fields.
    pub instrument: Option<String>,
    pub instrument_blocked_reason: Option<String>,
    /// Missing-value policy, checked against what the loader actually does.
    pub missing_policy: MissingPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingPolicy {
    /// Drops rows with missing values, so no loaded row may contain one.
    Drop,
    /// Keeps missing values, so their presence is not a violation.
    Keep,
}

#[derive(Debug, Clone)]
pub struct ObservableChecks {
    pub n_rows: usize,
    /// Number of non-finite (NaN or infinite) loaded values.
    pub n_missing: usize,
    /// Whether the missing-value policy was respected.
    pub missing_ok: bool,
    /// Median observation, printed next to the declared `unit`; not a verdict.
    pub median: f64,
    /// Width of the narrowest bin.
    pub min_bin_width: f64,
    /// Median reported uncertainty, or `NaN` when unreported.
    pub reported_uncertainty: f64,
    /// Verdict: is the narrowest bin wider than the instrument's resolution?
    /// `None` when the uncertainty is unreported, i.e. undetermined.
    pub above_detection_limit: Option<bool>,
    /// Median observation count, where available.
    pub sampling_median: f64,
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.len() % 2 == 1 { v[v.len() / 2] } else { 0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2]) }
}

/// `thresholds` holds the encoder's bin edges in ascending order; widths are edge differences.
///
/// The outer bins are unbounded and are skipped; only the finite inner bins are measured,
/// otherwise an infinite width would pass trivially.
pub fn measure(
    spec: &ObservableSpec,
    values: &[f64],
    thresholds: &[f64],
    uncertainties: &[f64],
    counts: &[f64],
) -> ObservableChecks {
    let n_missing = values.iter().filter(|x| !x.is_finite()).count();
    let missing_ok = match spec.missing_policy {
        MissingPolicy::Drop => n_missing == 0,
        MissingPolicy::Keep => true,
    };
    let mut vv = values.to_vec();
    let med = median(&mut vv);
    let mut uu = uncertainties.to_vec();
    let ru = median(&mut uu);
    let mut cc = counts.to_vec();
    let sm = median(&mut cc);

    let min_bin_width = thresholds
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| d.is_finite())
        .fold(f64::INFINITY, f64::min);

    let above = if ru.is_finite() && min_bin_width.is_finite() {
        Some(min_bin_width > ru)
    } else {
        None
    };

    ObservableChecks {
        n_rows: values.len(),
        n_missing,
        missing_ok,
        median: med,
        min_bin_width,
        reported_uncertainty: ru,
        above_detection_limit: above,
        sampling_median: sm,
    }
}

pub fn render(spec: &ObservableSpec, c: &ObservableChecks) -> String {
    let mut s = String::new();
    s.push_str(&format!("**ObservableSpec `{}`**, {} rows\n\n", spec.measurand, c.n_rows));
    s.push_str("| Field | Value | Verdict |\n|---|--:|---|\n");
    s.push_str(&format!(
        "| declared unit, observed median | `{}`, {:.4e} | reported; the reader checks the magnitude |\n",
        spec.unit, c.median
    ));
    s.push_str(&format!(
        "| missing policy `{:?}`, non-finite values | {} | {} |\n",
        spec.missing_policy,
        c.n_missing,
        if c.missing_ok { "as declared" } else { "**violates the declaration**" }
    ));
    s.push_str(&format!("| sampling, median observation count | {:.0} | reported |\n", c.sampling_median));
    s.push_str(&format!(
        "| **detection limit**: min bin width vs reported uncertainty | {:.4e} vs {:.4e} | {} |\n",
        c.min_bin_width,
        c.reported_uncertainty,
        match c.above_detection_limit {
            Some(true) => "the bin is wider than the resolution".to_string(),
            Some(false) => "**the bin is narrower than the resolution: this distinction is noise**".to_string(),
            None => "undetermined (uncertainty unreported)".to_string(),
        }
    ));
    match (&spec.instrument, &spec.instrument_blocked_reason) {
        (Some(i), _) => s.push_str(&format!("| instrument and calibration | `{i}` | reported |\n")),
        (None, Some(r)) => s.push_str(&format!("| instrument and calibration | n/a | **blocked**: {r} |\n")),
        (None, None) => s.push_str("| instrument and calibration | n/a | **undeclared**: record a reason if blocked |\n"),
    }
    s.push_str("\nOnly the missing-value policy and the detection limit are judged: only thresholds that come from the data.\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(p: MissingPolicy) -> ObservableSpec {
        ObservableSpec {
            measurand: "density".into(),
            unit: "kg/m^3".into(),
            instrument: None,
            instrument_blocked_reason: Some("the source records only the method name".into()),
            missing_policy: p,
        }
    }

    #[test]
    fn test_observable_spec_detects_missing_policy_violation() {
        let v = vec![1.0, f64::NAN, 3.0];
        let c = measure(&spec(MissingPolicy::Drop), &v, &[1.0, 2.0], &[0.1], &[3.0]);
        assert!(!c.missing_ok, "declaring Drop while missing values remain is a violation");
        let c2 = measure(&spec(MissingPolicy::Keep), &v, &[1.0, 2.0], &[0.1], &[3.0]);
        assert!(c2.missing_ok, "under Keep, missing values are not a violation");
    }

    #[test]
    fn test_observable_spec_detection_limit_fails_when_bin_narrower() {
        // A bin width of 0.5 against a reported uncertainty of 2.0 makes the distinction noise.
        let c = measure(&spec(MissingPolicy::Keep), &[1.0, 2.0, 3.0], &[1.0, 1.5, 2.0], &[2.0], &[1.0]);
        assert_eq!(c.above_detection_limit, Some(false));
        assert!(render(&spec(MissingPolicy::Keep), &c).contains("noise"));
    }

    #[test]
    fn test_observable_spec_detection_limit_passes_when_bin_wider() {
        let c = measure(&spec(MissingPolicy::Keep), &[1.0, 2.0, 3.0], &[0.0, 100.0, 200.0], &[2.0], &[1.0]);
        assert_eq!(c.above_detection_limit, Some(true));
    }

    #[test]
    fn test_observable_spec_undetermined_without_uncertainty() {
        // Without an uncertainty the verdict is undetermined, not a pass.
        let c = measure(&spec(MissingPolicy::Keep), &[1.0, 2.0], &[0.0, 10.0], &[], &[1.0]);
        assert_eq!(c.above_detection_limit, None);
        assert!(render(&spec(MissingPolicy::Keep), &c).contains("undetermined"));
    }

    #[test]
    fn test_observable_spec_single_threshold_has_no_finite_bin() {
        // A single edge leaves no inner bin, so the width is infinite and the verdict is undetermined.
        let c = measure(&spec(MissingPolicy::Keep), &[1.0], &[5.0], &[0.1], &[1.0]);
        assert!(!c.min_bin_width.is_finite());
        assert_eq!(c.above_detection_limit, None);
    }

    #[test]
    fn test_observable_spec_blocked_instrument_is_recorded_not_hidden() {
        let c = measure(&spec(MissingPolicy::Keep), &[1.0], &[0.0, 1.0], &[0.1], &[1.0]);
        let r = render(&spec(MissingPolicy::Keep), &c);
        assert!(r.contains("blocked"), "a blocked field must leave a reason rather than silence");
    }
}
