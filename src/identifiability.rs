//! Identifiability: whether one partition hides states that disagree on the target.
//!
//! # How this differs from faithfulness
//!
//! Faithfulness in [`encoder_card`](crate::encoder_card) measures whether distinct inputs
//! collapse onto the same partition. A collision is not by itself a defect: coarsening is
//! what an encoder does. The defect appears when the collided states disagree on the target:
//! the partition then cannot separate them, and the inverse problem is unidentifiable.
//!
//! So identifiability is target spread within a block, and that spread is a prediction floor:
//! any predictor must give the same value to every element of a block.
//!
//! # Which makes this a diagnostic
//!
//! Comparing an experiment's actual error against this floor locates the bottleneck:
//!
//! - error close to the floor: the encoder is the bottleneck, and a better predictor cannot help.
//! - error far above the floor: the predictor is the bottleneck, and the encoder has headroom.
//!
//! # No threshold is imposed
//!
//! A pass mark such as "identifiability above 0.8" would be invented here. This module reports
//! the floor and places it beside the measured error; nothing more.

use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct IdentifiabilityReport {
    /// Number of distinct blocks (equivalence classes).
    pub classes: usize,
    pub n: usize,
    /// Prediction floor: the median absolute error when predicting each block's median.
    /// No predictor can go below this.
    pub floor: f64,
    /// Median absolute error when predicting the global median, i.e. the no-coarsening baseline.
    pub global_error: f64,
    /// `1 - floor/global`: how much target spread the encoder removed. Zero means it separated nothing.
    pub identifiability: f64,
    /// The block hiding the most, as `(id, target spread within it)`.
    pub worst_class: Option<(u32, f64)>,
    /// Triviality flag: with as many blocks as elements (all singletons) a floor of 0 is guaranteed by construction.
    pub all_singletons: bool,
}

fn med(v: &mut Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.len() % 2 == 1 { v[v.len() / 2] } else { 0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2]) }
}

/// `ids` is the partition (element to block) and `target` the per-element target value.
pub fn measure(ids: &[u32], target: &[f64]) -> IdentifiabilityReport {
    let n = ids.len().min(target.len());
    let mut by: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    for i in 0..n {
        if target[i].is_finite() {
            by.entry(ids[i]).or_default().push(target[i]);
        }
    }
    let classes = by.len();

    // Floor: median |error| when predicting each block's median.
    let mut errs: Vec<f64> = Vec::new();
    let mut worst: Option<(u32, f64)> = None;
    for (k, vals) in by.iter() {
        let mut v = vals.clone();
        let m = med(&mut v);
        let e: Vec<f64> = vals.iter().map(|x| (x - m).abs()).collect();
        let spread = med(&mut e.clone());
        if worst.map(|(_, s)| spread > s).unwrap_or(true) && vals.len() > 1 {
            worst = Some((*k, spread));
        }
        errs.extend(e);
    }
    let floor = med(&mut errs);

    let all: Vec<f64> = target.iter().copied().filter(|x| x.is_finite()).collect();
    let gm = med(&mut all.clone());
    let mut ge: Vec<f64> = all.iter().map(|x| (x - gm).abs()).collect();
    let global_error = med(&mut ge);

    let identifiability = if global_error > 0.0 && floor.is_finite() {
        1.0 - floor / global_error
    } else {
        f64::NAN
    };

    IdentifiabilityReport {
        classes,
        n,
        floor,
        global_error,
        identifiability,
        worst_class: worst,
        all_singletons: classes > 0 && classes == n,
    }
}

/// Renders the report, locating the bottleneck against a measured error.
pub fn render(r: &IdentifiabilityReport, arm_error: Option<f64>) -> String {
    let mut s = String::new();
    s.push_str("| Item | Value |\n|---|--:|\n");
    s.push_str(&format!("| classes / elements | {} / {} |\n", r.classes, r.n));
    s.push_str(&format!("| **prediction floor** (block medians) | {:.4} |\n", r.floor));
    s.push_str(&format!("| global-median error (no coarsening) | {:.4} |\n", r.global_error));
    s.push_str(&format!("| identifiability `1 - floor/global` | {:.4} |\n", r.identifiability));
    if let Some((k, sp)) = r.worst_class {
        s.push_str(&format!("| block hiding the most | id {k}, spread {sp:.4} |\n"));
    }
    if r.all_singletons {
        s.push_str("\n**All singletons**: a floor of 0 is guaranteed by construction, not a finding.\n");
    }
    if let Some(e) = arm_error {
        let ratio = if r.floor > 0.0 { e / r.floor } else { f64::INFINITY };
        s.push_str(&format!(
            "\n**Bottleneck**: measured error {e:.4} vs floor {:.4} (x{ratio:.2})\n{}\n",
            r.floor,
            if ratio < 1.2 {
                "the encoder is the bottleneck: a better predictor cannot help, a finer encoder is needed"
            } else {
                "the predictor is the bottleneck: the encoder still has headroom"
            }
        ));
        s.push_str("The x1.2 mark is a reading aid, not a pass mark; the judgement is reading both numbers together.\n");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identifiability_perfect_split_has_zero_floor() {
        // A partition that fully separates the target has floor 0.
        let ids = vec![0, 0, 1, 1];
        let t = vec![1.0, 1.0, 5.0, 5.0];
        let r = measure(&ids, &t);
        assert_eq!(r.floor, 0.0);
        assert!(r.identifiability > 0.99);
        assert!(!r.all_singletons);
    }

    #[test]
    fn test_identifiability_degenerate_partition_hides_everything() {
        // One block separates nothing, so identifiability is 0.
        let ids = vec![0, 0, 0, 0];
        let t = vec![1.0, 2.0, 9.0, 10.0];
        let r = measure(&ids, &t);
        assert_eq!(r.classes, 1);
        assert!(r.identifiability.abs() < 1e-12, "{r:?}");
    }

    #[test]
    fn test_identifiability_flags_all_singletons_as_trivial() {
        // All singletons: a floor of 0 is guaranteed by construction.
        let ids = vec![0, 1, 2, 3];
        let t = vec![1.0, 2.0, 9.0, 10.0];
        let r = measure(&ids, &t);
        assert_eq!(r.floor, 0.0);
        assert!(r.all_singletons);
        assert!(render(&r, None).contains("guaranteed by construction"));
    }

    #[test]
    fn test_identifiability_finds_worst_hiding_class() {
        let ids = vec![0, 0, 1, 1];
        let t = vec![1.0, 1.1, 0.0, 20.0];
        let r = measure(&ids, &t);
        assert_eq!(r.worst_class.map(|(k, _)| k), Some(1), "the high-spread block must be flagged");
    }

    #[test]
    fn test_identifiability_diagnoses_encoder_vs_predictor_bottleneck() {
        let ids = vec![0, 0, 1, 1];
        let t = vec![1.0, 3.0, 10.0, 12.0];
        let r = measure(&ids, &t);
        assert!(render(&r, Some(r.floor * 1.05)).contains("encoder is the bottleneck"));
        assert!(render(&r, Some(r.floor * 5.0)).contains("predictor is the bottleneck"));
    }
}
