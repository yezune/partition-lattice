//! `EncoderCard`: the encoder clause of a data contract, as executable checks.
//!
//! A prose table of properties an encoder should satisfy disqualifies nothing on its own.
//! Four of those properties are made executable here.
//!
//! | Property | Here | Verdict |
//! |---|---|---|
//! | degrees of freedom | [`EncoderCard::dof`] | **zero disqualifies** |
//! | faithfulness | do distinct inputs give distinct partitions (collision rate) | reported |
//! | invariance | affine (`ax+b`) and element permutation | reported; a mismatch with the declaration is an artefact |
//! | stability | how much blocks jump under an epsilon perturbation | reported |
//!
//! # Why most of these are reported rather than judged
//!
//! Choosing a threshold here would be inventing a gate. Zero degrees of freedom is the one
//! verdict, because it is a structural disqualification rather than a chosen cut-off.
//!
//! # Why zero degrees of freedom disqualifies
//!
//! If the phenomenon fully determines the encoder, most of the properties lose their control
//! arms and go vacuous, producing flat or perfect results.
//!
//! For example, an equal-frequency K-bin encoder forces uniform block sizes, so `h` is pinned
//! to the balanced value independently of the data. The coarseness axis then has zero freedom,
//! and its headroom reports the chosen bin count rather than the instrument. Reporting degrees
//! of freedom catches this before any measurement.

use crate::Partition;
use std::collections::{BTreeMap, BTreeSet};

/// The free choices an encoder leaves open. An empty list disqualifies.
#[derive(Debug, Clone)]
pub struct EncoderCard {
    pub name: String,
    pub universe: usize,
    /// Choices the phenomenon does not fix, such as the bin count, the threshold rule, or the fitting set.
    pub dof: Vec<String>,
}

impl EncoderCard {
    /// Zero degrees of freedom is a disqualification, not a sign of cleanliness.
    pub fn dof_admissible(&self) -> bool {
        !self.dof.is_empty()
    }
}

/// The measured properties. No thresholds: these are numbers.
#[derive(Debug, Clone)]
pub struct EncoderChecks {
    /// Fraction of distinct inputs mapping to the same partition; 0 means fully faithful.
    pub collision_rate: f64,
    /// Fraction of samples whose partition is invariant under `a*x + b` with `a > 0`.
    pub affine_invariance: f64,
    /// Fraction equivariant under permutation of the universe: `encode(sigma x) = sigma encode(x)`.
    pub permutation_equivariance: f64,
    /// Mean fraction of elements changing block under an epsilon perturbation.
    pub jump_rate: f64,
    pub samples: usize,
}

fn canon(ids: &[u32]) -> Vec<u32> {
    // Label-blind: renumber by first appearance, so equal partitions give equal vectors.
    let mut map: BTreeMap<u32, u32> = BTreeMap::new();
    let mut next = 0u32;
    ids.iter()
        .map(|k| *map.entry(*k).or_insert_with(|| { let v = next; next += 1; v }))
        .collect()
}

/// Measures all four properties. `encode` is the value-to-partition boundary, supplied by the caller.
///
/// - `samples`: observation vectors, each of length `universe`
/// - `eps`: perturbation size for the stability probe, in value-world units
pub fn measure<F>(card: &EncoderCard, encode: F, samples: &[Vec<f64>], eps: f64) -> EncoderChecks
where
    F: Fn(&[f64]) -> Vec<u32>,
{
    let n = samples.len();
    if n == 0 {
        return EncoderChecks {
            collision_rate: f64::NAN,
            affine_invariance: f64::NAN,
            permutation_equivariance: f64::NAN,
            jump_rate: f64::NAN,
            samples: 0,
        };
    }
    let enc: Vec<Vec<u32>> = samples.iter().map(|s| canon(&encode(s))).collect();

    // Faithfulness: did distinct inputs collapse onto the same partition?
    let distinct_in: usize =
        samples.iter().map(|s| s.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
            .collect::<BTreeSet<_>>().len();
    let distinct_out: usize = enc.iter().cloned().collect::<BTreeSet<_>>().len();
    let collision_rate = if distinct_in == 0 {
        f64::NAN
    } else {
        1.0 - distinct_out as f64 / distinct_in as f64
    };

    // Affine invariance with a > 0; a quantile encoder must pass this.
    let (a, b) = (2.5f64, -7.25f64);
    let aff = samples
        .iter()
        .zip(&enc)
        .filter(|(s, e)| {
            let t: Vec<f64> = s.iter().map(|x| a * x + b).collect();
            canon(&encode(&t)) == **e
        })
        .count() as f64
        / n as f64;

    // Permutation equivariance, probed with the reversal permutation.
    let perm = samples
        .iter()
        .zip(&enc)
        .filter(|(s, e)| {
            let mut t = (*s).clone();
            t.reverse();
            let mut want = (*e).clone();
            want.reverse();
            canon(&encode(&t)) == canon(&want)
        })
        .count() as f64
        / n as f64;

    // Stability: fraction of elements changing block under an epsilon perturbation.
    let mut jump = 0.0;
    for (si, s) in samples.iter().enumerate() {
        let t: Vec<f64> = s.iter().enumerate().map(|(i, x)| x + if i % 2 == 0 { eps } else { -eps }).collect();
        let e2 = canon(&encode(&t));
        let m = e2.iter().zip(&enc[si]).filter(|(x, y)| x != y).count();
        jump += m as f64 / s.len().max(1) as f64;
    }

    let _ = card;
    EncoderChecks {
        collision_rate,
        affine_invariance: aff,
        permutation_equivariance: perm,
        jump_rate: jump / n as f64,
        samples: n,
    }
}

/// Renders the card as markdown.
pub fn render(card: &EncoderCard, c: &EncoderChecks) -> String {
    let mut s = String::new();
    s.push_str(&format!("**EncoderCard `{}`**, universe {}, {} samples\n\n", card.name, card.universe, c.samples));
    s.push_str("| Property | Value | Verdict |\n|---|--:|---|\n");
    s.push_str(&format!(
        "| **degrees of freedom** ({}) | {} | {} |\n",
        card.dof.join(" · "),
        card.dof.len(),
        if card.dof_admissible() { "admissible" } else { "**disqualified**: the phenomenon fully determines the encoder" }
    ));
    s.push_str(&format!("| faithfulness (collision rate) | {:.4} | reported |\n", c.collision_rate));
    s.push_str(&format!("| invariance, affine `ax+b` | {:.4} | reported |\n", c.affine_invariance));
    s.push_str(&format!("| invariance, permutation equivariance | {:.4} | reported |\n", c.permutation_equivariance));
    s.push_str(&format!("| stability (epsilon jump rate) | {:.4} | reported |\n", c.jump_rate));
    s.push_str("\nApart from degrees of freedom no threshold is imposed: a chosen cut-off would be an invented gate.\n");
    s
}

/// Convenience wrapper returning partitions.
pub fn as_partition(ids: &[u32]) -> Partition {
    Partition::from_ids(canon(ids))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quantile K-bin encoder.
    fn quantile(k: usize) -> impl Fn(&[f64]) -> Vec<u32> {
        move |v: &[f64]| {
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let thr: Vec<f64> = (1..k)
                .map(|i| {
                    let pos = i as f64 / k as f64 * (s.len() - 1) as f64;
                    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
                    s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)
                })
                .collect();
            v.iter().map(|x| thr.iter().filter(|t| x > t).count() as u32).collect()
        }
    }

    /// Absolute-threshold encoder, which must break under an affine transform; used as a control.
    fn absolute(thr: Vec<f64>) -> impl Fn(&[f64]) -> Vec<u32> {
        move |v: &[f64]| v.iter().map(|x| thr.iter().filter(|t| x > *t).count() as u32).collect()
    }

    fn samples() -> Vec<Vec<f64>> {
        (0..40)
            .map(|s| (0..12).map(|i| ((s * 7 + i * 13) % 37) as f64 * 0.5 + s as f64).collect())
            .collect()
    }

    #[test]
    fn test_encoder_card_dof_zero_is_disqualified() {
        let c = EncoderCard { name: "x".into(), universe: 12, dof: vec![] };
        assert!(!c.dof_admissible(), "zero degrees of freedom must disqualify");
        let ok = EncoderCard { name: "x".into(), universe: 12, dof: vec!["K".into()] };
        assert!(ok.dof_admissible());
    }

    #[test]
    fn test_encoder_card_quantile_is_affine_invariant() {
        let card = EncoderCard { name: "quantile6".into(), universe: 12, dof: vec!["K".into()] };
        let c = measure(&card, quantile(6), &samples(), 0.001);
        assert!(c.affine_invariance > 0.999, "a quantile encoder must be affine invariant: {c:?}");
    }

    /// Samples with a narrow value range, so the absolute-threshold control does not degenerate.
    fn narrow_samples() -> Vec<Vec<f64>> {
        (0..40)
            .map(|s| (0..12).map(|i| ((s * 7 + i * 13) % 37) as f64 * 0.5).collect())
            .collect()
    }

    #[test]
    fn test_encoder_card_absolute_threshold_breaks_under_affine() {
        // Negative control, showing that the check can fail.
        //
        // An earlier version used the wide-range samples and got 0.500. The clue was a collision
        //    rate of 0.475: half the samples had every value above the threshold and so collapsed
        //    into one block, counting as trivially invariant. A degenerate control is not a control,
        //    hence the narrow-range samples.
        let card = EncoderCard { name: "abs".into(), universe: 12, dof: vec!["thr".into()] };
        let smp = narrow_samples();
        let abs_c = measure(&card, absolute(vec![5.0, 10.0, 15.0]), &smp, 0.001);
        let qnt_c = measure(&card, quantile(4), &smp, 0.001);

        // Degeneracy is checked structurally; a ratio threshold would be an invented gate.
        //   An earlier version asserted `collision_rate < 0.1` and thereby blocked a perfectly good
        //   control, inventing exactly the threshold this module's docs disclaim.
        let distinct: std::collections::BTreeSet<Vec<u32>> =
            smp.iter().map(|v| canon(&absolute(vec![5.0, 10.0, 15.0])(v))).collect();
        assert!(distinct.len() > 1, "a control collapsing to one partition is meaningless: {abs_c:?}");
        assert!(distinct.iter().any(|p| p.iter().max() > Some(&0)),
            "a control that is all one block is trivially invariant: {abs_c:?}");
        // Judge by contrast rather than by an invented threshold.
        assert!(
            qnt_c.affine_invariance - abs_c.affine_invariance > 0.5,
            "the quantile (invariant) and absolute (variant) encoders must separate: qnt={:.3} abs={:.3}",
            qnt_c.affine_invariance, abs_c.affine_invariance
        );
    }

    #[test]
    fn test_encoder_card_quantile_is_permutation_equivariant() {
        let card = EncoderCard { name: "quantile6".into(), universe: 12, dof: vec!["K".into()] };
        let c = measure(&card, quantile(6), &samples(), 0.001);
        assert!(c.permutation_equivariance > 0.999, "must be equivariant under element order: {c:?}");
    }

    #[test]
    fn test_encoder_card_stability_is_reported_not_gated() {
        let card = EncoderCard { name: "quantile6".into(), universe: 12, dof: vec!["K".into()] };
        let small = measure(&card, quantile(6), &samples(), 1e-9);
        let large = measure(&card, quantile(6), &samples(), 50.0);
        assert!(small.jump_rate <= large.jump_rate, "a larger perturbation must jump at least as much");
        assert!(large.jump_rate > 0.0, "a perturbation changing nothing means a dead instrument");
    }

    #[test]
    fn test_encoder_card_render_marks_disqualification() {
        let c0 = EncoderCard { name: "zero".into(), universe: 4, dof: vec![] };
        let m = measure(&c0, quantile(2), &samples(), 0.1);
        assert!(render(&c0, &m).contains("disqualified"));
    }
}
