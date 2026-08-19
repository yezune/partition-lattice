//! Per-value derivation chains: the encoding side of a calibration record.
//!
//! # Why this exists
//!
//! A survey of eight candidate corpora rejected all eight in the same way: calibration
//! history exists as prose, not as data. So instead of looking for one, this builds one.
//!
//! # The record has two halves; only one is here
//!
//! | Half | What | Here |
//! |---|---|:-:|
//! | measurement | physical instrument to published value | no |
//! | encoding | published value to partition (parse, aggregate, threshold, bin) | yes |
//!
//! A traceability requirement that asks for calibration of the *measurand* is therefore not
//! satisfied by this module, which covers the encoding instead. Conflating the two would
//! declare an unmet requirement met.
//!
//! # What it still gives
//!
//! Prose asserts a chain; this replays one:
//!
//! > a chain is valid exactly when replaying it reproduces the value bit-exactly.
//!
//! It is an executable derivation rather than a description. One mismatched step breaks the
//! chain, and that verdict involves no chosen threshold.
//!
//! # What plays the role of a standard
//!
//! A metrological chain ends at a reference standard; this one ends at the SHA-256 of a fixed
//! source: an immutable, named, redistributable artefact. A different source hash means the
//! chain points at a different standard and is void.

use std::collections::BTreeMap;

/// One derivation step. `op` must be a replayable name, not prose.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub op: String,
    /// Parameters of the operation, kept sorted so the record is deterministic.
    pub params: BTreeMap<String, String>,
    /// Output of this step, compared against on replay.
    pub out: f64,
}

/// The complete derivation of a single value, from a fixed source to the final number.
#[derive(Debug, Clone)]
pub struct Derivation {
    /// The endpoint: the source's SHA-256, standing in for a reference standard.
    pub source_sha256: String,
    /// Location within the source (row, column, and so on), which makes the chain per-value.
    pub source_locator: String,
    pub steps: Vec<Step>,
}

impl Derivation {
    pub fn value(&self) -> Option<f64> {
        self.steps.last().map(|s| s.out)
    }
}

/// Replay outcome.
#[derive(Debug, Clone, PartialEq)]
pub enum Replay {
    /// Every step reproduced bit-exactly.
    Reproduced,
    /// Which step disagreed.
    Diverged { step: usize, op: String, recorded: f64, replayed: f64 },
    /// An operation that cannot be replayed: the chain has names but no execution, i.e. a prose chain.
    UnknownOp { step: usize, op: String },
    /// A different source, i.e. a different standard.
    SourceMismatch { recorded: String, actual: String },
    /// No steps at all.
    Empty,
}

impl Replay {
    pub fn ok(&self) -> bool {
        *self == Replay::Reproduced
    }
}

/// Maps an operation name to its execution, taking the previous step's output (`None` for the first) and the parameters.
pub type OpFn = fn(Option<f64>, &BTreeMap<String, String>) -> Option<f64>;

/// Replays the chain. No threshold is involved: anything short of bit-exact is broken.
pub fn replay(d: &Derivation, actual_source_sha256: &str, ops: &BTreeMap<String, OpFn>) -> Replay {
    if d.source_sha256 != actual_source_sha256 {
        return Replay::SourceMismatch {
            recorded: d.source_sha256.clone(),
            actual: actual_source_sha256.to_string(),
        };
    }
    if d.steps.is_empty() {
        return Replay::Empty;
    }
    let mut prev: Option<f64> = None;
    for (i, s) in d.steps.iter().enumerate() {
        let Some(f) = ops.get(&s.op) else {
            return Replay::UnknownOp { step: i, op: s.op.clone() };
        };
        let Some(got) = f(prev, &s.params) else {
            return Replay::Diverged { step: i, op: s.op.clone(), recorded: s.out, replayed: f64::NAN };
        };
        if got.to_bits() != s.out.to_bits() {
            return Replay::Diverged { step: i, op: s.op.clone(), recorded: s.out, replayed: got };
        }
        prev = Some(got);
    }
    Replay::Reproduced
}

pub fn render(d: &Derivation, r: &Replay) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "**Derivation chain**, source `{}...`, location `{}`, {} steps\n\n",
        &d.source_sha256[..d.source_sha256.len().min(12)],
        d.source_locator,
        d.steps.len()
    ));
    s.push_str("| # | operation | parameters | output |\n|--:|---|---|--:|\n");
    for (i, st) in d.steps.iter().enumerate() {
        let p: Vec<String> = st.params.iter().map(|(k, v)| format!("{k}={v}")).collect();
        s.push_str(&format!("| {i} | `{}` | {} | {:.6} |\n", st.op, p.join(" "), st.out));
    }
    s.push_str(&format!(
        "\n**Replay: {}**\n",
        match r {
            Replay::Reproduced => "reproduced bit-exactly, the chain is valid".to_string(),
            Replay::Diverged { step, op, recorded, replayed } =>
                format!("**step {step} `{op}` disagreed**: recorded {recorded} vs replayed {replayed}"),
            Replay::UnknownOp { step, op } =>
                format!("**step {step} `{op}` cannot be replayed**: a name without an execution, i.e. a prose chain"),
            Replay::SourceMismatch { .. } => "**source mismatch**: this points at a different standard".to_string(),
            Replay::Empty => "no steps".to_string(),
        }
    ));
    s.push_str("\nThis is an *encoding* chain. The *measurement* chain, from physical instrument to published\n");
    s.push_str("value, is not covered here; do not read one as the other.\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(kv: &[(&str, &str)]) -> BTreeMap<String, String> {
        kv.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    fn ops() -> BTreeMap<String, OpFn> {
        let mut m: BTreeMap<String, OpFn> = BTreeMap::new();
        m.insert("literal".into(), |_, q| q.get("v").and_then(|s| s.parse().ok()));
        m.insert("scale".into(), |x, q| {
            Some(x? * q.get("by")?.parse::<f64>().ok()?)
        });
        m.insert("log10".into(), |x, _| Some(x?.log10()));
        m
    }

    fn good() -> Derivation {
        Derivation {
            source_sha256: "abc123".into(),
            source_locator: "row=7,col=density".into(),
            steps: vec![
                Step { op: "literal".into(), params: p(&[("v", "1000")]), out: 1000.0 },
                Step { op: "scale".into(), params: p(&[("by", "0.001")]), out: 1.0 },
                Step { op: "log10".into(), params: p(&[]), out: 0.0 },
            ],
        }
    }

    #[test]
    fn test_derivation_valid_chain_replays_bit_exact() {
        assert_eq!(replay(&good(), "abc123", &ops()), Replay::Reproduced);
        assert!(replay(&good(), "abc123", &ops()).ok());
    }

    #[test]
    fn test_derivation_tampered_value_breaks_chain() {
        // Changing one digit must break the chain; a prose chain would not catch this.
        let mut d = good();
        d.steps[1].out = 1.0000001;
        match replay(&d, "abc123", &ops()) {
            Replay::Diverged { step, .. } => assert_eq!(step, 1),
            other => panic!("the mismatch must be caught: {other:?}"),
        }
    }

    #[test]
    fn test_derivation_unexecutable_op_is_document_chain() {
        // A chain with names but no execution is exactly what the surveyed corpora offered.
        let mut d = good();
        d.steps[1].op = "calibrated against NIST SRM 911b".into();
        match replay(&d, "abc123", &ops()) {
            Replay::UnknownOp { op, .. } => assert!(op.contains("NIST")),
            other => panic!("the unreplayable step must be caught: {other:?}"),
        }
        assert!(render(&d, &replay(&d, "abc123", &ops())).contains("prose chain"));
    }

    #[test]
    fn test_derivation_source_change_invalidates_chain() {
        match replay(&good(), "different", &ops()) {
            Replay::SourceMismatch { .. } => {}
            other => panic!("the source mismatch must be caught: {other:?}"),
        }
    }

    #[test]
    fn test_derivation_empty_is_not_valid() {
        let d = Derivation { source_sha256: "abc123".into(), source_locator: "x".into(), steps: vec![] };
        assert_eq!(replay(&d, "abc123", &ops()), Replay::Empty);
        assert!(!replay(&d, "abc123", &ops()).ok());
    }

    #[test]
    fn test_derivation_value_is_last_step() {
        assert_eq!(good().value(), Some(0.0));
    }
}
