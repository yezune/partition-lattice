//! Loss ledger: for each applicable cell, what is represented, what is discarded, and
//! what is merely pronounced.
//!
//! # Three columns, and why the third carries the load
//!
//! | Column | Question | Here |
//! |---|---|---|
//! | represented | what the encoding keeps | [`representation`], surviving distinctions |
//! | discarded | what it collapses | [`representation`], pairs split in the raw data but merged by the encoding |
//! | pronounced | what it answers without measuring | [`is_pronounced`] |
//!
//! ## Why pronouncement slips through the gates
//!
//! Matter is **fully encoded** as elements of `U`, so an applicability check passes this cell.
//! But the partition lattice is finite, so a predicate such as "is this orbit aperiodic"
//! answers "no" by the pigeonhole principle, whatever the observation says. The gates cannot
//! catch that.
//!
//! Hence the mechanical check: a predicate that gives the same answer on every input is
//! pronouncing rather than measuring.
//!
//! ## Termination is not a conclusion
//!
//! A finite lattice guarantees termination, not an answer. If the target lies outside the
//! Meet-closure of the observations, terminating still leaves no answer; see [`reachable`].

/// Represented and discarded distinctions.
#[derive(Debug, Clone)]
pub struct Representation {
    /// Pairs distinguished in the raw data.
    pub raw_distinguished: usize,
    /// Pairs still distinguished after encoding.
    pub kept: usize,
    /// Pairs split in the raw data but merged by the encoding: the discarded column.
    pub discarded: usize,
    /// `kept / raw_distinguished`.
    pub retention: f64,
}

/// `raw` holds the raw values (equal values are equivalent) and `encoded` the resulting block ids.
pub fn representation(raw: &[f64], encoded: &[u32]) -> Representation {
    let n = raw.len().min(encoded.len());
    let (mut rawd, mut kept) = (0usize, 0usize);
    for i in 0..n {
        for j in (i + 1)..n {
            if raw[i] != raw[j] {
                rawd += 1;
                if encoded[i] != encoded[j] {
                    kept += 1;
                }
            }
        }
    }
    Representation {
        raw_distinguished: rawd,
        kept,
        discarded: rawd - kept,
        retention: if rawd == 0 { f64::NAN } else { kept as f64 / rawd as f64 },
    }
}

/// A predicate giving the same answer on every input is pronouncing, not measuring.
///
/// `inputs` must contain genuinely different cases (real, random, degenerate). A single input
/// would always read as pronouncement and make the check vacuous.
#[derive(Debug, Clone, PartialEq)]
pub enum Pronounced {
    /// The same answer on every input: pronouncement. Observation cannot change it.
    Yes { answer: String },
    /// The answer varies with the input, so something is actually being measured.
    No,
    /// Fewer than two inputs: undetermined.
    Undetermined,
}

pub fn is_pronounced<T, F>(predicate: F, inputs: &[T]) -> Pronounced
where
    F: Fn(&T) -> String,
{
    if inputs.len() < 2 {
        return Pronounced::Undetermined;
    }
    let first = predicate(&inputs[0]);
    if inputs.iter().all(|x| predicate(x) == first) {
        Pronounced::Yes { answer: first }
    } else {
        Pronounced::No
    }
}

/// Whether the target lies inside the Meet-closure of the observations.
///
/// The target must be coarser than or equal to the Meet of the observations, i.e. each block
/// of the Meet must sit inside one block of the target.
#[derive(Debug, Clone, PartialEq)]
pub enum Reach {
    /// Inside the closure: terminating yields an answer.
    Reachable,
    /// Outside the closure: terminating yields no answer. Reports the number of offending pairs.
    Outside { violating_pairs: usize },
}

pub fn reachable(target: &[u32], observations: &[Vec<u32>]) -> Reach {
    let n = target.len();
    if observations.is_empty() {
        return Reach::Outside { violating_pairs: 0 };
    }
    // Meet: same block iff the same block in every observation.
    let same_in_meet = |i: usize, j: usize| observations.iter().all(|o| o[i] == o[j]);
    let mut bad = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            // The Meet cannot separate them but the target does, so it is underivable.
            if same_in_meet(i, j) && target[i] != target[j] {
                bad += 1;
            }
        }
    }
    if bad == 0 { Reach::Reachable } else { Reach::Outside { violating_pairs: bad } }
}

pub fn render(cell: &str, rep: &Representation, pron: &[(String, Pronounced)], reach: &Reach) -> String {
    let mut s = String::new();
    s.push_str(&format!("**Loss ledger for `{cell}`**\n\n"));
    s.push_str("| Column | Value |\n|---|--:|\n");
    s.push_str(&format!("| **represented**: surviving distinguished pairs | {} |\n", rep.kept));
    s.push_str(&format!("| **discarded**: pairs split in raw data but merged | **{}** |\n", rep.discarded));
    s.push_str(&format!("| retention | {:.4} |\n", rep.retention));
    s.push_str("\n**Pronouncement**: a predicate with the same answer on every input is pronouncing, not measuring\n\n");
    s.push_str("| Predicate | Verdict |\n|---|---|\n");
    for (name, p) in pron {
        s.push_str(&format!(
            "| {name} | {} |\n",
            match p {
                Pronounced::Yes { answer } =>
                    format!("**pronounced**: always `{answer}`, whatever the observation"),
                Pronounced::No => "genuinely measured".to_string(),
                Pronounced::Undetermined => "undetermined (fewer than two inputs)".to_string(),
            }
        ));
    }
    s.push_str(&format!(
        "\n**Termination is not a conclusion**: is the target inside the Meet-closure? {}\n",
        match reach {
            Reach::Reachable => "reachable: terminating yields an answer".to_string(),
            Reach::Outside { violating_pairs } =>
                format!("**outside the closure**: {violating_pairs} offending pairs, so terminating yields no answer"),
        }
    ));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_ledger_counts_discarded_distinctions() {
        // Four distinct raw values give 6 pairs; a two-block encoding keeps 4 of them.
        let raw = vec![1.0, 2.0, 10.0, 20.0];
        let enc = vec![0u32, 0, 1, 1];
        let r = representation(&raw, &enc);
        assert_eq!(r.raw_distinguished, 6);
        assert_eq!(r.kept, 4);
        assert_eq!(r.discarded, 2);
    }

    #[test]
    fn test_loss_ledger_pronounced_predicate_is_detected() {
        // A predicate that always gives the same answer on a finite lattice.
        let inputs = vec![vec![0u32, 1, 2], vec![0, 0, 0], vec![2, 1, 0]];
        let p = is_pronounced(|v: &Vec<u32>| format!("{}", v.len() < 1000), &inputs);
        assert_eq!(p, Pronounced::Yes { answer: "true".into() });
        assert!(render("x", &representation(&[1.0], &[0]), &[("is_finite".into(), p)], &Reach::Reachable)
            .contains("pronounced"));
    }

    #[test]
    fn test_loss_ledger_measuring_predicate_is_not_pronounced() {
        let inputs = vec![vec![0u32, 1, 2], vec![0, 0, 0]];
        let p = is_pronounced(|v: &Vec<u32>| format!("{}", v.iter().collect::<std::collections::BTreeSet<_>>().len()), &inputs);
        assert_eq!(p, Pronounced::No);
    }

    #[test]
    fn test_loss_ledger_single_input_is_undetermined() {
        // A single input would always look pronounced, so the verdict must be undetermined.
        assert_eq!(is_pronounced(|_: &u32| "x".to_string(), &[1u32]), Pronounced::Undetermined);
    }

    #[test]
    fn test_loss_ledger_target_inside_meet_closure_is_reachable() {
        // The Meet of two observations refines the target.
        let obs = vec![vec![0u32, 0, 1, 1], vec![0, 1, 0, 1]];
        let target = vec![0u32, 0, 1, 1];
        assert_eq!(reachable(&target, &obs), Reach::Reachable);
    }

    #[test]
    fn test_loss_ledger_target_outside_closure_terminates_without_answer() {
        // The Meet cannot separate 0 and 1 but the target does, so terminating yields no answer.
        let obs = vec![vec![0u32, 0, 1, 1]];
        let target = vec![0u32, 1, 0, 1];
        match reachable(&target, &obs) {
            Reach::Outside { violating_pairs } => assert!(violating_pairs > 0),
            other => panic!("this must fall outside the closure: {other:?}"),
        }
    }
}
