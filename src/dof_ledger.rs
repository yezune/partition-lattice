//! Degrees-of-freedom ledger: report the encoder's free choices before collecting data.
//! Zero disqualifies; two or more unresolved choices must be registered as debt.
//!
//! # How this differs from [`encoder_card`](crate::encoder_card)
//!
//! `EncoderCard` counts the degrees of freedom; this asks the next question, whether each one
//! is *resolved*.
//!
//! | Term | Meaning |
//! |---|---|
//! | resolved | changing the choice does not change the verdict, so the choice does not matter |
//! | unresolved | changing it flips the verdict, so the conclusion hangs on the one point chosen |
//!
//! Resolution is therefore a sweep rather than an opinion: vary the choice and watch the verdict.
//!
//! # Why counting unresolved choices matters
//!
//! An expressibility check passes *both* of two incompatible specifications when each is
//! representable, so coverage cannot detect the ambiguity. An unresolved degree of freedom is
//! a structural blind spot of coverage.
//!
//! # Filtering trivial resolutions with a positive control
//!
//! If every alternative gives the same verdict, either the verdict is genuinely insensitive or
//! the verdict function is dead. An extreme alternative that *ought* to flip it is included;
//! if even that fails to flip it, the result is [`Resolution::DeadVerdict`], not resolution.

/// One free choice left open by the encoder.
#[derive(Debug, Clone)]
pub struct Dof {
    pub name: String,
    /// The value actually chosen.
    pub chosen: String,
    /// Other values the phenomenon does not rule out.
    pub alternatives: Vec<String>,
    /// Positive control: an extreme value that ought to flip the verdict. If it does not, the verdict function is dead.
    pub extreme: String,
    /// Whether this choice was declared before seeing the data.
    pub declared_before_data: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Resolution {
    /// Every alternative agrees *and* the extreme flips: genuinely insensitive, hence resolved.
    Resolved,
    /// Some alternative flips the verdict: unresolved, and therefore debt.
    Unresolved { flipping: Vec<String> },
    /// Not even the extreme flips it: the verdict function is dead and this must not read as resolved.
    DeadVerdict,
    /// Resolved by principle: not insensitive, but forced by another requirement.
    /// A sweep would flip the verdict, so it looks unresolved, yet there is no choice to make.
    /// For example, fitting on all data rather than the training set is forbidden by the no-leakage requirement.
    /// The reason must be recorded; applying this label without one hides an unresolved choice.
    ResolvedByPrinciple { forced_by: String },
}

#[derive(Debug, Clone)]
pub struct LedgerRow {
    pub dof: Dof,
    pub resolution: Resolution,
}

/// `verdict` maps a choice to a verdict label; equal labels count as the same verdict.
pub fn assess<F>(dof: &Dof, verdict: F) -> Resolution
where
    F: Fn(&str) -> String,
{
    let base = verdict(&dof.chosen);
    let flipping: Vec<String> =
        dof.alternatives.iter().filter(|a| verdict(a) != base).cloned().collect();
    if !flipping.is_empty() {
        return Resolution::Unresolved { flipping };
    }
    if verdict(&dof.extreme) == base {
        return Resolution::DeadVerdict;
    }
    Resolution::Resolved
}

/// Ledger verdict: two or more unresolved choices must be registered as debt.
#[derive(Debug, Clone, PartialEq)]
pub enum A3Verdict {
    /// No degrees of freedom, which disqualifies.
    DisqualifiedZeroDof,
    /// Two or more unresolved choices, to be registered as debt.
    DebtRegistrationRequired { unresolved: usize },
    /// At most one unresolved choice.
    Ok { unresolved: usize },
    /// Some entry has a dead verdict function, so the ledger cannot be trusted.
    Undetermined { dead: usize },
}

pub fn verdict(rows: &[LedgerRow]) -> A3Verdict {
    if rows.is_empty() {
        return A3Verdict::DisqualifiedZeroDof;
    }
    let dead = rows.iter().filter(|r| r.resolution == Resolution::DeadVerdict).count();
    if dead > 0 {
        return A3Verdict::Undetermined { dead };
    }
    // Resolution by principle does not count as unresolved: with no choice available, the conclusion does not hang on one point.
    let unresolved =
        rows.iter().filter(|r| matches!(r.resolution, Resolution::Unresolved { .. })).count();
    if unresolved >= 2 {
        A3Verdict::DebtRegistrationRequired { unresolved }
    } else {
        A3Verdict::Ok { unresolved }
    }
}

pub fn render(rows: &[LedgerRow]) -> String {
    let mut s = String::new();
    s.push_str("| Choice | Chosen | Alternatives | Pre-declared | Verdict |\n|---|---|---|:-:|---|\n");
    for r in rows {
        let res = match &r.resolution {
            Resolution::Resolved => "**resolved**: the verdict is unchanged".to_string(),
            Resolution::Unresolved { flipping } =>
                format!("**unresolved**, flipped by: {}", flipping.join(", ")),
            Resolution::DeadVerdict => "**dead verdict function**: not even the extreme flips it".to_string(),
            Resolution::ResolvedByPrinciple { forced_by } =>
                format!("**resolved by principle**, no choice available: {forced_by}"),
        };
        s.push_str(&format!(
            "| `{}` | {} | {} | {} | {res} |\n",
            r.dof.name,
            r.dof.chosen,
            r.dof.alternatives.join(" · "),
            if r.dof.declared_before_data { "yes" } else { "post hoc" }
        ));
    }
    s.push_str(&format!(
        "\n⟹ **A3: {}**\n",
        match verdict(rows) {
            A3Verdict::DisqualifiedZeroDof =>
                "**Disqualified**: no degrees of freedom, so the phenomenon fully determines the encoder".to_string(),
            A3Verdict::Undetermined { dead } =>
                format!("**Undetermined**: {dead} entries have a dead verdict function; do not read these as resolved"),
            A3Verdict::DebtRegistrationRequired { unresolved } =>
                format!("**Register as debt**: {unresolved} unresolved choices"),
            A3Verdict::Ok { unresolved } => format!("{unresolved} unresolved, below the debt threshold"),
        }
    ));
    s.push_str("\nResolved means the conclusion is the same whatever was chosen, not that the choice was right.\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(name: &str, chosen: &str, alts: &[&str], extreme: &str) -> Dof {
        Dof {
            name: name.into(),
            chosen: chosen.into(),
            alternatives: alts.iter().map(|s| s.to_string()).collect(),
            extreme: extreme.into(),
            declared_before_data: true,
        }
    }

    #[test]
    fn test_dof_insensitive_choice_is_resolved() {
        // Alternatives agree and the extreme flips: genuinely insensitive.
        let dof = d("K", "6", &["5", "7", "8"], "2");
        let r = assess(&dof, |v| if v.parse::<i32>().unwrap() >= 5 { "PASS".into() } else { "FAIL".into() });
        assert_eq!(r, Resolution::Resolved);
    }

    #[test]
    fn test_dof_flipping_alternative_is_unresolved() {
        let dof = d("K", "6", &["5", "12", "20"], "2");
        let r = assess(&dof, |v| if v.parse::<i32>().unwrap() <= 8 { "PASS".into() } else { "FAIL".into() });
        match r {
            Resolution::Unresolved { flipping } => assert_eq!(flipping, vec!["12", "20"]),
            other => panic!("this must be unresolved: {other:?}"),
        }
    }

    #[test]
    fn test_dof_dead_verdict_is_not_resolved() {
        // A constant verdict must not read as resolved.
        let dof = d("K", "6", &["5", "7"], "999999");
        let r = assess(&dof, |_| "PASS".into());
        assert_eq!(r, Resolution::DeadVerdict);
        let rows = vec![LedgerRow { dof, resolution: r }];
        assert_eq!(verdict(&rows), A3Verdict::Undetermined { dead: 1 });
        assert!(render(&rows).contains("Undetermined"));
    }

    #[test]
    fn test_dof_zero_is_disqualified() {
        assert_eq!(verdict(&[]), A3Verdict::DisqualifiedZeroDof);
        assert!(render(&[]).contains("Disqualified"));
    }

    #[test]
    fn test_dof_two_unresolved_requires_debt_registration() {
        let mk = |n: &str| LedgerRow {
            dof: d(n, "a", &["b"], "z"),
            resolution: Resolution::Unresolved { flipping: vec!["b".into()] },
        };
        let rows = vec![mk("x"), mk("y")];
        assert_eq!(verdict(&rows), A3Verdict::DebtRegistrationRequired { unresolved: 2 });
        // One unresolved choice is below the threshold.
        assert_eq!(verdict(&rows[..1]), A3Verdict::Ok { unresolved: 1 });
    }

    #[test]
    fn test_dof_resolved_by_principle_is_not_counted_unresolved() {
        // A sweep flips it, but another requirement forces the choice, so it is not debt.
        let rows = vec![LedgerRow {
            dof: d("fitting set", "train_only", &["all"], "all"),
            resolution: Resolution::ResolvedByPrinciple { forced_by: "the no-leakage requirement".into() },
        }];
        assert_eq!(verdict(&rows), A3Verdict::Ok { unresolved: 0 });
        assert!(render(&rows).contains("resolved by principle"));
    }

    #[test]
    fn test_dof_post_hoc_declaration_is_flagged() {
        let mut dof = d("K", "6", &["5"], "2");
        dof.declared_before_data = false;
        let rows = vec![LedgerRow { dof, resolution: Resolution::Resolved }];
        assert!(render(&rows).contains("post hoc"), "a timing violation must be visible");
    }
}
