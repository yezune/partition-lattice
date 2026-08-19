//! Standard errors and confidence intervals, in one place.
//!
//! # This module does not change any published number
//!
//! The forms collected here are the ones already dominant across the host project, copied
//! verbatim; `tests/interval_parity.rs` pins them bit-exactly against the inline
//! implementations. Only once that is pinned can call sites be switched to delegate
//! without moving any sealed figure.
//!
//! # Contents
//!
//! | Function | Form |
//! |---|---|
//! | [`se_mean`] `(var/n).sqrt()` | `let se = (var / n as f64).sqrt()` × 4 |
//! | [`se_cluster`] | `var.sqrt() / sqrt(k)`, cluster bootstrap |
//! | [`ci95`] | `m ± 1.96 * se` |
//!
//! Note that `1.96` is a normal-approximation constant. It is a verification aid rather than
//! (no background probability is assumed), so any interval built on it
//! a primary lattice-native quantity, and is recorded as such next to the constant.

/// Two-sided 95% normal-approximation coefficient. See the module note before using it.
pub const Z95: f64 = 1.96;

/// Standard error of the mean, in the form `(var / n).sqrt()`.
///
/// Returns `NaN` for `n = 0`: no interval is produced without samples.
pub fn se_mean(var: f64, n: usize) -> f64 {
    if n == 0 {
        return f64::NAN;
    }
    (var / n as f64).sqrt()
}

/// Cluster-bootstrap standard error, `var.sqrt() / sqrt(k)` for `k` clusters.
pub fn se_cluster(var: f64, clusters: usize) -> f64 {
    if clusters == 0 {
        return f64::NAN;
    }
    var.sqrt() / (clusters as f64).sqrt()
}

/// Unbiased sample variance (denominator `n - 1`); `NaN` for `n < 2`.
pub fn sample_var(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 2 {
        return f64::NAN;
    }
    let m = v.iter().sum::<f64>() / n as f64;
    v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1) as f64
}

/// `m ± Z95 * se`, i.e. `m - 1.96 * se` and `m + 1.96 * se`.
pub fn ci95(mean: f64, se: f64) -> (f64, f64) {
    (mean - Z95 * se, mean + Z95 * se)
}

/// Mean, standard error and interval, straight from a slice.
pub fn mean_ci95(v: &[f64]) -> (f64, f64, (f64, f64)) {
    let n = v.len();
    if n == 0 {
        return (f64::NAN, f64::NAN, (f64::NAN, f64::NAN));
    }
    let m = v.iter().sum::<f64>() / n as f64;
    let se = se_mean(sample_var(v), n);
    (m, se, ci95(m, se))
}

/// Whether the interval clears `chance`, i.e. whether its lower bound exceeds it.
pub fn lower_ci_above(mean: f64, se: f64, chance: f64) -> bool {
    ci95(mean, se).0 > chance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_se_mean_matches_inline_form() {
        // Bit-exact against the inline form.
        let (var, n) = (2.25f64, 9usize);
        assert_eq!(se_mean(var, n), (var / n as f64).sqrt());
    }

    #[test]
    fn test_interval_se_cluster_matches_inline_form() {
        let (var, k) = (4.0f64, 16usize);
        assert_eq!(se_cluster(var, k), var.sqrt() / (k as f64).sqrt());
    }

    #[test]
    fn test_interval_ci95_matches_inline_form() {
        let (m, se) = (0.5f64, 0.1f64);
        assert_eq!(ci95(m, se), (m - 1.96 * se, m + 1.96 * se));
    }

    #[test]
    fn test_interval_empty_and_singleton_are_nan_not_zero() {
        // No samples means no interval: returning 0 would read as a narrow interval.
        assert!(se_mean(1.0, 0).is_nan());
        assert!(se_cluster(1.0, 0).is_nan());
        assert!(sample_var(&[]).is_nan());
        assert!(sample_var(&[1.0]).is_nan());
        let (m, se, _) = mean_ci95(&[]);
        assert!(m.is_nan() && se.is_nan());
    }

    #[test]
    fn test_interval_sample_var_is_unbiased() {
        // Denominator n-1, matching the inline form.
        let v = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n;
        let want = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0);
        assert_eq!(sample_var(&v), want);
    }

    #[test]
    fn test_interval_lower_ci_above_chance_gate() {
        assert!(lower_ci_above(0.6, 0.01, 0.5));
        assert!(!lower_ci_above(0.51, 0.05, 0.5));
    }
}
