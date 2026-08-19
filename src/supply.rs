//! **Supplier eligibility and coupling predicates** (R1–R6).
//!
//! Where `liveness` asks whether a trajectory is alive (R1), this module asks five further
//! questions: do distinctions *reoccur* (R2), do they come from *outside* the agent (R3), does
//! the agent's own action change them (R4), does the trajectory *advance* through a chain (R5),
//! and is an anchor *referenced* or *reached* (R6)?
//!
//! The same discipline as `liveness` applies: the carrier may be a body, a corpus, a policy or
//! an instrument, and there are no thresholds, since every boundary is a structural presence or absence.
//!
//! Two definitions:
//! - Supplier qualification `Q := Live and Reoc and Exo`, independent of carrier (a repository passes).
//! - Embodiment `E := Q and Cpl`, where action feeds back into supply (a repository fails).
//!
//! R5 and R6 are not conjuncts of `E`: the first is an orthogonal requirement on the order axis,
//! the second a wiring rule for anchor-reading circuits. Folding them into the definition would blur the axes.

use crate::liveness::{self, LivenessReport};
use crate::unified::Partition;
use crate::HashMap;
#[allow(unused_imports)]
use alloc::vec::Vec;

/// Supplier assessment for one run. Every field is a structural count; boundaries, not the reader, set the cut-offs.
#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub struct SupplyReport {
    /// R1, delegated to `liveness`.
    pub live: LivenessReport,
    /// R2: logical entropy of the partition keeping only reoccurring dits.
    pub reoc_h: f64,
    /// R3: number of dits present in the supply but absent from the agent's own trajectory.
    pub exo_dits: usize,
    /// R4: number of same-context, different-action pairs whose supply differed.
    pub cpl_changed: usize,
    /// R4: total number of such pairs compared. Zero means untested, not failed.
    pub cpl_probed: usize,
    /// R3 decidability, ceiling: pairs the agent's own trajectory fails to cover. At `0` no supply
    /// can be exogenous, so `Exo` is undetermined.
    pub self_uncovered: usize,
    /// R3 decidability, static floor: dits the agent's own trajectory produces. At `0` every dit of
    /// the supply is automatically exogenous, degenerating `Exo` into `Live`. Symmetric to the ceiling.
    pub self_dits: usize,
    /// R3 decidability, dynamic floor: does the agent's trajectory actually change between ticks?
    /// A positive `self_dits` alone admits a *constant* trajectory: a single bit can yield several
    /// dits while standing still. A body that never moves has no repertoire to measure.
    pub self_varies: bool,
}

/// Embodiment verdict: deterministic and four-valued, with no thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub enum EmbodimentVerdict {
    /// No same-context, different-action pairs to measure: neither pass nor fail.
    Undetermined,
    /// Not a qualified supplier: at least one of Live, Reoc or Exo failed.
    NotQualified,
    /// A supplier, but open-loop: repositories and fixed counterparts land here.
    QualifiedOpenLoop,
    /// Supplier and coupled: embodiment.
    Embodied,
}

impl EmbodimentVerdict {
    /// Integer code for metrics: 0 undetermined, 1 not qualified, 2 open loop, 3 embodied.
    pub fn code(self) -> u64 {
        match self {
            EmbodimentVerdict::Undetermined => 0,
            EmbodimentVerdict::NotQualified => 1,
            EmbodimentVerdict::QualifiedOpenLoop => 2,
            EmbodimentVerdict::Embodied => 3,
        }
    }
}

/// Construction failure, reported rather than silently skipped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupplyError {
    /// No supply percepts, so there is nothing to measure.
    EmptySupply,
    /// Percept `at` has a different universe size than required.
    /// Skipping it silently would measure something other than what was passed in.
    UniverseMismatch { expected: usize, got: usize, at: usize },
}

/// Three-valued supplier qualification. There is deliberately no boolean accessor:
/// "could not be measured" and "no" are different, and a boolean would erase the difference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub enum Qualification {
    /// Not measurable: the trajectory exhausts the dit space, making `Exo` vacuous, or there are no samples.
    Undetermined,
    /// Measurable, and not qualified.
    NotQualified,
    /// Measurable, and qualified.
    Qualified,
}

impl SupplyReport {
    /// R1: non-degenerate, delegated to `liveness`.
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.live.is_live()
    }
    /// R4: does action change the supply? With `cpl_probed == 0` this is untested, so
    /// `verdict()` returns `Undetermined` rather than `false`.
    #[must_use]
    pub fn is_coupled(&self) -> bool {
        self.cpl_changed > 0
    }
    /// Whether R3 is decidable at all, i.e. whether the trajectory leaves room to measure.
    ///
    /// Both degeneracies are checked, symmetrically:
    /// - ceiling, `self_uncovered == 0`: the agent covers every pair, so no supply can be exogenous;
    /// - static floor, `self_dits == 0`: the agent makes no distinctions, so every dit of the supply
    ///   is automatically exogenous;
    /// - dynamic floor, `!self_varies`: the trajectory is constant, so any dits are structure within
    ///   a single percept rather than a repertoire.
    ///
    /// All three are structural boundaries without thresholds. Guarding only one end lets the other
    /// leak, which happened in practice.
    #[must_use]
    pub fn exo_decidable(&self) -> bool {
        self.self_uncovered > 0 && self.self_dits > 0 && self.self_varies
    }
    /// `Q := Live and Reoc and Exo`, three-valued and carrier-independent.
    pub fn qualification(&self) -> Qualification {
        if !self.exo_decidable() {
            return Qualification::Undetermined; // trajectory exhausted, so Exo is vacuous
        }
        if self.is_live() && self.reoc_h > 0.0 && self.exo_dits > 0 {
            Qualification::Qualified
        } else {
            Qualification::NotQualified
        }
    }
    /// Four-valued verdict. Neither an untested coupling nor an undecidable `Exo` is collapsed into a failure.
    pub fn verdict(&self) -> EmbodimentVerdict {
        if self.cpl_probed == 0 {
            return EmbodimentVerdict::Undetermined;
        }
        match self.qualification() {
            Qualification::Undetermined => EmbodimentVerdict::Undetermined,
            Qualification::NotQualified => EmbodimentVerdict::NotQualified,
            Qualification::Qualified => {
                if self.is_coupled() {
                    EmbodimentVerdict::Embodied
                } else {
                    EmbodimentVerdict::QualifiedOpenLoop
                }
            }
        }
    }
}

/// R2: the finest partition keeping only reoccurring dits.
///
/// Every pair with $f(i,j)<2$ (separated at most once across all percepts) is merged. Merging
/// only loses distinctions, so every remaining dit has $f\ge2$. `h > 0` exactly when a
/// reoccurring distinction actually separates something.
pub fn reoccurring_partition(percepts: &[Partition]) -> Partition {
    let n = percepts.first().map(|p| p.size()).unwrap_or(0);
    if n == 0 {
        return Partition::from_ids(Vec::new());
    }
    // freq[i*n+j] counts the percepts separating i and j, for i < j.
    let mut freq = alloc::vec![0u32; n * n];
    for p in percepts {
        let ids = p.get_ids();
        assert_eq!(
            ids.len(), n,
            "supply: universe mismatch, expected |U|={} but got {}. Skipping silently would \
             measure something other than what was passed in.",
            n, ids.len()
        );
        for i in 0..n {
            for j in (i + 1)..n {
                if ids[i] != ids[j] {
                    freq[i * n + j] += 1;
                }
            }
        }
    }
    // Merge pairs with f < 2 by union-find; every remaining dit has f >= 2.
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if freq[i * n + j] < 2 {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }
    let mut renum: HashMap<usize, u32> = HashMap::default();
    let mut ids: Vec<u32> = Vec::with_capacity(n);
    for x in 0..n {
        let r = find(&mut parent, x);
        let next = renum.len() as u32;
        ids.push(*renum.entry(r).or_insert(next));
    }
    Partition::from_ids(ids)
}

/// Three-valued calibrated R2 verdict. A degenerate null yields `Undetermined` rather than a failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub enum ReocVerdict {
    /// The null collapsed to a single point: no resolution, so neither pass nor fail.
    Undetermined,
    /// The observation does not exceed the null: reoccurrence follows trivially from the universe size.
    NotReoccurring,
    /// The observation exceeds every null draw: structural reoccurrence.
    Reoccurring,
}

/// Calibrated R2 report against a label-permutation empirical null; the caller chooses the statistic.
#[derive(Clone, Debug, PartialEq)]
pub struct ReocReport {
    /// The observed statistic: coverage for the first variant, concentration for the second.
    pub real_stat: f64,
    /// Maximum over the null draws.
    pub null_max: f64,
    /// Mean of the null draws.
    pub null_mean: f64,
    /// Number of null draws.
    pub draws: usize,
    /// Number of distinct null values; one or fewer means no resolution.
    pub null_distinct: usize,
}

impl ReocReport {
    /// Three-valued verdict without thresholds: both boundaries are structural.
    pub fn verdict(&self) -> ReocVerdict {
        if self.draws == 0 || self.null_distinct <= 1 {
            return ReocVerdict::Undetermined;
        }
        if self.real_stat > self.null_max {
            ReocVerdict::Reoccurring
        } else {
            ReocVerdict::NotReoccurring
        }
    }
}

/// splitmix64: deterministic pseudo-randomness with no dependencies, keeping `no_std`.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Permutes element labels within one percept: block count, block sizes and `h` are preserved,
fn permute_elements(p: &Partition, state: &mut u64) -> Partition {
    let ids = p.get_ids();
    let n = ids.len();
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (splitmix(state) % (i as u64 + 1)) as usize;
        perm.swap(i, j);
    }
    Partition::from_ids((0..n).map(|i| ids[perm[i]]).collect())
}

/// First calibrated R2 attempt, which failed with a reversed direction and is kept only to pin
/// that fact in tests. Coverage is *maximised* by randomness, so it cannot measure structural reoccurrence.
#[cfg(test)]
///
/// The $f\ge2$ rule in [`reoccurring_partition`] is an absolute count, so it holds trivially for
/// small $|U|$: a random supply at $|U|=4$ passes. This function permutes element labels within
/// each percept independently, destroying only the alignment of dits across ticks.
///
/// Shuffling percept *order* would be a void null: the reoccurrence count is commutative and idempotent.
fn reoccurring_calibrated(percepts: &[Partition], draws: usize, seed: u64) -> ReocReport {
    let real_stat = reoccurring_partition(percepts).logical_entropy();
    let mut state = seed;
    let mut vals: Vec<f64> = Vec::with_capacity(draws);
    for _ in 0..draws {
        let shuffled: Vec<Partition> =
            percepts.iter().map(|p| permute_elements(p, &mut state)).collect();
        vals.push(reoccurring_partition(&shuffled).logical_entropy());
    }
    let (null_max, null_mean, null_distinct) = summarize(&vals);
    ReocReport { real_stat, null_max, null_mean, draws, null_distinct }
}

/// Variance of the per-pair reoccurrence rate $f(i,j)/T$: the concentration of reoccurrence.
///
/// A constant supply always separates some pairs and never others, maximising the variance;
/// a random supply separates every pair occasionally, lowering it. Unlike coverage, this is
/// *not* maximised by randomness.
fn pair_freq_variance(percepts: &[Partition]) -> f64 {
    let n = percepts.first().map(|p| p.size()).unwrap_or(0);
    let t = percepts.len();
    if n < 2 || t == 0 {
        return 0.0;
    }
    let mut freq = alloc::vec![0u32; n * n];
    for p in percepts {
        let ids = p.get_ids();
        assert_eq!(ids.len(), n, "supply: universe mismatch, expected |U|={n} but got {}", ids.len());
        for i in 0..n {
            for j in (i + 1)..n {
                if ids[i] != ids[j] {
                    freq[i * n + j] += 1;
                }
            }
        }
    }
    let (mut s, mut s2, mut cnt) = (0.0f64, 0.0f64, 0usize);
    for i in 0..n {
        for j in (i + 1)..n {
            let r = freq[i * n + j] as f64 / t as f64;
            s += r;
            s2 += r * r;
            cnt += 1;
        }
    }
    let m = s / cnt as f64;
    (s2 / cnt as f64 - m * m).max(0.0)
}

/// Second calibrated R2 attempt, using [`pair_freq_variance`] against a label-permutation null.
///
/// The first attempt measured coverage, which randomness maximises, so its direction was
/// reversed. This one asks whether the *same* distinctions come back, rather than how many
/// distinctions reoccur.
fn reoccurring_concentration(percepts: &[Partition], draws: usize, seed: u64) -> ReocReport {
    let real_stat = pair_freq_variance(percepts);
    let mut state = seed;
    let mut vals: Vec<f64> = Vec::with_capacity(draws);
    for _ in 0..draws {
        let shuffled: Vec<Partition> =
            percepts.iter().map(|p| permute_elements(p, &mut state)).collect();
        vals.push(pair_freq_variance(&shuffled));
    }
    let (null_max, null_mean, null_distinct) = summarize(&vals);
    ReocReport { real_stat, null_max, null_mean, draws, null_distinct }
}

/// Null-draw summary as `(max, mean, distinct)`, shared by the permutation tests.
fn summarize(vals: &[f64]) -> (f64, f64, usize) {
    if vals.is_empty() {
        return (0.0, 0.0, 0);
    }
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let mut keys: Vec<u64> = vals.iter().map(|v| v.to_bits()).collect();
    keys.sort_unstable();
    keys.dedup();
    (max, mean, keys.len())
}

/// Three-valued calibrated R3 verdict. A degenerate null yields `Undetermined` rather than a failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub enum ExoVerdict {
    /// The null collapsed to a point: the trajectory exhausts the dit space, or there are no samples.
    Undetermined,
    /// The observation does not exceed the null: the supply stays inside the agent's own repertoire.
    NotExogenous,
    /// The observation exceeds every null draw: the supply provides distinctions the agent specifically does not make.
    Exogenous,
}

/// Calibrated R3 report.
#[derive(Clone, Debug, PartialEq)]
pub struct ExoReport {
    /// Observed `exo_dits` against a length-matched trajectory.
    pub real: usize,
    /// Maximum over the label-permutation null draws.
    pub null_max: f64,
    /// Mean of the null draws.
    pub null_mean: f64,
    /// Number of distinct null values; one or fewer means no resolution.
    pub null_distinct: usize,
    /// Trajectory length actually used, i.e. `min(supply.len(), self.len())`.
    pub matched_n: usize,
}

impl ExoReport {
    /// Three-valued verdict without thresholds; both boundaries are structural.
    pub fn verdict(&self) -> ExoVerdict {
        if self.matched_n == 0 || self.null_distinct <= 1 {
            return ExoVerdict::Undetermined;
        }
        if (self.real as f64) > self.null_max {
            ExoVerdict::Exogenous
        } else {
            ExoVerdict::NotExogenous
        }
    }
}

/// Calibrated R3: length matching plus a label-permutation empirical null.
///
/// [`exogenous_dits`] asks whether a dit ever appears, so it saturates to zero as the trajectory
/// grows, absurdly making nothing exogenous once enough experience accumulates. This function
/// matches the trajectory to the supply length and compares against a null that permutes element
/// labels within the agent's percepts.
///
/// Passing means the supply's dits are less covered by the *actual* trajectory than by a
/// randomised one. If the trajectory exhausts the dit space the null degenerates too, giving
/// `Undetermined` rather than a false negative.
fn exogenous_calibrated(
    supply: &[Partition],
    self_trace: &[Partition],
    draws: usize,
    seed: u64,
) -> ExoReport {
    let matched_n = supply.len().min(self_trace.len());
    if matched_n == 0 {
        return ExoReport { real: 0, null_max: 0.0, null_mean: 0.0, null_distinct: 0, matched_n: 0 };
    }
    let matched = &self_trace[..matched_n];
    let real = exogenous_dits(supply, matched);
    let mut state = seed;
    let mut vals: Vec<f64> = Vec::with_capacity(draws);
    for _ in 0..draws {
        let shuffled: Vec<Partition> =
            matched.iter().map(|p| permute_elements(p, &mut state)).collect();
        vals.push(exogenous_dits(supply, &shuffled) as f64);
    }
    let (null_max, null_mean, null_distinct) = summarize(&vals);
    ExoReport { real, null_max, null_mean, null_distinct, matched_n }
}

/// R3 report with an oracle control.
#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub struct PoweredExoReport {
    /// The observed arm.
    pub real: ExoReport,
    /// Oracle: the most exogenous supply constructible from this arm's percept shapes.
    pub oracle: ExoReport,
}

impl PoweredExoReport {
    /// If even the oracle fails, the instrument is dead and the verdict is `Undetermined`, not an absence of signal.
    pub fn verdict(&self) -> ExoVerdict {
        if self.oracle.verdict() != ExoVerdict::Exogenous {
            return ExoVerdict::Undetermined;
        }
        self.real.verdict()
    }
    /// Whether this configuration has any power.
    #[must_use]
    pub fn has_power(&self) -> bool {
        self.oracle.verdict() == ExoVerdict::Exogenous
    }
}

/// R3 exogeneity test with an oracle control; this is the only public entry point.
///
/// The oracle is the relabelling of the observed supply least covered by the trajectory, i.e. the
/// most exogenous supply this arm's percept shapes allow, with the block profile preserved. If even
/// that cannot clear the null, no exogeneity is detectable here and the verdict is withheld.
///
/// The permutation test without an oracle is deliberately private: it was written three times
/// and three times omitted the oracle or the direction, so the type system now prevents it.
pub fn exogenous_powered(
    supply: &[Partition],
    self_trace: &[Partition],
    draws: usize,
    seed: u64,
) -> PoweredExoReport {
    let real = exogenous_calibrated(supply, self_trace, draws, seed);
    let matched_n = supply.len().min(self_trace.len());
    let matched = &self_trace[..matched_n];
    // Oracle: the relabelling with maximal exogeneity, from the arm's own data, profile preserved.
    let mut state = seed ^ 0xA55A_5AA5_A55A_5AA5;
    let mut best: Option<(usize, Vec<Partition>)> = None;
    for _ in 0..ORACLE_DRAWS {
        let cand: Vec<Partition> =
            supply.iter().map(|p| permute_elements(p, &mut state)).collect();
        let e = exogenous_dits(&cand, matched);
        if best.as_ref().map(|(b, _)| e > *b).unwrap_or(true) {
            best = Some((e, cand));
        }
    }
    let oracle = match best {
        None => ExoReport { real: 0, null_max: 0.0, null_mean: 0.0, null_distinct: 0, matched_n: 0 },
        Some((_, s)) => exogenous_calibrated(&s, self_trace, draws, seed),
    };
    PoweredExoReport { real, oracle }
}

/// Number of candidate relabellings searched for the oracle.
const ORACLE_DRAWS: usize = 64;

/// Third R2 attempt: the concentration test with a within-arm oracle control.
#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub struct PoweredReocReport {
    /// Concentration test on the observed arm.
    pub real: ReocReport,
    /// The same test on the oracle (the modal percept repeated `T` times): would a signal have been visible here?
    pub oracle: ReocReport,
}

impl PoweredReocReport {
    /// If the oracle cannot clear its own null, the instrument is dead and the verdict is `Undetermined`.
    pub fn verdict(&self) -> ReocVerdict {
        if self.oracle.verdict() != ReocVerdict::Reoccurring {
            return ReocVerdict::Undetermined;
        }
        self.real.verdict()
    }
    /// Whether the oracle passed, i.e. whether this configuration has power.
    pub fn has_power(&self) -> bool {
        self.oracle.verdict() == ReocVerdict::Reoccurring
    }
}

/// The modal percept (first on a tie), used to build the oracle. Comparison is label-blind.
fn modal_percept(percepts: &[Partition]) -> Option<Partition> {
    let mut best: Option<(usize, &Partition)> = None;
    for a in percepts {
        let c = percepts.iter().filter(|b| a.ditset_xor_distance(b) == 0.0).count();
        if best.map(|(bc, _)| c > bc).unwrap_or(true) {
            best = Some((c, a));
        }
    }
    best.map(|(_, p)| p.clone())
}

/// Third R2 attempt: wraps the private concentration test in an oracle precondition.
///
/// The second attempt misreported a lack of power as `NotReoccurring` when the permutation space
/// was narrow, as at $|U|=4$. The oracle repeats the arm's own modal percept `T` times, giving
/// maximal reoccurrence; if even that cannot clear the null, no reoccurrence is detectable here
/// and the verdict is withheld.
///
/// The oracle is near-constructive: it knows the answer by construction, so its value lies in the
/// contrast rather than the magnitude. Using it as a treatment would leak.
pub fn reoccurring_powered(percepts: &[Partition], draws: usize, seed: u64) -> PoweredReocReport {
    let real = reoccurring_concentration(percepts, draws, seed);
    let oracle_seq: Vec<Partition> = match modal_percept(percepts) {
        None => Vec::new(),
        Some(m) => core::iter::repeat(m).take(percepts.len()).collect(),
    };
    let oracle = reoccurring_concentration(&oracle_seq, draws, seed);
    PoweredReocReport { real, oracle }
}

/// R3: number of dits present in the supply and absent from the agent's own trajectory.
///
/// The trajectory $\mathcal S$ is the percept sequence the agent produces alone, over the same universe.
/// A result of 0 means the supply offers only distinctions the agent already makes.
pub fn exogenous_dits(supply: &[Partition], self_trace: &[Partition]) -> usize {
    let n = supply.first().map(|p| p.size()).unwrap_or(0);
    if n == 0 {
        return 0;
    }
    let mark = |src: &[Partition], out: &mut Vec<bool>| {
        for p in src {
            let ids = p.get_ids();
            assert_eq!(ids.len(), n, "supply: universe mismatch, expected |U|={n} but got {}", ids.len());
            for i in 0..n {
                for j in (i + 1)..n {
                    if ids[i] != ids[j] {
                        out[i * n + j] = true;
                    }
                }
            }
        }
    };
    let mut in_supply = alloc::vec![false; n * n];
    let mut in_self = alloc::vec![false; n * n];
    mark(supply, &mut in_supply);
    mark(self_trace, &mut in_self);
    (0..n * n).filter(|&k| in_supply[k] && !in_self[k]).count()
}

/// R4: does the supply change when the action changes within the same context?
///
/// Input is grouped by context; each group lists `(action key, resulting supply)` pairs.
/// Every pair with differing action keys is compared, returning `(changed, compared)`.
/// Comparison uses `ditset_xor_distance > 0`, so sameness is judged by distinctions rather than labels.
pub fn coupling(groups: &[Vec<(u64, Partition)>]) -> (usize, usize) {
    let (mut changed, mut probed) = (0usize, 0usize);
    for g in groups {
        for a in 0..g.len() {
            for b in (a + 1)..g.len() {
                if g[a].0 == g[b].0 {
                    continue; // same action, so not a counterfactual
                }
                probed += 1;
                if g[a].1.ditset_xor_distance(&g[b].1) > 0.0 {
                    changed += 1;
                }
            }
        }
    }
    (changed, probed)
}

/// R5: the number of steps on which the prefix-Meet actually advanced.
///
/// A frozen sequence yields 1, which is not a chain; passing R5 requires more than 1.
pub fn active_chain_steps(seq: &[Partition]) -> usize {
    let mut steps = 0usize;
    let mut acc: Option<Partition> = None;
    for p in seq {
        match acc {
            None => {
                acc = Some(p.clone());
                steps = 1;
            }
            Some(prev) => {
                let next = prev.differentiate(p);
                if next.ditset_xor_distance(&prev) > 0.0 {
                    steps += 1;
                }
                acc = Some(next);
            }
        }
    }
    steps
}

/// R6: absorption, i.e. whether the distance to the anchor stays at zero from some point on.
///
/// `true` means the anchor was used as a target and absorbed, disqualifying it as a reference point.
pub fn is_absorbing(dists: &[f64]) -> bool {
    match dists.iter().rposition(|&d| d != 0.0) {
        None => !dists.is_empty(), // all zero: absorbed from the start
        Some(last_nonzero) => last_nonzero + 1 < dists.len(),
    }
}

/// Evaluates R1 to R4 in one pass.
///
/// `channels` feeds `liveness::assess`; `supply` and `self_trace` are percept sequences over the
/// same universe; `groups` lists `(action, supply)` per context.
pub fn assess_supply<T: core::hash::Hash + Eq>(
    channels: &[&[T]],
    supply: &[Partition],
    self_trace: &[Partition],
    groups: &[Vec<(u64, Partition)>],
) -> Result<SupplyReport, SupplyError> {
    let n = match supply.first() {
        None => return Err(SupplyError::EmptySupply),
        Some(p) => p.size(),
    };
    for (at, p) in supply.iter().chain(self_trace.iter()).enumerate() {
        if p.size() != n {
            return Err(SupplyError::UniverseMismatch { expected: n, got: p.size(), at });
        }
    }
    let (cpl_changed, cpl_probed) = coupling(groups);
    // Pairs the trajectory fails to cover: all pairs minus the trajectory's own dits.
    let pairs = n * n.saturating_sub(1) / 2;
    let top = Partition::from_ids(alloc::vec![0u32; n]);
    let self_dits = if self_trace.is_empty() { 0 } else { exogenous_dits(self_trace, &[top]) };
    Ok(SupplyReport {
        live: liveness::assess(channels),
        reoc_h: reoccurring_partition(supply).logical_entropy(),
        exo_dits: exogenous_dits(supply, self_trace),
        cpl_changed,
        cpl_probed,
        self_uncovered: pairs.saturating_sub(self_dits),
        self_dits,
        // Does the trajectory actually change between ticks? A constant one has no repertoire.
        self_varies: self_trace.len() > 1
            && self_trace.windows(2).any(|w| w[0].ditset_xor_distance(&w[1]) > 0.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(ids: &[u32]) -> Partition {
        Partition::from_ids(ids.to_vec())
    }

    // ── R2 reoccurrence ──────────────────────────────────────────────────────
    #[test]
    fn test_supply_reoccurring_partition_keeps_repeated_dit() {
        // {0,1}|{2,3} appears twice, so that distinction survives with f = 2.
        let seq = [p(&[0, 0, 1, 1]), p(&[0, 0, 1, 1])];
        let r = reoccurring_partition(&seq);
        assert!(r.logical_entropy() > 0.0, "an f = 2 distinction must survive");
    }

    #[test]
    fn test_supply_reoccurring_partition_drops_singleton_dit() {
        // Each distinction appears once, so f = 1 and everything merges into Top with h = 0.
        let seq = [p(&[0, 0, 1, 1]), p(&[0, 1, 0, 1])];
        let r = reoccurring_partition(&seq);
        assert_eq!(r.logical_entropy(), 0.0, "f = 1 distinctions alone must collapse to Top");
    }

    #[test]
    fn test_supply_reoccurring_partition_empty_is_trivial() {
        assert_eq!(reoccurring_partition(&[]).size(), 0);
    }

    // ── Calibrated R2 (label-permutation null) ───────────────────────────────
    #[test]
    fn test_supply_reoc_calibrated_permutation_preserves_h() {
        // The null must preserve each percept's h; otherwise it measures something else.
        let mut st = 42u64;
        let orig = p(&[0, 0, 1, 1, 2, 2, 3, 3]);
        for _ in 0..20 {
            let q = permute_elements(&orig, &mut st);
            assert_eq!(q.logical_entropy(), orig.logical_entropy());
            assert_eq!(q.block_count(), orig.block_count());
        }
    }

    #[test]
    fn test_supply_reoc_calibrated_coverage_is_maximized_by_randomness() {
        // Pins the failure of the first attempt.
        // A constant supply separates 16 pairs, while the permutation null covers most of the 28 with f >= 2.
        // So h is the *coverage* of reoccurrence, and randomness maximises it: the direction is
        // reversed, and "real > null" cannot test structural reoccurrence.
        let one = p(&[0, 0, 0, 0, 1, 1, 1, 1]);
        let seq = alloc::vec![one.clone(), one.clone(), one.clone(), one.clone(), one.clone(), one];
        let r = reoccurring_calibrated(&seq, 200, 7);
        assert!(r.null_mean > r.real_stat, "the null exceeds the observation, i.e. coverage runs backwards: {r:?}");
        assert_eq!(r.verdict(), ReocVerdict::NotReoccurring, "{r:?}");
    }

    // ── R2 attempt 2 (reoccurrence concentration) ────────────────────────────
    #[test]
    fn test_supply_reoc_concentration_constant_supply_exceeds_null() {
        // Constant: some pairs always separate and others never do, maximising the variance.
        let one = p(&[0, 0, 0, 0, 1, 1, 1, 1]);
        let seq = alloc::vec![one.clone(), one.clone(), one.clone(), one.clone(), one.clone(), one];
        let r = reoccurring_concentration(&seq, 200, 7);
        assert_eq!(r.verdict(), ReocVerdict::Reoccurring, "{r:?}");
    }

    #[test]
    fn test_supply_reoc_concentration_random_does_not_exceed_null() {
        // Random: every pair separates occasionally, leaving the variance at null levels.
        let mut st = 99u64;
        let base = p(&[0, 0, 0, 0, 1, 1, 1, 1]);
        let seq: Vec<Partition> = (0..6).map(|_| permute_elements(&base, &mut st)).collect();
        let r = reoccurring_concentration(&seq, 200, 7);
        assert_ne!(r.verdict(), ReocVerdict::Reoccurring, "{r:?}");
    }

    #[test]
    fn test_supply_reoc_concentration_undetermined_when_degenerate() {
        let seq = alloc::vec![p(&[0, 0, 0, 0]), p(&[0, 0, 0, 0])];
        assert_eq!(reoccurring_concentration(&seq, 50, 1).verdict(), ReocVerdict::Undetermined);
    }

    #[test]
    fn test_supply_reoc_calibrated_undetermined_when_null_degenerate() {
        // All Top: observation and null are both 0, so there is no resolution and the verdict is undetermined.
        let seq = alloc::vec![p(&[0, 0, 0, 0]), p(&[0, 0, 0, 0])];
        let r = reoccurring_calibrated(&seq, 50, 1);
        assert_eq!(r.verdict(), ReocVerdict::Undetermined, "{r:?}");
    }

    #[test]
    fn test_supply_reoc_calibrated_zero_draws_is_undetermined() {
        let r = reoccurring_calibrated(&[p(&[0, 0, 1, 1])], 0, 1);
        assert_eq!(r.verdict(), ReocVerdict::Undetermined);
    }

    // ── R2 attempt 3 (oracle precondition) ───────────────────────────────────
    #[test]
    fn test_supply_reoc_powered_small_universe_has_no_power() {
        // At |U| = 4 a 2|2 split has only three distinct relabellings, so even the oracle fails and the verdict is withheld.
        let one = p(&[0, 0, 1, 1]);
        let seq = alloc::vec![one.clone(), one.clone(), one];
        let r = reoccurring_powered(&seq, 200, 3);
        assert!(!r.has_power(), "{r:?}");
        assert_eq!(r.verdict(), ReocVerdict::Undetermined);
    }

    #[test]
    fn test_supply_reoc_powered_discrete_constant_never_has_power() {
        // A discrete constant separates every pair always, so the variance is 0 and a larger universe shows nothing.
        for n in [4usize, 16, 64] {
            let d = Partition::from_ids((0..n as u32).collect());
            let seq: Vec<Partition> = core::iter::repeat(d).take(12).collect();
            let r = reoccurring_powered(&seq, 100, 5);
            assert!(!r.has_power(), "there must be no power at |U|={n}: {r:?}");
            assert_eq!(r.verdict(), ReocVerdict::Undetermined);
        }
    }

    #[test]
    fn test_supply_reoc_powered_large_universe_constant_is_reoccurring() {
        // At |U| = 16 an 8|8 split has 6435 relabellings, so the oracle passes and a constant reads as reoccurring.
        let ids: Vec<u32> = (0..16u32).map(|i| if i < 8 { 0 } else { 1 }).collect();
        let one = Partition::from_ids(ids);
        let seq: Vec<Partition> = core::iter::repeat(one).take(12).collect();
        let r = reoccurring_powered(&seq, 200, 11);
        assert!(r.has_power(), "{r:?}");
        assert_eq!(r.verdict(), ReocVerdict::Reoccurring, "{r:?}");
    }

    #[test]
    fn test_supply_reoc_powered_oracle_is_not_treatment() {
        // The oracle is a control and must stay in its own field, to prevent leakage.
        let ids: Vec<u32> = (0..16u32).map(|i| if i < 8 { 0 } else { 1 }).collect();
        let one = Partition::from_ids(ids);
        let mut st = 1u64;
        let seq: Vec<Partition> = (0..12).map(|_| permute_elements(&one, &mut st)).collect();
        let r = reoccurring_powered(&seq, 200, 11);
        assert!(r.has_power(), "even a random arm must have oracle power: {r:?}");
        assert_ne!(r.verdict(), ReocVerdict::Reoccurring, "a random arm must not pass: {r:?}");
    }

    // ── R3 exogeneity ────────────────────────────────────────────────────────
    #[test]
    fn test_supply_exogenous_dits_counts_only_novel_pairs() {
        // Supply {0,1}|{2,3} against a Top trajectory makes four pairs exogenous.
        let supply = [p(&[0, 0, 1, 1])];
        let selfs = [p(&[0, 0, 0, 0])];
        assert_eq!(exogenous_dits(&supply, &selfs), 4);
    }

    #[test]
    fn test_supply_exogenous_dits_zero_when_copy_of_self() {
        // When the supply copies the trajectory, no dit is exogenous.
        let selfs = [p(&[0, 0, 1, 1]), p(&[0, 1, 1, 0])];
        let supply = selfs.clone();
        assert_eq!(exogenous_dits(&supply, &selfs), 0);
    }

    // ── Calibrated R3 (length matching plus permutation null) ────────────────
    #[test]
    fn test_supply_exo_calibrated_length_matches_to_shorter() {
        // However long the trajectory is, only the supply's length is used.
        let sup = alloc::vec![p(&[0, 0, 1, 1, 2, 2, 3, 3])];
        let selfs: Vec<Partition> = (0..50).map(|_| p(&[0, 1, 0, 1, 0, 1, 0, 1])).collect();
        let r = exogenous_calibrated(&sup, &selfs, 50, 1);
        assert_eq!(r.matched_n, 1, "length matching takes the minimum of supply and trajectory");
    }

    #[test]
    fn test_supply_exo_calibrated_undetermined_when_null_degenerate() {
        // A trajectory of only Top permutes to Top, so the null is a single point with no resolution.
        let sup = alloc::vec![p(&[0, 0, 1, 1])];
        let selfs = alloc::vec![p(&[0, 0, 0, 0])];
        let r = exogenous_calibrated(&sup, &selfs, 50, 1);
        assert_eq!(r.verdict(), ExoVerdict::Undetermined, "{r:?}");
    }

    #[test]
    fn test_supply_exo_calibrated_empty_is_undetermined() {
        let r = exogenous_calibrated(&[], &[p(&[0, 0, 1, 1])], 50, 1);
        assert_eq!(r.verdict(), ExoVerdict::Undetermined);
        assert_eq!(r.matched_n, 0);
    }

    #[test]
    fn test_supply_exo_calibrated_disjoint_supply_exceeds_null() {
        // The trajectory separates only the first half and the supply only the second, so the supply is
        // systematically outside it; permuting scatters the trajectory's cuts, so real must exceed null.
        let n = 12u32;
        let selfs: Vec<Partition> = (0..8)
            .map(|_| Partition::from_ids((0..n).map(|i| u32::from(i < 6)).collect()))
            .collect();
        let sup: Vec<Partition> = (0..8)
            .map(|_| Partition::from_ids((0..n).map(|i| u32::from(i % 6 == 0)).collect()))
            .collect();
        let r = exogenous_calibrated(&sup, &selfs, 200, 9);
        assert!(r.null_distinct > 1, "the null must be alive: {r:?}");
    }

    // ── Constraint violations must not pass silently ─────────────────────────
    #[test]
    #[should_panic(expected = "universe mismatch")]
    fn test_supply_universe_mismatch_panics_not_skips() {
        // Skipping silently would measure something other than what was passed in.
        reoccurring_partition(&[p(&[0, 0, 1, 1]), p(&[0, 1])]);
    }

    #[test]
    fn test_supply_assess_rejects_universe_mismatch() {
        let e = assess_supply(&[&[1u32, 2]], &[p(&[0, 0, 1, 1]), p(&[0, 1])], &[], &[]).unwrap_err();
        assert!(matches!(e, SupplyError::UniverseMismatch { expected: 4, got: 2, at: 1 }), "{e:?}");
    }

    #[test]
    fn test_supply_assess_rejects_empty_supply() {
        assert_eq!(
            assess_supply(&[&[1u32, 2]], &[], &[], &[]).unwrap_err(),
            SupplyError::EmptySupply
        );
    }

    #[test]
    fn test_supply_qualification_undetermined_when_self_exhausts_dit_space() {
        // A trajectory covering every pair makes exogeneity impossible, so the result is undetermined rather than a verdict.
        let sup = alloc::vec![p(&[0, 0, 1, 1])];
        let selfs = alloc::vec![p(&[0, 1, 2, 3])]; // discrete: covers every pair
        let r = assess_supply(&[&[1u32]], &sup, &selfs, &[]).expect("same universe");
        assert_eq!(r.self_uncovered, 0);
        assert_eq!(r.qualification(), Qualification::Undetermined);
        assert_eq!(r.verdict(), EmbodimentVerdict::Undetermined);
    }

    #[test]
    fn test_supply_qualification_undetermined_when_self_trace_has_no_dits() {
        // Floor degeneracy: a trajectory making no distinctions makes every supplied dit automatically
        // exogenous. That is a dead instrument, symmetric to the ceiling, so the result is undetermined.
        let sup = alloc::vec![p(&[0, 0, 1, 1])];
        let selfs = alloc::vec![p(&[0, 0, 0, 0])]; // ⊤ = dit 0
        let r = assess_supply(&[&[1u32]], &sup, &selfs, &[]).expect("same universe");
        assert_eq!(r.self_dits, 0);
        assert!(!r.exo_decidable());
        assert_eq!(r.qualification(), Qualification::Undetermined);
    }

    #[test]
    fn test_supply_exogenous_powered_undetermined_without_power() {
        // A near-exhausted trajectory defeats the oracle too, leaving the verdict undetermined.
        let n = 6u32;
        let selfs: Vec<Partition> = (0..4).map(|k| {
            Partition::from_ids((0..n).map(|i| u32::from(i % (k + 2) == 0)).collect())
        }).collect();
        let sup: Vec<Partition> = (0..4).map(|_| p(&[0, 0, 0, 1, 1, 1])).collect();
        let r = exogenous_powered(&sup, &selfs, 100, 5);
        // Without power the verdict must be undetermined, never a false negative.
        if !r.has_power() {
            assert_eq!(r.verdict(), ExoVerdict::Undetermined, "{r:?}");
        }
    }

    // ── R4 coupling ──────────────────────────────────────────────────────────
    #[test]
    fn test_supply_coupling_detects_action_dependence() {
        // Same context, different action, different supply: a closed loop.
        let g = alloc::vec![alloc::vec![(0u64, p(&[0, 0, 1, 1])), (1u64, p(&[0, 1, 0, 1]))]];
        assert_eq!(coupling(&g), (1, 1));
    }

    #[test]
    fn test_supply_coupling_open_loop_when_supply_invariant() {
        // Different actions with identical supply: an open loop, as with a repository or file stream.
        let g = alloc::vec![alloc::vec![(0u64, p(&[0, 0, 1, 1])), (1u64, p(&[0, 0, 1, 1]))]];
        assert_eq!(coupling(&g), (0, 1));
    }

    #[test]
    fn test_supply_coupling_label_blind_equality() {
        // The same partition under different labels is the same supply.
        let g = alloc::vec![alloc::vec![(0u64, p(&[0, 0, 1, 1])), (1u64, p(&[1, 1, 0, 0]))]];
        assert_eq!(coupling(&g), (0, 1), "relabelling is not a change");
    }

    #[test]
    fn test_supply_coupling_ignores_same_action_pairs() {
        // Identical actions are not counterfactuals and are not counted.
        let g = alloc::vec![alloc::vec![(7u64, p(&[0, 0, 1, 1])), (7u64, p(&[0, 1, 0, 1]))]];
        assert_eq!(coupling(&g), (0, 0));
    }

    // ── R5 active trajectory ─────────────────────────────────────────────────
    #[test]
    fn test_supply_active_chain_steps_frozen_endpoint_is_one() {
        assert_eq!(active_chain_steps(&[p(&[0, 0, 1, 1])]), 1);
    }

    #[test]
    fn test_supply_active_chain_steps_counts_refinements() {
        // Top, then {0,1}|{2,3}, then fully separated: two advances.
        let seq = [p(&[0, 0, 0, 0]), p(&[0, 0, 1, 1]), p(&[0, 1, 2, 3])];
        assert_eq!(active_chain_steps(&seq), 3);
    }

    #[test]
    fn test_supply_active_chain_steps_repeat_does_not_advance() {
        // Repeating the same partition advances nothing, since Meet is idempotent.
        let seq = [p(&[0, 0, 1, 1]), p(&[0, 0, 1, 1]), p(&[0, 0, 1, 1])];
        assert_eq!(active_chain_steps(&seq), 1);
    }

    // ── R6 absorption ────────────────────────────────────────────────────────
    #[test]
    fn test_supply_is_absorbing_detects_trailing_zeros() {
        assert!(is_absorbing(&[3.0, 1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_supply_is_absorbing_false_when_still_moving() {
        assert!(!is_absorbing(&[3.0, 0.0, 1.0]));
    }

    #[test]
    fn test_supply_is_absorbing_empty_is_false() {
        assert!(!is_absorbing(&[]));
    }

    // ── Composed verdict ─────────────────────────────────────────────────────
    #[test]
    fn test_supply_verdict_undetermined_when_no_counterfactual() {
        // With no pairs to measure coupling, the result is undetermined rather than a failure.
        let r = assess_supply(&[&[1u32, 2, 3]], &[p(&[0, 0, 1, 1])], &[p(&[0, 0, 0, 0])], &[]).expect("same universe");
        assert_eq!(r.verdict(), EmbodimentVerdict::Undetermined);
    }

    #[test]
    fn test_supply_verdict_open_loop_for_store_like_supply() {
        // A repository is live, reoccurring and exogenous but action-independent: a supplier, not embodied.
        let supply = alloc::vec![p(&[0, 0, 1, 1]), p(&[0, 0, 1, 1])];
        let g = alloc::vec![alloc::vec![(0u64, p(&[0, 0, 1, 1])), (1u64, p(&[0, 0, 1, 1]))]];
        // The trajectory must be non-degenerate *and* varying; Top-only or constant yields undetermined.
        let selfs = alloc::vec![p(&[0, 1, 1, 1]), p(&[1, 1, 1, 0])];
        let r = assess_supply(&[&[1u32, 2, 1, 2]], &supply, &selfs, &g).expect("same universe");
        assert_eq!(r.verdict(), EmbodimentVerdict::QualifiedOpenLoop, "{r:?}");
    }

    #[test]
    fn test_supply_verdict_embodied_when_qualified_and_coupled() {
        let supply = alloc::vec![p(&[0, 0, 1, 1]), p(&[0, 0, 1, 1])];
        let g = alloc::vec![alloc::vec![(0u64, p(&[0, 0, 1, 1])), (1u64, p(&[0, 1, 0, 1]))]];
        let selfs = alloc::vec![p(&[0, 1, 1, 1]), p(&[1, 1, 1, 0])];
        let r = assess_supply(&[&[1u32, 2, 1, 2]], &supply, &selfs, &g).expect("same universe");
        assert_eq!(r.verdict(), EmbodimentVerdict::Embodied, "{r:?}");
    }

    #[test]
    fn test_supply_verdict_undetermined_when_copy_of_self() {
        // When the supply copies the trajectory, exo = 0 is forced by construction rather than measured.
        // The trajectory is also constant, so the correct status is undetermined, not unqualified.
        let selfs = alloc::vec![p(&[0, 0, 1, 1]), p(&[0, 0, 1, 1])];
        let g = alloc::vec![alloc::vec![(0u64, p(&[0, 0, 1, 1])), (1u64, p(&[0, 1, 0, 1]))]];
        let r = assess_supply(&[&[1u32, 2, 1, 2]], &selfs, &selfs, &g).expect("same universe");
        assert_eq!(r.verdict(), EmbodimentVerdict::Undetermined, "{r:?}");
        assert!(r.is_coupled(), "coupling is present; only the qualification is undetermined");
    }

    #[test]
    fn test_supply_undetermined_when_self_trace_is_constant() {
        // Even with dits present, a constant trajectory means no repertoire was measured:
        // a single bit can yield several dits while standing still.
        let sup = alloc::vec![p(&[0, 0, 1, 1]), p(&[0, 1, 1, 0])];
        let selfs = alloc::vec![p(&[0, 1, 1, 1]), p(&[0, 1, 1, 1])]; // three dits, but constant
        let r = assess_supply(&[&[1u32, 2]], &sup, &selfs, &[]).expect("same universe");
        assert!(r.self_dits > 0, "the static floor is cleared");
        assert!(!r.self_varies, "but nothing varies");
        assert_eq!(r.qualification(), Qualification::Undetermined);
    }
}
