//! Arm machinery for PCC dynamics.
//!
//! Holds the arm decomposition (gold, random-replace, no-meet-refine, random-walk, anti-PCC)
//! together with the trajectory instrumentation, so the same dynamics is implemented once.
//!
//! It was lifted out of the measurement binaries because running the same arms over different
//! inputs (random partitions versus real data) otherwise forces a re-implementation.
//!
//! The move was behaviour-preserving: the full output of the original measurement binary was
//! byte-identical before and after, checked by hash.

use crate::curvature::{discrete_curvature, dsigma_moments};
use crate::Partition;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

// ── Mechanism parameters (part of the machinery) ─────────────────────────────
//    Decision thresholds (seed counts, universe sizes, effect sizes) deliberately live with
//    the pre-registered hypothesis instead of here.
pub const CRYSTAL_SIZE: usize = 8;
pub const ITERATIONS: usize = 50;
pub const EPS: f64 = 1e-12; // delta-h gate threshold


#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arm {
    Gold,          // A1: the PCC cycle itself: delta-h gate plus h-min replacement
    RandomReplace, // A2: same Meet candidates, but the replaced slot is random
    NoMeetRefine,  // A3: pure block splitting, no Meet
    RandomWalk,    // A4: replacement by a random partition, i.e. no refinement pressure
    AntiPcc,       // A5: sign-flipped: h-max replacement via Join (coarsening)
}

impl Arm {
    pub fn label(self) -> &'static str {
        match self {
            Arm::Gold => "A1 GOLD",
            Arm::RandomReplace => "A2 RAND-REPL",
            Arm::NoMeetRefine => "A3 NO-MEET-REF",
            Arm::RandomWalk => "A4 RAND-WALK",
            Arm::AntiPcc => "A5 ANTI-PCC",
        }
    }
    pub fn all() -> [Arm; 5] {
        [
            Arm::Gold,
            Arm::RandomReplace,
            Arm::NoMeetRefine,
            Arm::RandomWalk,
            Arm::AntiPcc,
        ]
    }
}

// ── Partition generation ───────────────────────────────────────────────────
/// Exhaustive enumeration for |U| <= 6.
pub fn all_partitions(n: usize) -> Vec<Partition> {
    let mut ids = vec![0u32; n];
    let mut out: Vec<Vec<u32>> = Vec::new();
    fn rec(i: usize, n: usize, max_block: u32, ids: &mut [u32], out: &mut Vec<Vec<u32>>) {
        if i == n {
            out.push(ids.to_vec());
            return;
        }
        for b in 0..=max_block + 1 {
            ids[i] = b;
            rec(i + 1, n, max_block.max(b), ids, out);
        }
    }
    ids[0] = 0;
    rec(1, n, 0, &mut ids, &mut out);
    out.into_iter().map(Partition::from_ids).collect()
}

/// Random partition for |U| >= 8, avoiding the Bell-number blow-up.
/// A uniform block count with uniform assignment is *not* uniform over set partitions. The same
/// bias applies to every arm, so between-arm comparison is valid; absolute values are not compared across |U|.
pub fn random_partition(n: usize, rng: &mut StdRng) -> Partition {
    let k = rng.gen_range(2..=n.max(2));
    let ids: Vec<u32> = (0..n).map(|_| rng.gen_range(0..k) as u32).collect();
    Partition::from_ids(ids)
}

pub fn initial_crystal(n: usize, all: Option<&[Partition]>, size: usize, seed: u64) -> Vec<Partition> {
    let mut rng = StdRng::seed_from_u64(seed);
    match all {
        Some(all) => {
            let mut idx: Vec<usize> = (0..all.len()).collect();
            idx.shuffle(&mut rng);
            idx.truncate(size);
            idx.iter().map(|&i| all[i].clone()).collect()
        }
        None => (0..size).map(|_| random_partition(n, &mut rng)).collect(),
    }
}

/// Splits one block of `p` in two: a pure refinement that never uses Meet.
/// Returns `p` unchanged when there is nothing to split (already discrete).
pub fn split_random_block(p: &Partition, rng: &mut StdRng) -> Partition {
    let ids = p.get_ids();
    let n = ids.len();
    let max_id = ids.iter().copied().max().unwrap_or(0);
    // Collect the blocks of size >= 2.
    let mut splittable: Vec<u32> = Vec::new();
    for b in 0..=max_id {
        if ids.iter().filter(|&&x| x == b).count() >= 2 {
            splittable.push(b);
        }
    }
    if splittable.is_empty() {
        return p.clone();
    }
    let target = splittable[rng.gen_range(0..splittable.len())];
    let members: Vec<usize> = (0..n).filter(|&i| ids[i] == target).collect();
    // Split the members into two non-empty groups at random.
    let cut = rng.gen_range(1..members.len());
    let mut shuffled = members.clone();
    shuffled.shuffle(rng);
    let mut new_ids = ids.to_vec();
    for &m in shuffled.iter().take(cut) {
        new_ids[m] = max_id + 1;
    }
    Partition::from_ids(new_ids)
}

pub fn h_of(p: &Partition) -> f64 {
    p.logical_entropy()
}

pub fn is_discrete(p: &Partition) -> bool {
    let ids = p.get_ids();
    let n = ids.len();
    let mut seen = ids.to_vec();
    seen.sort_unstable();
    seen.dedup();
    seen.len() == n
}

// ── Absorbing states, which differ structurally per arm ─────────────────────
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Absorbing {
    /// Refining arms: every element discrete, so nothing is left to split.
    AllDiscrete,
    /// Coarsening arms: everything indiscrete, so nothing is left to merge.
    AllIndiscrete,
    /// No directional pressure, hence no absorbing state and no rate to measure.
    None,
}

pub fn absorbing_target(arm: Arm) -> Absorbing {
    match arm {
        Arm::Gold | Arm::RandomReplace | Arm::NoMeetRefine => Absorbing::AllDiscrete,
        Arm::AntiPcc => Absorbing::AllIndiscrete,
        Arm::RandomWalk => Absorbing::None,
    }
}

pub fn is_absorbed(arm: Arm, crystal: &[Partition], n: usize) -> bool {
    match absorbing_target(arm) {
        Absorbing::AllDiscrete => crystal.iter().all(|p| p.num_blocks() == n),
        Absorbing::AllIndiscrete => crystal.iter().all(|p| p.num_blocks() == 1),
        Absorbing::None => false,
    }
}

/// Steps to a *stipulated* absorbing state. Inconclusive in practice:
/// the gold, A2 and A5 arms are fully censored, because the strict delta-h gate rejects
/// `Meet(discrete, x) = discrete`, so the all-discrete state is structurally unreachable.
/// Only A3's α = 1.007 held up.
#[allow(dead_code)]
pub fn run_to_absorption(
    arm: Arm,
    n: usize,
    initial: Vec<Partition>,
    seed: u64,
    max_steps: usize,
) -> Option<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut crystal = initial;
    if is_absorbed(arm, &crystal, n) {
        return Some(0);
    }
    for s in 1..=max_steps {
        step(arm, &mut crystal, n, &mut rng);
        if is_absorbed(arm, &crystal, n) {
            return Some(s);
        }
    }
    None
}

/// Observes the endpoint instead of stipulating it: the first step at which the crystal stays
/// bit-identical for `w` consecutive steps. Returns `(s*, final discrete fraction, final mean d_Sigma)`.
/// `s*` is the step at which the window opened, i.e. just after the last change; `None` means censored.
pub fn run_to_fixpoint(
    arm: Arm,
    n: usize,
    initial: Vec<Partition>,
    seed: u64,
    max_steps: usize,
    w: usize,
) -> (Option<usize>, f64, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut crystal = initial;
    let mut unchanged = 0usize;
    let mut window_start: usize = 0;
    let mut result: Option<usize> = None;

    for s in 1..=max_steps {
        let before: Vec<Vec<u32>> = crystal.iter().map(|p| p.get_ids().to_vec()).collect();
        step(arm, &mut crystal, n, &mut rng);
        let same = crystal
            .iter()
            .zip(before.iter())
            .all(|(p, b)| p.get_ids() == b.as_slice());
        if same {
            if unchanged == 0 {
                window_start = s;
            }
            unchanged += 1;
            if unchanged >= w {
                result = Some(window_start);
                break;
            }
        } else {
            unchanged = 0;
        }
    }
    let disc = crystal.iter().filter(|p| is_discrete(p)).count() as f64 / crystal.len() as f64;
    let md = dsigma_moments(n, &crystal).0;
    (result, disc, md)
}

/// Log-log OLS slope, i.e. the scaling exponent in `s* ~ |U|^alpha`.
pub fn loglog_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let lx: Vec<f64> = xs.iter().map(|x| x.ln()).collect();
    let ly: Vec<f64> = ys.iter().map(|y| y.ln()).collect();
    let mx = lx.iter().sum::<f64>() / n;
    let my = ly.iter().sum::<f64>() / n;
    let num: f64 = lx.iter().zip(&ly).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = lx.iter().map(|x| (x - mx).powi(2)).sum();
    if den.abs() < 1e-15 {
        f64::NAN
    } else {
        num / den
    }
}

pub fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

// ── One step per arm ───────────────────────────────────────────────────────
pub fn step(arm: Arm, crystal: &mut [Partition], n: usize, rng: &mut StdRng) {
    let len = crystal.len();
    if len < 2 {
        return;
    }
    let i = rng.gen_range(0..len);
    let mut j = rng.gen_range(0..len - 1);
    if j >= i {
        j += 1;
    }

    match arm {
        // A1: the PCC crystal step itself: Meet, delta-h gate, h-min replacement.
        Arm::Gold => {
            let candidate = crystal[i].differentiate(&crystal[j]);
            let h_candidate = h_of(&candidate);
            let h_base = h_of(&crystal[i]).max(h_of(&crystal[j]));
            if h_candidate <= h_base + EPS {
                return;
            }
            let (worst_idx, worst_h) = crystal
                .iter()
                .enumerate()
                .map(|(idx, p)| (idx, h_of(p)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            if h_candidate > worst_h + EPS {
                crystal[worst_idx] = candidate;
            }
        }
        // A2: candidates as in gold, but the replaced slot is random, removing h-min pressure.
        Arm::RandomReplace => {
            let candidate = crystal[i].differentiate(&crystal[j]);
            let h_candidate = h_of(&candidate);
            let h_base = h_of(&crystal[i]).max(h_of(&crystal[j]));
            if h_candidate <= h_base + EPS {
                return;
            }
            let victim = rng.gen_range(0..len);
            crystal[victim] = candidate;
        }
        // A3: pure refinement without Meet, asking whether refinement alone reproduces the trend.
        Arm::NoMeetRefine => {
            let victim = rng.gen_range(0..len);
            let refined = split_random_block(&crystal[victim], rng);
            crystal[victim] = refined;
        }
        // A4: no refinement pressure at all, a liveness control for the instrument.
        Arm::RandomWalk => {
            let victim = rng.gen_range(0..len);
            crystal[victim] = random_partition(n, rng);
        }
        // A5: sign-flipped, coarsening by Join with h-max replacement.
        Arm::AntiPcc => {
            let candidate = crystal[i].integrate(&crystal[j]);
            let h_candidate = h_of(&candidate);
            let h_base = h_of(&crystal[i]).min(h_of(&crystal[j]));
            if h_candidate >= h_base - EPS {
                return;
            }
            let (best_idx, best_h) = crystal
                .iter()
                .enumerate()
                .map(|(idx, p)| (idx, h_of(p)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            if h_candidate < best_h - EPS {
                crystal[best_idx] = candidate;
            }
        }
    }
}

pub struct Trace {
    pub kappa: Vec<f64>,
    /// First moment of the d_Σ distribution — a *direction* indicator
    pub mean_dsigma: Vec<f64>,
    pub mean_h: Vec<f64>,
    pub discrete_frac_final: f64,
}

pub fn run_trace(arm: Arm, n: usize, initial: Vec<Partition>, seed: u64) -> Trace {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut crystal = initial;
    let mut kappa = Vec::with_capacity(ITERATIONS + 1);
    let mut mean_dsigma = Vec::with_capacity(ITERATIONS + 1);
    let mut mean_h = Vec::with_capacity(ITERATIONS + 1);
    // Mean and variance together; the variance is kappa. Delegated, bit-exact.
    let (m0, k0) = dsigma_moments(n, &crystal);
    debug_assert_eq!(k0, discrete_curvature(n, &crystal));
    kappa.push(k0);
    mean_dsigma.push(m0);
    mean_h.push(crystal.iter().map(h_of).sum::<f64>() / crystal.len() as f64);
    for _ in 0..ITERATIONS {
        step(arm, &mut crystal, n, &mut rng);
        let (m, k) = dsigma_moments(n, &crystal);
        kappa.push(k);
        mean_dsigma.push(m);
        mean_h.push(crystal.iter().map(h_of).sum::<f64>() / crystal.len() as f64);
    }
    let discrete_frac_final =
        crystal.iter().filter(|p| is_discrete(p)).count() as f64 / crystal.len() as f64;
    Trace {
        kappa,
        mean_dsigma,
        mean_h,
        discrete_frac_final,
    }
}
