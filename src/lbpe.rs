//! L-BPE: Byte Pair Encoding as Partition Lattice Descent
//!
//! Implements BPE tokenization using the partition lattice framework.
//! The core mathematical identity is:
//!
//! **Δh = 2 · Σ(s_a · s_b) / N²**
//!
//! where `s_a`, `s_b` are the sizes (in characters) of the two merged token
//! blocks, and `N` is the corpus size. This expresses each BPE merge step
//! as a measurable descent on logical entropy `h(π)`.
//!
//! # Theory
//!
//! Given a corpus as a sequence of characters, BPE greedily merges the most
//! frequent adjacent pair at each step. In partition terms:
//! - Initial state: discrete partition `π₀` (each character its own block)
//! - Each merge: `π_{t+1} = π_t ∨ π_merge` (Join with the merge partition)
//! - Entropy `h(π)` decreases monotonically: `h(π₀) ≥ h(π₁) ≥ … ≥ h(π_T)`
//! - Δh at each step is exactly `2·Σ(s_a·s_b)/N²`
//!
//! # Quick Start
//!
//! ```rust
//! use partition_lattice::lbpe::BpeEngine;
//!
//! let text = b"abracadabra";
//! let mut engine = BpeEngine::new(text.len());
//! let trace = engine.run(text, 5);
//!
//! assert!(trace.is_monotone_decreasing());
//! assert!(trace.max_delta_h_error() < 1e-10);
//! println!("Compression ratio: {:.3}", trace.compression_ratio());
//! ```
//!
//! # References

use crate::unified::Partition;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::FloatExt;
use crate::HashMap;
#[allow(unused_imports)]
use alloc::borrow::ToOwned;
#[allow(unused_imports)]
use alloc::boxed::Box;
#[allow(unused_imports)]
use alloc::string::{String, ToString};
#[allow(unused_imports)]
use alloc::vec::Vec;

// ── Public types ──────────────────────────────────────────────────────────

/// Result of a single BPE merge step (reference implementation).
#[derive(Debug, Clone)]
pub struct BpeMergeResult {
    /// Partition after merge.
    pub partition: Partition,
    /// Logical entropy after merge.
    pub entropy: f64,
    /// Theoretical Δh = 2·Σ(s_a·s_b)/N².
    pub delta_h_theoretical: f64,
    /// Actual Δh = h(π_before) − h(π_after).
    pub delta_h_actual: f64,
    /// Number of non-overlapping merge occurrences.
    pub merge_count: usize,
}

/// Per-step record stored in a [`BpeTrace`].
#[derive(Debug, Clone)]
pub struct BpeStepRecord {
    /// Step index (0-based).
    pub step: usize,
    /// Logical entropy after this step.
    pub entropy: f64,
    /// Theoretical Δh (formula).
    pub delta_h_theoretical: f64,
    /// Actual Δh = h_before − h_after.
    pub delta_h_actual: f64,
    /// |Δh_actual − Δh_theoretical| — should be < 1e-10.
    pub delta_h_error: f64,
    /// Token count (sequence length) after this step.
    pub token_count: usize,
    /// Number of non-overlapping merges performed.
    pub merge_count: usize,
}

/// Complete BPE trace returned by [`run_bpe`] or [`BpeEngine::run`].
#[derive(Debug, Clone)]
pub struct BpeTrace {
    /// Number of bytes in the input.
    pub input_size: usize,
    /// Initial entropy h(π₀).
    pub initial_entropy: f64,
    /// Per-step records.
    pub steps: Vec<BpeStepRecord>,
}

impl BpeTrace {
    /// Maximum |Δh_actual − Δh_theoretical| across all steps.
    pub fn max_delta_h_error(&self) -> f64 {
        self.steps
            .iter()
            .map(|s| s.delta_h_error)
            .fold(0.0_f64, f64::max)
    }

    /// Whether entropy strictly decreases at every step.
    pub fn is_monotone_decreasing(&self) -> bool {
        let mut prev = self.initial_entropy;
        for s in &self.steps {
            if s.entropy >= prev {
                return false;
            }
            prev = s.entropy;
        }
        true
    }

    /// Final entropy after all merge steps.
    pub fn final_entropy(&self) -> f64 {
        self.steps
            .last()
            .map(|s| s.entropy)
            .unwrap_or(self.initial_entropy)
    }

    /// Compression ratio: h(π_final) / h(π₀). 0 = maximally compressed.
    pub fn compression_ratio(&self) -> f64 {
        if self.initial_entropy == 0.0 {
            return 1.0;
        }
        self.final_entropy() / self.initial_entropy
    }
}

// ── Core primitives ───────────────────────────────────────────────────────

/// Build a character-level discrete partition and token sequence from a byte slice.
///
/// Returns `(partition, sequence)` where:
/// - `partition` is the discrete partition over byte positions (N blocks)
/// - `sequence` maps byte values to canonical IDs (same byte → same ID)
///
/// # Example
/// ```rust
/// use partition_lattice::lbpe::char_level_partition;
/// let (part, seq) = char_level_partition(b"abab");
/// assert_eq!(seq, vec![0, 1, 0, 1]); // 'a'→0, 'b'→1
/// ```
pub fn char_level_partition(text: &[u8]) -> (Partition, Vec<u32>) {
    let n = text.len();
    let partition = Partition::discrete(n);
    let mut char_to_id: HashMap<u8, u32> = HashMap::new();
    let mut next_id = 0u32;
    let sequence: Vec<u32> = text
        .iter()
        .map(|&byte| {
            let id = char_to_id.entry(byte).or_insert_with(|| {
                let cur = next_id;
                next_id += 1;
                cur
            });
            *id
        })
        .collect();
    (partition, sequence)
}

/// Find the most frequent adjacent pair in a token sequence.
///
/// Tie-breaking: among pairs with equal frequency, the lexicographically
/// smallest pair is chosen for determinism.
pub fn find_most_frequent_pair(ids: &[u32]) -> Option<((u32, u32), usize)> {
    if ids.len() < 2 {
        return None;
    }
    let mut freq: HashMap<(u32, u32), usize> = HashMap::new();
    for w in ids.windows(2) {
        *freq.entry((w[0], w[1])).or_insert(0) += 1;
    }
    freq.into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
}

/// Rebuild the token sequence from a partition (collapses consecutive same-ID positions).
pub fn rebuild_sequence(partition: &Partition) -> Vec<u32> {
    let ids = partition.get_ids();
    let mut seq = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        let bid = ids[i];
        seq.push(bid);
        while i < ids.len() && ids[i] == bid {
            i += 1;
        }
    }
    seq
}

/// Perform one BPE merge step (reference implementation, allocates per-step).
///
/// Merges all non-overlapping left-to-right occurrences of the most frequent
/// adjacent pair. Returns `None` if no pair exists (sequence too short).
///
/// For performance-critical loops, prefer [`BpeEngine::merge_step`].
pub fn bpe_merge_step(current: &Partition, sequence: &[u32]) -> Option<BpeMergeResult> {
    let ((a, b), _freq) = find_most_frequent_pair(sequence)?;
    let n = current.size();

    // Build position map: sequence index → character positions
    let ids_slice = current.get_ids();
    let mut token_positions: Vec<(usize, usize)> = Vec::new(); // (start, len)
    let mut i = 0;
    while i < ids_slice.len() {
        let start = i;
        let bid = ids_slice[i];
        while i < ids_slice.len() && ids_slice[i] == bid {
            i += 1;
        }
        token_positions.push((start, i - start));
    }

    let mut merge_ids: Vec<u32> = (0..n as u32).collect();
    let mut merge_group = n as u32;
    let mut sum_products = 0usize;
    let mut actual_merges = 0usize;
    let mut skip_next = false;

    for (seq_i, w) in sequence.windows(2).enumerate() {
        if skip_next {
            skip_next = false;
            continue;
        }
        if w[0] == a && w[1] == b {
            if seq_i < token_positions.len() && seq_i + 1 < token_positions.len() {
                let (start_a, len_a) = token_positions[seq_i];
                let (start_b, len_b) = token_positions[seq_i + 1];
                sum_products += len_a * len_b;
                actual_merges += 1;

                for p in start_a..start_a + len_a {
                    merge_ids[p] = merge_group;
                }
                for p in start_b..start_b + len_b {
                    merge_ids[p] = merge_group;
                }
                merge_group += 1;
                skip_next = true;
            }
        }
    }

    let delta_h_theoretical = 2.0 * sum_products as f64 / (n as f64 * n as f64);
    let merge_partition = Partition::from_ids(merge_ids);
    let merged = current.integrate(&merge_partition);
    let h_before = current.logical_entropy();
    let h_after = merged.logical_entropy();

    Some(BpeMergeResult {
        partition: merged,
        entropy: h_after,
        delta_h_theoretical,
        delta_h_actual: h_before - h_after,
        merge_count: actual_merges,
    })
}

/// Run BPE for up to `max_steps` on a byte slice.
///
/// Convenience wrapper around `BpeEngine`. For repeated use on variable-length
/// inputs, prefer constructing a [`BpeEngine`] directly.
pub fn run_bpe(text: &[u8], max_steps: usize) -> BpeTrace {
    let mut engine = BpeEngine::new(text.len());
    engine.run(text, max_steps)
}

// ── BpeEngine (Phase A: pooled-buffer optimized implementation) ───────────

/// Optimized BPE engine with pooled buffers.
///
/// Reuses internal allocations across merge steps to minimize GC pressure:
/// - `merge_ids` buffer: cleared and refilled (no realloc)
/// - `pair_freq` HashMap: `clear()` preserves backing storage
/// - `token_positions`: flat `(start, len)` tuples (avoids inner `Vec`)
///
/// The engine produces **identical results** to [`bpe_merge_step`] /
/// [`run_bpe`] but with significantly lower allocation overhead on large inputs.
///
/// # Example
/// ```rust
/// use partition_lattice::lbpe::BpeEngine;
///
/// let text = b"hello world";
/// let mut engine = BpeEngine::new(text.len());
/// let trace = engine.run(text, 20);
/// assert!(trace.max_delta_h_error() < 1e-10, "Δh identity must hold");
/// ```
pub struct BpeEngine {
    merge_ids: Vec<u32>,
    pair_freq: HashMap<(u32, u32), usize>,
    token_positions: Vec<(usize, usize)>,
    n: usize,
}

/// Result of a single step from [`BpeEngine::merge_step`].
#[derive(Debug)]
pub struct EngineMergeResult {
    /// Partition after merge.
    pub partition: Partition,
    /// Updated token sequence.
    pub sequence: Vec<u32>,
    /// Logical entropy after merge.
    pub entropy: f64,
    /// Theoretical Δh = 2·Σ(s_a·s_b)/N².
    pub delta_h_theoretical: f64,
    /// Actual Δh = h_before − h_after.
    pub delta_h_actual: f64,
    /// Number of non-overlapping merge sites.
    pub merge_count: usize,
    /// Token count after merge.
    pub token_count: usize,
}

impl BpeEngine {
    /// Create a new engine for a corpus of size `n` bytes.
    pub fn new(n: usize) -> Self {
        Self {
            merge_ids: Vec::with_capacity(n),
            pair_freq: HashMap::new(),
            token_positions: Vec::new(),
            n,
        }
    }

    /// Perform one BPE merge step using pooled buffers.
    ///
    /// Returns `None` if no adjacent pair exists for merging.
    pub fn merge_step(
        &mut self,
        current: &Partition,
        sequence: &[u32],
    ) -> Option<EngineMergeResult> {
        let ((a, b), _freq) = self.find_most_frequent_pair(sequence)?;

        self.build_token_positions(current);
        self.reset_merge_ids();

        let mut merge_group = self.n as u32;
        let mut sum_products = 0usize;
        let mut actual_merges = 0usize;
        let mut skip_next = false;

        for (seq_i, w) in sequence.windows(2).enumerate() {
            if skip_next {
                skip_next = false;
                continue;
            }
            if w[0] == a && w[1] == b {
                if seq_i < self.token_positions.len() && seq_i + 1 < self.token_positions.len() {
                    let (start_a, len_a) = self.token_positions[seq_i];
                    let (start_b, len_b) = self.token_positions[seq_i + 1];
                    sum_products += len_a * len_b;
                    actual_merges += 1;

                    for p in start_a..start_a + len_a {
                        self.merge_ids[p] = merge_group;
                    }
                    for p in start_b..start_b + len_b {
                        self.merge_ids[p] = merge_group;
                    }
                    merge_group += 1;
                    skip_next = true;
                }
            }
        }

        let delta_h_theoretical = 2.0 * sum_products as f64 / (self.n as f64 * self.n as f64);
        let merge_partition = Partition::from_ids(self.merge_ids.clone());
        let merged = current.integrate(&merge_partition);
        let h_before = current.logical_entropy();
        let h_after = merged.logical_entropy();

        let new_sequence = Self::rebuild_seq(&merged);
        let token_count = new_sequence.len();

        Some(EngineMergeResult {
            partition: merged,
            sequence: new_sequence,
            entropy: h_after,
            delta_h_theoretical,
            delta_h_actual: h_before - h_after,
            merge_count: actual_merges,
            token_count,
        })
    }

    /// Run BPE for up to `max_steps` on `text`, returning the full trace.
    pub fn run(&mut self, text: &[u8], max_steps: usize) -> BpeTrace {
        let (mut partition, mut sequence) = char_level_partition(text);
        let initial_entropy = partition.logical_entropy();
        let mut steps = Vec::new();

        for step_idx in 0..max_steps {
            if sequence.len() <= 1 {
                break;
            }
            match self.merge_step(&partition, &sequence) {
                Some(result) => {
                    steps.push(BpeStepRecord {
                        step: step_idx,
                        entropy: result.entropy,
                        delta_h_theoretical: result.delta_h_theoretical,
                        delta_h_actual: result.delta_h_actual,
                        delta_h_error: (result.delta_h_actual - result.delta_h_theoretical).abs(),
                        token_count: result.token_count,
                        merge_count: result.merge_count,
                    });
                    partition = result.partition;
                    sequence = result.sequence;
                }
                None => break,
            }
        }

        BpeTrace {
            input_size: text.len(),
            initial_entropy,
            steps,
        }
    }

    // ── Private helpers ─────────────────────────────────────────────

    fn find_most_frequent_pair(&mut self, sequence: &[u32]) -> Option<((u32, u32), usize)> {
        self.pair_freq.clear();
        for w in sequence.windows(2) {
            *self.pair_freq.entry((w[0], w[1])).or_insert(0) += 1;
        }
        self.pair_freq
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            .map(|(&pair, &count)| (pair, count))
    }

    fn build_token_positions(&mut self, partition: &Partition) {
        self.token_positions.clear();
        let ids = partition.get_ids();
        let mut i = 0;
        while i < ids.len() {
            let start = i;
            let bid = ids[i];
            while i < ids.len() && ids[i] == bid {
                i += 1;
            }
            self.token_positions.push((start, i - start));
        }
    }

    fn reset_merge_ids(&mut self) {
        self.merge_ids.clear();
        self.merge_ids.extend(0..self.n as u32);
    }

    fn rebuild_seq(partition: &Partition) -> Vec<u32> {
        rebuild_sequence(partition)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    // ── P1: Δh identity (toy scale) ──────────────────────────────

    #[test]
    fn test_lbpe_p1_delta_h_abab() {
        let text = b"abab";
        let (partition, sequence) = char_level_partition(text);
        let h0 = partition.logical_entropy();
        assert!((h0 - 0.75).abs() < EPS, "h0={h0}, expected 0.75");
        let result = bpe_merge_step(&partition, &sequence).unwrap();
        assert!(
            (result.delta_h_actual - result.delta_h_theoretical).abs() < EPS,
            "Δh actual={} vs theoretical={}",
            result.delta_h_actual,
            result.delta_h_theoretical,
        );
        assert!(result.delta_h_actual > 0.0, "entropy must decrease");
    }

    #[test]
    fn test_lbpe_p1_delta_h_abracadabra() {
        let text = b"abracadabra";
        let (mut partition, mut sequence) = char_level_partition(text);
        for _ in 0..5 {
            if let Some(result) = bpe_merge_step(&partition, &sequence) {
                assert!(
                    (result.delta_h_actual - result.delta_h_theoretical).abs() < EPS,
                    "Δh identity violated",
                );
                sequence = rebuild_sequence(&result.partition);
                partition = result.partition;
            }
        }
    }

    // ── P2: char_level_partition ──────────────────────────────────

    #[test]
    fn test_lbpe_char_partition_identity_ids() {
        let (part, seq) = char_level_partition(b"abab");
        assert_eq!(seq, vec![0, 1, 0, 1]); // 'a'→0, 'b'→1
        assert_eq!(part.size(), 4);
        // discrete partition: N blocks
        assert_eq!(part.num_blocks(), 4);
    }

    #[test]
    fn test_lbpe_char_partition_uniform_text() {
        let (_, seq) = char_level_partition(b"aaaa");
        assert_eq!(seq, vec![0, 0, 0, 0]);
    }

    // ── P3: monotone entropy descent ─────────────────────────────

    #[test]
    fn test_lbpe_monotone_decreasing() {
        let trace = run_bpe(b"abracadabra", 20);
        assert!(
            trace.is_monotone_decreasing(),
            "entropy must strictly decrease"
        );
    }

    #[test]
    fn test_lbpe_compression_ratio_bounded() {
        let trace = run_bpe(b"aaabbbccc", 5);
        let r = trace.compression_ratio();
        assert!(r >= 0.0 && r <= 1.0, "compression ratio out of [0,1]: {r}");
    }

    // ── BpeEngine: correctness matches reference ──────────────────

    #[test]
    fn test_lbpe_engine_matches_reference() {
        let text = b"abracadabra";
        let steps = 10;
        let reference = {
            let (mut partition, mut sequence) = char_level_partition(text);
            let initial_entropy = partition.logical_entropy();
            let mut records = Vec::new();
            for i in 0..steps {
                if sequence.len() <= 1 {
                    break;
                }
                if let Some(r) = bpe_merge_step(&partition, &sequence) {
                    let tok = rebuild_sequence(&r.partition).len();
                    records.push((i, r.entropy, r.delta_h_theoretical, r.delta_h_actual, tok));
                    sequence = rebuild_sequence(&r.partition);
                    partition = r.partition;
                }
            }
            (initial_entropy, records)
        };

        let mut engine = BpeEngine::new(text.len());
        let trace = engine.run(text, steps);

        assert!((reference.0 - trace.initial_entropy).abs() < EPS);
        assert_eq!(reference.1.len(), trace.steps.len(), "step count mismatch");
        for (i, (_, h_ref, dth_ref, dta_ref, tc_ref)) in reference.1.iter().enumerate() {
            let s = &trace.steps[i];
            assert!((h_ref - s.entropy).abs() < EPS, "step {i} entropy");
            assert!(
                (dth_ref - s.delta_h_theoretical).abs() < EPS,
                "step {i} Δh_theo"
            );
            assert!(
                (dta_ref - s.delta_h_actual).abs() < EPS,
                "step {i} Δh_actual"
            );
            assert_eq!(*tc_ref, s.token_count, "step {i} token_count");
        }
    }

    #[test]
    fn test_lbpe_engine_pooled_hashmap_no_contamination() {
        let mut engine = BpeEngine::new(10);
        let seq1 = vec![0u32, 1, 0, 1, 2];
        let r1 = engine.find_most_frequent_pair(&seq1);
        assert_eq!(r1, Some(((0, 1), 2)));
        let seq2 = vec![3u32, 4, 3, 4, 3, 4];
        let r2 = engine.find_most_frequent_pair(&seq2);
        assert_eq!(r2, Some(((3, 4), 3)));
        assert!(!engine.pair_freq.contains_key(&(0, 1)));
    }

    #[test]
    fn test_lbpe_engine_delta_h_identity_holds() {
        let mut engine = BpeEngine::new(9);
        let trace = engine.run(b"abcabcabc", 3);
        for rec in &trace.steps {
            assert!(
                rec.delta_h_error < 1e-10,
                "step {}: Δh error={}",
                rec.step,
                rec.delta_h_error
            );
        }
    }

    #[test]
    fn test_lbpe_engine_monotonicity() {
        let mut engine = BpeEngine::new(11);
        let trace = engine.run(b"abracadabra", 100);
        assert!(trace.is_monotone_decreasing());
    }

    #[test]
    fn test_lbpe_engine_single_char_no_crash() {
        let mut engine = BpeEngine::new(1);
        let trace = engine.run(b"a", 10);
        assert!(trace.steps.is_empty());
    }

    #[test]
    fn test_lbpe_engine_empty_no_crash() {
        let mut engine = BpeEngine::new(0);
        let trace = engine.run(b"", 10);
        assert!(trace.steps.is_empty());
    }

    // ── find_most_frequent_pair ───────────────────────────────────

    #[test]
    fn test_lbpe_find_pair_basic() {
        assert_eq!(find_most_frequent_pair(&[0, 1, 0, 1, 2]), Some(((0, 1), 2)));
    }

    #[test]
    fn test_lbpe_find_pair_too_short() {
        assert_eq!(find_most_frequent_pair(&[0]), None);
        assert_eq!(find_most_frequent_pair(&[]), None);
    }

    #[test]
    fn test_lbpe_find_pair_tie_breaking_lexicographic() {
        // (0,1) and (1,0) each appear once → smallest lex = (0,1)
        let pair = find_most_frequent_pair(&[0, 1, 1, 0]);
        assert_eq!(pair, Some(((0, 1), 1)));
    }

    // ── rebuild_sequence ─────────────────────────────────────────

    #[test]
    fn test_lbpe_rebuild_sequence_discrete() {
        let (part, _) = char_level_partition(b"abab");
        let seq = rebuild_sequence(&part);
        // Discrete partition: N distinct blocks → sequence length = N
        assert_eq!(seq.len(), 4);
    }
}
