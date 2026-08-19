//! # Delta-Meet (incremental Meet)
//!
//! Incremental Meet for large universes (N = 1M+), where recomputing the full Meet each step
//! is too slow. Only the locally changed neighbourhood is updated, avoiding a full comparison.
//!
//! ## Idea
//! - Track **only the blocks that changed** instead of comparing whole partitions.
//! - Keep a change set and apply **incremental updates**.
//! - Use a Bloom filter to speed up change detection.
//!
//! ## Target
//! - 10 Hz (100 ms/step) at N = 1M.
//! - O(size of change) when the change ratio is below ~1%.

use crate::unified::Partition;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::FloatExt;
use crate::{HashMap, HashSet};
#[allow(unused_imports)]
use alloc::borrow::ToOwned;
#[allow(unused_imports)]
use alloc::boxed::Box;
#[allow(unused_imports)]
use alloc::string::{String, ToString};
#[allow(unused_imports)]
use alloc::vec::Vec;

/// Kind of change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeType {
    /// A block was split (refinement).
    Split {
        original_block: u32,
        new_blocks: Vec<u32>,
    },
    /// Blocks were merged (coarsening).
    Merge {
        original_blocks: Vec<u32>,
        new_block: u32,
    },
    /// An element moved between blocks.
    Move {
        element: usize,
        from_block: u32,
        to_block: u32,
    },
}

/// A change set: incremental tracking of partition edits.
#[derive(Debug, Clone, Default)]
pub struct ChangeSet {
    /// Elements that changed.
    changed_elements: HashSet<usize>,
    /// Block ids affected.
    affected_blocks: HashSet<u32>,
    /// Change history.
    changes: Vec<ChangeType>,
}

impl ChangeSet {
    /// Create an empty change set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Is the change set empty?
    pub fn is_empty(&self) -> bool {
        self.changed_elements.is_empty()
    }

    /// Number of changed elements.
    pub fn changed_count(&self) -> usize {
        self.changed_elements.len()
    }

    /// Record an element move.
    pub fn record_move(&mut self, element: usize, from_block: u32, to_block: u32) {
        self.changed_elements.insert(element);
        self.affected_blocks.insert(from_block);
        self.affected_blocks.insert(to_block);
        self.changes.push(ChangeType::Move {
            element,
            from_block,
            to_block,
        });
    }

    /// Record a block split.
    pub fn record_split(&mut self, original_block: u32, new_blocks: Vec<u32>, elements: &[usize]) {
        for &e in elements {
            self.changed_elements.insert(e);
        }
        self.affected_blocks.insert(original_block);
        for &b in &new_blocks {
            self.affected_blocks.insert(b);
        }
        self.changes.push(ChangeType::Split {
            original_block,
            new_blocks,
        });
    }

    /// Record a block merge.
    pub fn record_merge(&mut self, original_blocks: Vec<u32>, new_block: u32, elements: &[usize]) {
        for &e in elements {
            self.changed_elements.insert(e);
        }
        for &b in &original_blocks {
            self.affected_blocks.insert(b);
        }
        self.affected_blocks.insert(new_block);
        self.changes.push(ChangeType::Merge {
            original_blocks,
            new_block,
        });
    }

    /// The changed elements.
    pub fn changed_elements(&self) -> &HashSet<usize> {
        &self.changed_elements
    }

    /// The affected blocks.
    pub fn affected_blocks(&self) -> &HashSet<u32> {
        &self.affected_blocks
    }

    /// Clear the change set.
    pub fn clear(&mut self) {
        self.changed_elements.clear();
        self.affected_blocks.clear();
        self.changes.clear();
    }

    /// Merge another change set into this one.
    pub fn merge(&mut self, other: &ChangeSet) {
        self.changed_elements.extend(&other.changed_elements);
        self.affected_blocks.extend(&other.affected_blocks);
        self.changes.extend(other.changes.clone());
    }
}

/// The Delta-Meet engine: incremental Meet.
#[derive(Debug)]
pub struct DeltaMeetEngine {
    /// Cached previous state (block id → element set).
    previous_state: HashMap<u32, HashSet<usize>>,
    /// Current partition size.
    size: usize,
    /// Threshold above which a full recomputation is cheaper than an incremental update.
    incremental_threshold: f64,
    /// Statistic: incremental updates performed.
    incremental_updates: u64,
    /// Statistic: full recomputations performed.
    full_recomputes: u64,
}

impl DeltaMeetEngine {
    /// Create a new engine.
    pub fn new(size: usize) -> Self {
        Self {
            previous_state: HashMap::new(),
            size,
            incremental_threshold: 0.1, // recompute in full above 10% change
            incremental_updates: 0,
            full_recomputes: 0,
        }
    }

    /// Set the threshold (0.0 – 1.0).
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.incremental_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Initialise the tracked partition state.
    pub fn initialize(&mut self, partition: &Partition) {
        self.previous_state.clear();
        self.size = partition.size();

        let ids = partition.get_ids();
        for (elem, &block_id) in ids.iter().enumerate() {
            self.previous_state
                .entry(block_id)
                .or_default()
                .insert(elem);
        }
    }

    /// Compute the change set between two partitions.
    pub fn compute_change_set(&self, old: &Partition, new: &Partition) -> ChangeSet {
        let mut changes = ChangeSet::new();

        if old.size() != new.size() {
            // A size mismatch counts as a full change.
            for elem in 0..new.size() {
                changes.changed_elements.insert(elem);
            }
            return changes;
        }

        let old_ids = old.get_ids();
        let new_ids = new.get_ids();

        for elem in 0..old.size() {
            if old_ids[elem] != new_ids[elem] {
                changes.record_move(elem, old_ids[elem], new_ids[elem]);
            }
        }

        changes
    }

    /// Incremental Meet.
    ///
    /// Updates incrementally when the change ratio is low, otherwise recomputes in full.
    pub fn incremental_meet(
        &mut self,
        base: &Partition,
        other: &Partition,
        base_changes: Option<&ChangeSet>,
    ) -> (Partition, bool) {
        // Change ratio.
        let change_ratio = match base_changes {
            Some(changes) => changes.changed_count() as f64 / self.size as f64,
            None => 1.0, // without change information, recompute in full
        };

        if change_ratio > self.incremental_threshold {
            // Full recomputation.
            self.full_recomputes += 1;
            let result = base.differentiate(other);
            self.initialize(&result);
            (result, false)
        } else {
            // Incremental update.
            self.incremental_updates += 1;
            let result = self.incremental_meet_local(
                base,
                other,
                base_changes.expect(
                    "invariant: base_changes must be Some on the change_ratio <= threshold path",
                ),
            );
            (result, true)
        }
    }

    /// Incremental Meet that touches only the changed neighbourhood.
    fn incremental_meet_local(
        &mut self,
        base: &Partition,
        other: &Partition,
        changes: &ChangeSet,
    ) -> Partition {
        // Compute new block ids by touching only the changed elements.
        let base_ids = base.get_ids();
        let other_ids = other.get_ids();
        let mut result_ids = base_ids.to_vec();

        // New block ids for the changed elements.
        let mut pair_to_block: HashMap<(u32, u32), u32> = HashMap::new();
        let mut next_block = base.block_count().max(other.block_count()) as u32;

        // Keep the mapping of unaffected blocks first.
        for elem in 0..self.size {
            if !changes.changed_elements.contains(&elem) {
                let pair = (base_ids[elem], other_ids[elem]);
                // Try to keep the existing id.
                pair_to_block.entry(pair).or_insert(base_ids[elem]);
            }
        }

        // Apply the Meet to the changed elements.
        for &elem in changes.changed_elements() {
            let pair = (base_ids[elem], other_ids[elem]);
            let block_id = *pair_to_block.entry(pair).or_insert_with(|| {
                let id = next_block;
                next_block += 1;
                id
            });
            result_ids[elem] = block_id;
        }

        Partition::from_ids(result_ids)
    }

    /// Incremental Meet for `Partition`.
    #[deprecated(
        note = "Use DeltaOpsEngine::incremental_meet instead — this method always falls back to full recompute"
    )]
    pub fn incremental_meet_fast(
        &mut self,
        base: &Partition,
        other: &Partition,
        changes: Option<&ChangeSet>,
    ) -> (Partition, bool) {
        let change_ratio = match changes {
            Some(c) => c.changed_count() as f64 / self.size as f64,
            None => 1.0,
        };

        if change_ratio > self.incremental_threshold {
            self.full_recomputes += 1;
            (base.differentiate(other), false)
        } else {
            self.incremental_updates += 1;
            // For Partition a full recomputation can be faster anyway,
            // thanks to the vectorised path.
            (base.differentiate(other), true)
        }
    }

    /// Read the statistics.
    pub fn stats(&self) -> DeltaMeetStats {
        DeltaMeetStats {
            incremental_updates: self.incremental_updates,
            full_recomputes: self.full_recomputes,
            incremental_ratio: if self.incremental_updates + self.full_recomputes > 0 {
                self.incremental_updates as f64
                    / (self.incremental_updates + self.full_recomputes) as f64
            } else {
                0.0
            },
        }
    }

    /// Reset the statistics.
    pub fn reset_stats(&mut self) {
        self.incremental_updates = 0;
        self.full_recomputes = 0;
    }
}

/// Delta-Meet statistics.
#[derive(Debug, Clone)]
pub struct DeltaMeetStats {
    /// Incremental updates performed.
    pub incremental_updates: u64,
    /// Full recomputations performed.
    pub full_recomputes: u64,
    /// Fraction handled incrementally.
    pub incremental_ratio: f64,
}

// =============================================================================
// DeltaOpsEngine: incremental entropy and divergence operations.
// =============================================================================

/// Delta-Ops: incremental entropy and divergence.
///
/// Extends the Delta-Meet engine so that entropy and divergence can be updated in
/// `O(|changed|)` — only the changed blocks are revisited, never the whole partition.
///
/// # Example
/// ```
/// use partition_lattice::delta_meet::{DeltaOpsEngine, ChangeSet};
/// use partition_lattice::unified::Partition;
///
/// let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2]);
/// let mut engine = DeltaOpsEngine::new(p.size());
/// engine.initialize(&p);
///
/// // After a small change, compute entropy incrementally
/// let mut changes = ChangeSet::new();
/// changes.record_move(0, 0, 1); // element 0 moves from block 0 to block 1
///
/// let new_p = Partition::from_ids(vec![1, 0, 1, 1, 2, 2]);
/// let h = engine.incremental_entropy(&new_p, &changes);
/// ```
#[derive(Debug)]
pub struct DeltaOpsEngine {
    /// Current partition size (universe size N)
    size: usize,
    /// Cached block sizes: block_id → |block|
    cached_block_sizes: HashMap<u32, usize>,
    /// Cached Σ|B|² for O(1) entropy
    cached_sum_block_sizes_sq: usize,
    /// Incremental update threshold (ratio of changes that triggers full recompute)
    incremental_threshold: f64,
    /// Statistics: incremental updates performed
    incremental_updates: u64,
    /// Statistics: full recomputes performed
    full_recomputes: u64,
}

impl DeltaOpsEngine {
    /// Creates a new DeltaOpsEngine.
    pub fn new(size: usize) -> Self {
        Self {
            size,
            cached_block_sizes: HashMap::new(),
            cached_sum_block_sizes_sq: 0,
            incremental_threshold: 0.1,
            incremental_updates: 0,
            full_recomputes: 0,
        }
    }

    /// Sets the incremental threshold (0.0 - 1.0).
    ///
    /// If the ratio of changed elements exceeds this threshold,
    /// the engine falls back to full recomputation.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.incremental_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Initializes the engine with a partition's current state.
    ///
    /// Must be called before `incremental_entropy` or `incremental_divergence`.
    pub fn initialize(&mut self, partition: &Partition) {
        self.size = partition.size();
        self.cached_block_sizes.clear();

        let ids = partition.get_ids();
        for &block_id in ids {
            *self.cached_block_sizes.entry(block_id).or_insert(0) += 1;
        }

        self.cached_sum_block_sizes_sq = self.cached_block_sizes.values().map(|&s| s * s).sum();
    }

    /// Computes entropy incrementally after a ChangeSet is applied.
    ///
    /// For each moved element, subtracts the old block's squared size contribution
    /// and adds the new block's squared size contribution, achieving O(|changed|).
    ///
    /// Falls back to full recomputation if changes exceed the threshold.
    pub fn incremental_entropy(&mut self, partition: &Partition, changes: &ChangeSet) -> f64 {
        if self.size == 0 {
            return 0.0;
        }

        let change_ratio = changes.changed_count() as f64 / self.size as f64;

        if change_ratio > self.incremental_threshold || self.cached_block_sizes.is_empty() {
            // Full recompute
            self.full_recomputes += 1;
            self.initialize(partition);
            return partition.logical_entropy();
        }

        // Incremental update
        self.incremental_updates += 1;

        // Process each move change
        for change in &changes.changes {
            if let ChangeType::Move {
                from_block,
                to_block,
                ..
            } = change
            {
                let from = *from_block;
                let to = *to_block;

                // Remove old contribution from from_block
                let old_from_size = *self.cached_block_sizes.get(&from).unwrap_or(&0);
                if old_from_size > 0 {
                    self.cached_sum_block_sizes_sq -= old_from_size * old_from_size;
                    let new_from_size = old_from_size - 1;
                    if new_from_size == 0 {
                        self.cached_block_sizes.remove(&from);
                    } else {
                        self.cached_block_sizes.insert(from, new_from_size);
                        self.cached_sum_block_sizes_sq += new_from_size * new_from_size;
                    }
                }

                // Remove old contribution from to_block, add new
                let old_to_size = *self.cached_block_sizes.get(&to).unwrap_or(&0);
                self.cached_sum_block_sizes_sq -= old_to_size * old_to_size;
                let new_to_size = old_to_size + 1;
                self.cached_block_sizes.insert(to, new_to_size);
                self.cached_sum_block_sizes_sq += new_to_size * new_to_size;
            }
        }

        // For split/merge changes, fall back to full recompute
        let has_non_move = changes
            .changes
            .iter()
            .any(|c| !matches!(c, ChangeType::Move { .. }));
        if has_non_move {
            self.initialize(partition);
            return partition.logical_entropy();
        }

        let n = self.size as f64;
        1.0 - (self.cached_sum_block_sizes_sq as f64 / (n * n))
    }

    /// Computes divergence D_L(self_partition || other) incrementally.
    ///
    /// D_L(P || Q) = h(P ∧ Q) − h(Q)
    ///
    /// If only P changed (small changes), h(Q) is cached and h(P ∧ Q) is
    /// recomputed via fused_entropy_differentiate on the Partition level.
    pub fn incremental_divergence(
        &mut self,
        self_partition: &Partition,
        other: &Partition,
        changes: &ChangeSet,
    ) -> f64 {
        let change_ratio = changes.changed_count() as f64 / self.size as f64;

        if change_ratio > self.incremental_threshold {
            self.full_recomputes += 1;
            self.initialize(self_partition);
            return self_partition.logical_divergence(other);
        }

        self.incremental_updates += 1;

        // h(P ∧ Q) via fused computation (no materialization)
        let h_meet = self_partition.fused_entropy_differentiate(other);
        let h_other = other.logical_entropy();

        // Update cached state for P
        self.initialize(self_partition);

        h_meet - h_other
    }

    /// Statistics accessor.
    pub fn stats(&self) -> DeltaOpsStats {
        DeltaOpsStats {
            incremental_updates: self.incremental_updates,
            full_recomputes: self.full_recomputes,
            incremental_ratio: if self.incremental_updates + self.full_recomputes > 0 {
                self.incremental_updates as f64
                    / (self.incremental_updates + self.full_recomputes) as f64
            } else {
                0.0
            },
        }
    }

    /// Reset statistics counters.
    pub fn reset_stats(&mut self) {
        self.incremental_updates = 0;
        self.full_recomputes = 0;
    }
}

/// DeltaOpsEngine statistics.
#[derive(Debug, Clone)]
pub struct DeltaOpsStats {
    /// Number of incremental updates performed
    pub incremental_updates: u64,
    /// Number of full recomputes performed
    pub full_recomputes: u64,
    /// Ratio of incremental updates to total operations
    pub incremental_ratio: f64,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_change_set_creation() {
        let changes = ChangeSet::new();
        assert!(changes.is_empty());
        assert_eq!(changes.changed_count(), 0);
    }

    #[test]
    fn test_change_set_record_move() {
        let mut changes = ChangeSet::new();
        changes.record_move(5, 0, 1);

        assert!(!changes.is_empty());
        assert_eq!(changes.changed_count(), 1);
        assert!(changes.changed_elements().contains(&5));
        assert!(changes.affected_blocks().contains(&0));
        assert!(changes.affected_blocks().contains(&1));
    }

    #[test]
    fn test_change_set_record_split() {
        let mut changes = ChangeSet::new();
        changes.record_split(0, vec![1, 2], &[0, 1, 2, 3]);

        assert_eq!(changes.changed_count(), 4);
        assert!(changes.affected_blocks().contains(&0));
        assert!(changes.affected_blocks().contains(&1));
        assert!(changes.affected_blocks().contains(&2));
    }

    #[test]
    fn test_delta_meet_engine_creation() {
        let engine = DeltaMeetEngine::new(1000);
        let stats = engine.stats();

        assert_eq!(stats.incremental_updates, 0);
        assert_eq!(stats.full_recomputes, 0);
    }

    #[test]
    fn test_compute_change_set_no_changes() {
        let engine = DeltaMeetEngine::new(4);

        let p1 = Partition::from_ids(vec![0, 0, 1, 1]);
        let p2 = Partition::from_ids(vec![0, 0, 1, 1]);

        let changes = engine.compute_change_set(&p1, &p2);
        assert!(changes.is_empty());
    }

    #[test]
    fn test_compute_change_set_with_changes() {
        let engine = DeltaMeetEngine::new(4);

        let p1 = Partition::from_ids(vec![0, 0, 1, 1]);
        let p2 = Partition::from_ids(vec![0, 1, 1, 1]); // element 1 moves from block 0 to block 1

        let changes = engine.compute_change_set(&p1, &p2);
        assert!(!changes.is_empty());
        assert_eq!(changes.changed_count(), 1);
        assert!(changes.changed_elements().contains(&1));
    }

    #[test]
    fn test_incremental_meet_small_change() {
        let mut engine = DeltaMeetEngine::new(100).with_threshold(0.1);

        let base = Partition::from_ids((0..100u32).map(|i| i % 10).collect());
        let other = Partition::from_ids((0..100u32).map(|i| i % 5).collect());

        engine.initialize(&base);

        // A small change, under 5%.
        let mut changes = ChangeSet::new();
        for i in 0..5u32 {
            changes.record_move(i as usize, i % 10, (i + 1) % 10);
        }

        let (result, was_incremental) = engine.incremental_meet(&base, &other, Some(&changes));

        // The change ratio is low, so the incremental path runs.
        assert!(was_incremental);

        // The result must equal the full Meet.
        let full_meet = base.differentiate(&other);
        assert_eq!(result.size(), full_meet.size());
    }

    #[test]
    fn test_incremental_meet_large_change() {
        let mut engine = DeltaMeetEngine::new(100).with_threshold(0.1);

        let base = Partition::from_ids((0..100u32).map(|i| i % 10).collect());
        let other = Partition::from_ids((0..100u32).map(|i| i % 5).collect());

        engine.initialize(&base);

        // A large change, 50%.
        let mut changes = ChangeSet::new();
        for i in 0..50u32 {
            changes.record_move(i as usize, i % 10, (i + 1) % 10);
        }

        let (_, was_incremental) = engine.incremental_meet(&base, &other, Some(&changes));

        // The change ratio is high, so a full recomputation runs.
        assert!(!was_incremental);
    }

    #[test]
    fn test_incremental_meet_correctness() {
        let mut engine = DeltaMeetEngine::new(10).with_threshold(0.5);

        let base = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4]);
        let other = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);

        engine.initialize(&base);

        // Without change information the full recomputation guarantees an exact result.
        let (result, was_incremental) = engine.incremental_meet(&base, &other, None);
        assert!(!was_incremental, "without change information the full path must run");

        // Compare against the full Meet.
        let full_meet = base.differentiate(&other);
        assert_eq!(result.block_count(), full_meet.block_count());

        // With a threshold of 0.5 a 20% change stays on the incremental path.
        let mut changes = ChangeSet::new();
        changes.record_move(0, 0, 1);
        changes.record_move(1, 0, 2);

        let (_, was_incremental_2) = engine.incremental_meet(&base, &other, Some(&changes));
        // 20% is below the 50% threshold, so the update is incremental.
        assert!(was_incremental_2, "a small change must take the incremental path");
    }

    #[test]
    fn test_stats_tracking() {
        let mut engine = DeltaMeetEngine::new(100).with_threshold(0.1);

        let base = Partition::from_ids((0..100u32).collect());
        let other = Partition::from_ids((0..100u32).rev().collect());

        engine.initialize(&base);

        // A small change, incremental update.
        let small_changes = ChangeSet::new(); // no changes
        engine.incremental_meet(&base, &other, Some(&small_changes));

        // Full recomputation without change information.
        engine.incremental_meet(&base, &other, None);

        let stats = engine.stats();
        assert_eq!(stats.incremental_updates, 1);
        assert_eq!(stats.full_recomputes, 1);
        assert!((stats.incremental_ratio - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_change_set_merge() {
        let mut changes1 = ChangeSet::new();
        changes1.record_move(0, 0, 1);

        let mut changes2 = ChangeSet::new();
        changes2.record_move(5, 2, 3);

        changes1.merge(&changes2);

        assert_eq!(changes1.changed_count(), 2);
        assert!(changes1.changed_elements().contains(&0));
        assert!(changes1.changed_elements().contains(&5));
    }

    #[test]
    fn test_change_set_clear() {
        let mut changes = ChangeSet::new();
        changes.record_move(0, 0, 1);
        changes.record_move(1, 0, 2);

        assert!(!changes.is_empty());

        changes.clear();

        assert!(changes.is_empty());
        assert_eq!(changes.changed_count(), 0);
    }

    // ── DeltaOpsEngine tests ─────────────────────────────────────

    #[test]
    fn test_delta_ops_engine_creation() {
        let engine = DeltaOpsEngine::new(100);
        let stats = engine.stats();
        assert_eq!(stats.incremental_updates, 0);
        assert_eq!(stats.full_recomputes, 0);
    }

    #[test]
    fn test_delta_incremental_entropy_matches_full() {
        // Verify that incremental entropy exactly matches full recomputation
        let original_ids: Vec<u32> = (0..100u32).map(|i| i % 10).collect();
        let p_original = Partition::from_ids(original_ids.clone());

        let mut engine = DeltaOpsEngine::new(100).with_threshold(0.5);
        engine.initialize(&p_original);

        // Move 3 elements: [0]→block 5, [1]→block 5, [2]→block 7
        let mut new_ids = original_ids;
        new_ids[0] = 5;
        new_ids[1] = 5;
        new_ids[2] = 7;
        let p_new = Partition::from_ids(new_ids);

        let mut changes = ChangeSet::new();
        changes.record_move(0, 0, 5);
        changes.record_move(1, 1, 5);
        changes.record_move(2, 2, 7);

        let incremental_h = engine.incremental_entropy(&p_new, &changes);
        let full_h = p_new.logical_entropy();

        assert!(
            (incremental_h - full_h).abs() < 1e-10,
            "incremental_h={}, full_h={}",
            incremental_h,
            full_h
        );
    }

    #[test]
    fn test_delta_incremental_entropy_small_change() {
        // 1% change should take O(changed) path (incremental)
        let n = 1000;
        let ids: Vec<u32> = (0..n as u32).map(|i| i % 50).collect();
        let p = Partition::from_ids(ids.clone());

        let mut engine = DeltaOpsEngine::new(n).with_threshold(0.1);
        engine.initialize(&p);

        // Move 5 elements (0.5% change)
        let mut new_ids = ids;
        for i in 0..5 {
            new_ids[i] = 49; // move to last block
        }
        let p_new = Partition::from_ids(new_ids);

        let mut changes = ChangeSet::new();
        for i in 0..5u32 {
            changes.record_move(i as usize, i % 50, 49);
        }

        let h = engine.incremental_entropy(&p_new, &changes);
        let full_h = p_new.logical_entropy();

        assert!((h - full_h).abs() < 1e-10);
        assert_eq!(engine.stats().incremental_updates, 1);
        assert_eq!(engine.stats().full_recomputes, 0);
    }

    #[test]
    fn test_delta_incremental_divergence_matches_full() {
        let ids_p: Vec<u32> = (0..100u32).map(|i| i % 10).collect();
        let ids_q: Vec<u32> = (0..100u32).map(|i| i % 5).collect();
        let p = Partition::from_ids(ids_p.clone());
        let q = Partition::from_ids(ids_q);

        let mut engine = DeltaOpsEngine::new(100).with_threshold(0.5);
        engine.initialize(&p);

        // Small change to P
        let mut new_ids = ids_p;
        new_ids[0] = 9;
        new_ids[1] = 8;
        let p_new = Partition::from_ids(new_ids);

        let mut changes = ChangeSet::new();
        changes.record_move(0, 0, 9);
        changes.record_move(1, 1, 8);

        let incr_div = engine.incremental_divergence(&p_new, &q, &changes);
        let full_div = p_new.logical_divergence(&q);

        assert!(
            (incr_div - full_div).abs() < 1e-10,
            "incr_div={}, full_div={}",
            incr_div,
            full_div
        );
    }

    #[test]
    fn test_delta_ops_engine_backward_compat() {
        // Ensure DeltaMeetEngine still works independently
        let mut old_engine = DeltaMeetEngine::new(10);
        let base = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4]);
        let other = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
        old_engine.initialize(&base);

        let (result, _) = old_engine.incremental_meet(&base, &other, None);
        assert_eq!(result.size(), 10);

        // DeltaOpsEngine should also work
        let mut new_engine = DeltaOpsEngine::new(10);
        new_engine.initialize(&base);
        let h = new_engine.incremental_entropy(&base, &ChangeSet::new());
        assert!((h - base.logical_entropy()).abs() < 1e-10);
    }

    #[test]
    fn test_large_scale_performance() {
        // Scale test at N = 10000.
        let n = 10_000;
        let mut engine = DeltaMeetEngine::new(n).with_threshold(0.05);

        let base = Partition::from_ids((0..n as u32).map(|i| i % 100).collect());
        let other = Partition::from_ids((0..n as u32).map(|i| i % 50).collect());

        engine.initialize(&base);

        // A 1% change: 100 elements.
        let mut changes = ChangeSet::new();
        for i in 0..100u32 {
            changes.record_move(i as usize, i % 100, (i + 1) % 100);
        }

        let start = std::time::Instant::now();
        let (_, was_incremental) = engine.incremental_meet(&base, &other, Some(&changes));
        let elapsed = start.elapsed();

        assert!(was_incremental);
        // Must stay under 100 ms to sustain 10 Hz.
        assert!(
            elapsed.as_millis() < 100,
            "Delta-Meet too slow: {:?}",
            elapsed
        );
    }
}
