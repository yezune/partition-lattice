//! Distributed partition computing via MapReduce on sharded universes.
//!
//! Enables partition lattice operations across multiple machines by sharding
//! the universe $U = S_1 \sqcup S_2 \sqcup \cdots \sqcup S_m$ so each node
//! holds only `π|_{Sᵢ}` and `σ|_{Sᵢ}`. Communication cost is $O(K_1 \cdot K_2)$
//! — independent of $|U|$ — while all results are mathematically exact.
//!
//! # Protocols
//!
//! | Protocol | Operation | Communication per shard | Reduce |
//! |----------|-----------|------------------------|--------|
//! | P1 | $h(\pi)$ | $O(K)$ block counts | merge + sum |
//! | P2 | $h(\pi \wedge \sigma)$ | $O(K_1 K_2)$ composite counts | merge + sum |
//! | P3 | $d_{\oplus}, D_L, I$ | reuses P1 + P2 | $O(1)$ |
//! | P4 | $\pi \wedge \sigma$ materialization | $O(0)$ (element-local) | none |
//! | P5 | $h(\pi \vee \sigma)$ | $O(K_1 K_2)$ edges + sizes | Union-Find |
//!
//! # Mathematical Guarantees
//!
//! - Meet distributes over restriction: $(π ∧ σ)|_S = π|_S ∧ σ|_S$
//! - All reduce operations are associative and commutative (CRDT-safe)
//! - Exact results: distributed ≡ monolithic (error < 1e-12)
//!
//! # Example
//!
//! ```
//! use partition_lattice::Partition;
//! use partition_lattice::distributed::{shard_universe, distributed_refine_entropy};
//!
//! let pi = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3]);
//! let sigma = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1]);
//! let n = pi.size();
//!
//! // Shard into 2 partitions (simulating 2 machines)
//! let shards = shard_universe(&pi, &sigma, 2);
//!
//! // Distributed h(π ∧ σ) — exact match with monolithic
//! let (h_meet, bytes) = distributed_refine_entropy(&shards, n);
//! let mono = pi.fused_entropy_differentiate(&sigma);
//! assert!((h_meet - mono).abs() < 1e-12);
//! ```

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

// ---------------------------------------------------------------------------
// Shard: a node's local view of the universe
// ---------------------------------------------------------------------------

/// A disjoint shard of the universe held by one distributed node.
///
/// In the Sharded-Universe Model (SUM), $U = S_1 \sqcup \cdots \sqcup S_m$
/// and each node stores only the block IDs for its elements. The `offset`
/// field records where this shard begins in the global element ordering.
///
/// # Construction
///
/// ```
/// use partition_lattice::Partition;
/// use partition_lattice::distributed::Shard;
///
/// let pi = Partition::from_ids(vec![0, 0, 1, 1]);
/// let sigma = Partition::from_ids(vec![0, 1, 0, 1]);
/// let shard = Shard::from_partitions(&pi, &sigma, 0, 2);
/// assert_eq!(shard.len(), 2);
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Shard {
    /// Global element offset: this shard covers elements `[offset, offset + len)`.
    pub offset: usize,
    /// Block IDs of π restricted to this shard.
    pub pi_ids: Vec<u32>,
    /// Block IDs of σ restricted to this shard.
    pub sigma_ids: Vec<u32>,
}

impl Shard {
    /// Create a shard from full partitions by slicing `[offset, offset + len)`.
    pub fn from_partitions(pi: &Partition, sigma: &Partition, offset: usize, len: usize) -> Self {
        let pi_all = pi.get_ids();
        let sigma_all = sigma.get_ids();
        Shard {
            offset,
            pi_ids: pi_all[offset..offset + len].to_vec(),
            sigma_ids: sigma_all[offset..offset + len].to_vec(),
        }
    }

    /// Create a shard directly from raw block-ID slices (zero-copy on the node).
    pub fn from_raw(offset: usize, pi_ids: Vec<u32>, sigma_ids: Vec<u32>) -> Self {
        assert_eq!(
            pi_ids.len(),
            sigma_ids.len(),
            "pi_ids and sigma_ids must have the same length"
        );
        Shard {
            offset,
            pi_ids,
            sigma_ids,
        }
    }

    /// Number of elements in this shard.
    pub fn len(&self) -> usize {
        self.pi_ids.len()
    }

    /// Whether this shard is empty.
    pub fn is_empty(&self) -> bool {
        self.pi_ids.is_empty()
    }
}

// ---------------------------------------------------------------------------
// MAP-phase message types (what flows over the network)
// ---------------------------------------------------------------------------

/// MAP output for Protocol 1: per-block element counts for one partition.
///
/// `bytes_transferred` tracks the serialized size for communication analysis.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlockCountMsg {
    /// `block_id → count` within this shard.
    pub counts: HashMap<u32, usize>,
    /// Estimated serialized size in bytes.
    pub bytes_transferred: usize,
}

/// MAP output for Protocol 2: composite `(π-block, σ-block) → count`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompositeCountMsg {
    /// `(block_π, block_σ) → count` within this shard.
    pub counts: HashMap<(u32, u32), usize>,
    /// Estimated serialized size in bytes.
    pub bytes_transferred: usize,
}

/// MAP output for Protocol 5: block co-occurrence edges and per-block sizes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlockGraphMsg {
    /// Unique (π-block, σ-block) edges observed in this shard.
    pub edges: HashSet<(u32, u32)>,
    /// `π-block → local element count`.
    pub pi_sizes: HashMap<u32, usize>,
    /// `σ-block → local element count`.
    pub sigma_sizes: HashMap<u32, usize>,
    /// Estimated serialized size in bytes.
    pub bytes_transferred: usize,
}

// ---------------------------------------------------------------------------
// MAP phase: local computation on each shard
// ---------------------------------------------------------------------------

/// P1-MAP: Compute block counts for one partition on this shard.
///
/// If `use_pi` is true, counts π-blocks; otherwise σ-blocks.
/// Uses dense array counting for O(1) per element (vs HashMap O(hash) overhead).
/// Communication cost: $O(K)$ entries × 12 bytes each.
pub fn map_block_counts(shard: &Shard, use_pi: bool) -> BlockCountMsg {
    let ids = if use_pi {
        &shard.pi_ids
    } else {
        &shard.sigma_ids
    };
    // Find max block ID to size dense array
    let max_id = ids.iter().copied().max().unwrap_or(0) as usize;
    let mut dense = vec![0usize; max_id + 1];
    for &bid in ids {
        dense[bid as usize] += 1;
    }
    // Convert non-zero entries to HashMap for wire format
    let counts: HashMap<u32, usize> = dense
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(id, &c)| (id as u32, c))
        .collect();
    let bytes = counts.len() * 12;
    BlockCountMsg {
        counts,
        bytes_transferred: bytes,
    }
}

/// P2-MAP: Compute composite block-pair counts on this shard.
///
/// Each entry `(block_π, block_σ) → count` represents how many elements
/// in this shard belong to that specific block pair.
/// Uses flat 2D array `[pi_block * stride + sigma_block]` for O(1) per element.
/// Communication cost: $O(K_1 \cdot K_2)$ entries × 16 bytes each.
pub fn map_composite_counts(shard: &Shard) -> CompositeCountMsg {
    let n = shard.len();
    if n == 0 {
        return CompositeCountMsg {
            counts: HashMap::new(),
            bytes_transferred: 0,
        };
    }
    let max_pi = shard.pi_ids.iter().copied().max().unwrap_or(0) as usize;
    let max_sigma = shard.sigma_ids.iter().copied().max().unwrap_or(0) as usize;
    let stride = max_sigma + 1;
    let mut dense = vec![0usize; (max_pi + 1) * stride];
    for i in 0..n {
        let idx = shard.pi_ids[i] as usize * stride + shard.sigma_ids[i] as usize;
        dense[idx] += 1;
    }
    // Convert non-zero entries to HashMap for wire format
    let counts: HashMap<(u32, u32), usize> = dense
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(idx, &c)| ((idx / stride) as u32, (idx % stride) as u32, c))
        .map(|(pi, si, c)| ((pi, si), c))
        .collect();
    let bytes = counts.len() * 16;
    CompositeCountMsg {
        counts,
        bytes_transferred: bytes,
    }
}

/// P5-MAP: Compute block co-occurrence edges and block sizes on this shard.
///
/// Returns the bipartite graph edges plus per-block sizes for Union-Find reduce.
/// Communication cost: $O(K_1 \cdot K_2 + K_1 + K_2)$.
pub fn map_block_graph(shard: &Shard) -> BlockGraphMsg {
    let mut edges: HashSet<(u32, u32)> = HashSet::new();
    let mut pi_sizes: HashMap<u32, usize> = HashMap::new();
    let mut sigma_sizes: HashMap<u32, usize> = HashMap::new();

    for i in 0..shard.len() {
        let (pb, sb) = (shard.pi_ids[i], shard.sigma_ids[i]);
        edges.insert((pb, sb));
        *pi_sizes.entry(pb).or_insert(0) += 1;
        *sigma_sizes.entry(sb).or_insert(0) += 1;
    }

    let bytes = edges.len() * 8 + pi_sizes.len() * 12 + sigma_sizes.len() * 12;
    BlockGraphMsg {
        edges,
        pi_sizes,
        sigma_sizes,
        bytes_transferred: bytes,
    }
}

// ---------------------------------------------------------------------------
// REDUCE phase: coordinator merges results from all shards
// ---------------------------------------------------------------------------

/// P1-REDUCE: Merge block counts and compute $h(\pi)$.
///
/// Aggregates `HashMap<BlockId, count>` from all shards, then computes
/// $h(\pi) = 1 - \sum_B (|B| / N)^2$.
pub fn reduce_entropy(msgs: &[BlockCountMsg], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut global: HashMap<u32, usize> = HashMap::new();
    for msg in msgs {
        for (&bid, &count) in &msg.counts {
            *global.entry(bid).or_insert(0) += count;
        }
    }
    let n_f = n as f64;
    let sum_sq: f64 = global.values().map(|&c| (c as f64 / n_f).powi(2)).sum();
    1.0 - sum_sq
}

/// P2-REDUCE: Merge composite counts and compute $h(\pi \wedge \sigma)$.
///
/// Aggregates `HashMap<(BlockId, BlockId), count>` from all shards, then
/// computes $h(\pi \wedge \sigma) = 1 - \sum_{(b_1,b_2)} (\text{count}/N)^2$.
pub fn reduce_refine_entropy(msgs: &[CompositeCountMsg], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut global: HashMap<(u32, u32), usize> = HashMap::new();
    for msg in msgs {
        for (&key, &count) in &msg.counts {
            *global.entry(key).or_insert(0) += count;
        }
    }
    let n_f = n as f64;
    let sum_sq: f64 = global.values().map(|&c| (c as f64 / n_f).powi(2)).sum();
    1.0 - sum_sq
}

/// P5-REDUCE: Merge block-graph edges and compute $h(\pi \vee \sigma)$
/// via Union-Find on $K_1 + K_2$ block nodes.
///
/// This exploits the meet/join duality: both operations share the same
/// block co-occurrence matrix, but join uses connected components
/// (transitive closure) instead of cells (product).
pub fn reduce_coarsen_entropy(msgs: &[BlockGraphMsg], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }

    // 1. Collect all unique blocks and merge sizes/edges
    let mut pi_set: HashSet<u32> = HashSet::new();
    let mut sigma_set: HashSet<u32> = HashSet::new();
    let mut all_edges: HashSet<(u32, u32)> = HashSet::new();
    let mut global_pi: HashMap<u32, usize> = HashMap::new();
    let mut global_sigma: HashMap<u32, usize> = HashMap::new();

    for msg in msgs {
        for &(pb, sb) in &msg.edges {
            all_edges.insert((pb, sb));
            pi_set.insert(pb);
            sigma_set.insert(sb);
        }
        for (&b, &s) in &msg.pi_sizes {
            *global_pi.entry(b).or_insert(0) += s;
            pi_set.insert(b);
        }
        for (&b, &s) in &msg.sigma_sizes {
            *global_sigma.entry(b).or_insert(0) += s;
            sigma_set.insert(b);
        }
    }

    // 2. Dense indexing: π-blocks [0, k1), σ-blocks [k1, k1+k2)
    let pi_blocks: Vec<u32> = pi_set.into_iter().collect();
    let sigma_blocks: Vec<u32> = sigma_set.into_iter().collect();
    let k1 = pi_blocks.len();
    let total = k1 + sigma_blocks.len();

    let pi_idx: HashMap<u32, usize> = pi_blocks.iter().enumerate().map(|(i, &b)| (b, i)).collect();
    let sigma_idx: HashMap<u32, usize> = sigma_blocks
        .iter()
        .enumerate()
        .map(|(i, &b)| (b, k1 + i))
        .collect();

    // 3. Union-Find on K₁ + K₂ nodes
    let mut parent: Vec<usize> = (0..total).collect();
    let mut rank: Vec<u8> = vec![0; total];

    for &(pb, sb) in &all_edges {
        uf_union(&mut parent, &mut rank, pi_idx[&pb], sigma_idx[&sb]);
    }

    // 4. Aggregate join-component sizes via π-block sizes
    let mut component_sizes: HashMap<usize, usize> = HashMap::new();
    for (&b, &size) in &global_pi {
        let root = uf_find(&mut parent, pi_idx[&b]);
        *component_sizes.entry(root).or_insert(0) += size;
    }

    let n_f = n as f64;
    let sum_sq: f64 = component_sizes
        .values()
        .map(|&c| (c as f64 / n_f).powi(2))
        .sum();
    1.0 - sum_sq
}

// ---------------------------------------------------------------------------
// High-level distributed pipelines
// ---------------------------------------------------------------------------

/// Shard a universe into `m` approximately-equal shards from two partitions.
///
/// Returns `m` [`Shard`]s whose `pi_ids`/`sigma_ids` cover the full universe.
/// If `N` is not divisible by `m`, the first `N % m` shards get one extra element.
///
/// # Panics
///
/// Panics if the two partitions have different sizes or `m == 0`.
///
/// # Example
///
/// ```
/// use partition_lattice::Partition;
/// use partition_lattice::distributed::shard_universe;
///
/// let pi = Partition::from_ids(vec![0, 0, 1, 1, 2, 2]);
/// let sigma = Partition::from_ids(vec![0, 1, 0, 1, 0, 1]);
/// let shards = shard_universe(&pi, &sigma, 3);
/// assert_eq!(shards.len(), 3);
/// assert_eq!(shards.iter().map(|s| s.len()).sum::<usize>(), 6);
/// ```
pub fn shard_universe(pi: &Partition, sigma: &Partition, m: usize) -> Vec<Shard> {
    let n = pi.size();
    assert_eq!(n, sigma.size(), "partitions must have same universe size");
    assert!(m > 0, "must have at least 1 shard");

    let chunk = n / m;
    let remainder = n % m;

    let mut shards = Vec::with_capacity(m);
    let mut offset = 0;
    for i in 0..m {
        let len = chunk + if i < remainder { 1 } else { 0 };
        shards.push(Shard::from_partitions(pi, sigma, offset, len));
        offset += len;
    }
    shards
}

/// Shard universe with **weighted** distribution.
///
/// Each worker receives elements proportional to its weight.
/// `weights[i]` should reflect worker i's relative processing speed
/// (e.g., CPU core count). Equal weights produce identical results
/// to [`shard_universe`].
///
/// # Panics
/// - If `weights` is empty.
/// - If `pi` and `sigma` have different sizes.
///
/// # Example
/// ```
/// use partition_lattice::Partition;
/// use partition_lattice::distributed::shard_universe_weighted;
///
/// let pi = Partition::from_ids(vec![0, 0, 1, 1, 2, 2]);
/// let sigma = Partition::from_ids(vec![0, 1, 0, 1, 0, 1]);
/// let shards = shard_universe_weighted(&pi, &sigma, &[2.0, 1.0]);
/// assert_eq!(shards[0].len(), 4); // 2/3 of 6
/// assert_eq!(shards[1].len(), 2); // 1/3 of 6
/// ```
pub fn shard_universe_weighted(pi: &Partition, sigma: &Partition, weights: &[f64]) -> Vec<Shard> {
    let n = pi.size();
    assert_eq!(n, sigma.size(), "partitions must have same universe size");
    assert!(!weights.is_empty(), "must have at least 1 weight");

    let m = weights.len();
    let total_weight: f64 = weights.iter().sum();
    assert!(total_weight > 0.0, "total weight must be positive");

    let mut shards = Vec::with_capacity(m);
    let mut offset = 0;
    for i in 0..m {
        let len = if i == m - 1 {
            n - offset // last worker gets remainder
        } else {
            ((n as f64) * (weights[i] / total_weight)).round() as usize
        };
        shards.push(Shard::from_partitions(pi, sigma, offset, len));
        offset += len;
    }
    shards
}

/// Distributed $h(\pi)$: Protocol 1 full pipeline.
///
/// Returns `(entropy, total_bytes_transferred)`.
pub fn distributed_entropy(shards: &[Shard], n: usize, use_pi: bool) -> (f64, usize) {
    let msgs: Vec<BlockCountMsg> = shards.iter().map(|s| map_block_counts(s, use_pi)).collect();
    let total_bytes: usize = msgs.iter().map(|m| m.bytes_transferred).sum();
    (reduce_entropy(&msgs, n), total_bytes)
}

/// Distributed $h(\pi \wedge \sigma)$: Protocol 2 full pipeline.
///
/// Returns `(meet_entropy, total_bytes_transferred)`.
pub fn distributed_refine_entropy(shards: &[Shard], n: usize) -> (f64, usize) {
    let msgs: Vec<CompositeCountMsg> = shards.iter().map(|s| map_composite_counts(s)).collect();
    let total_bytes: usize = msgs.iter().map(|m| m.bytes_transferred).sum();
    (reduce_refine_entropy(&msgs, n), total_bytes)
}

/// Distributed $d_{\oplus}(\pi, \sigma)$: Protocol 3 (derived from P1 + P2).
///
/// Returns `(ditset_xor_distance, total_bytes_transferred)`.
pub fn distributed_ditset_xor_distance(shards: &[Shard], n: usize) -> (f64, usize) {
    let (h_pi, b1) = distributed_entropy(shards, n, true);
    let (h_sigma, b2) = distributed_entropy(shards, n, false);
    let (h_meet, b3) = distributed_refine_entropy(shards, n);
    (2.0 * h_meet - h_pi - h_sigma, b1 + b2 + b3)
}

/// Distributed $D_L(\pi \| \sigma)$: Protocol 3 (derived).
///
/// Returns `(logical_divergence, total_bytes_transferred)`.
pub fn distributed_logical_divergence(shards: &[Shard], n: usize) -> (f64, usize) {
    let (h_sigma, b1) = distributed_entropy(shards, n, false);
    let (h_meet, b2) = distributed_refine_entropy(shards, n);
    (h_meet - h_sigma, b1 + b2)
}

/// Distributed $I(\pi; \sigma)$: Protocol 3 (derived).
///
/// Returns `(mutual_information, total_bytes_transferred)`.
pub fn distributed_mutual_information(shards: &[Shard], n: usize) -> (f64, usize) {
    let (h_pi, b1) = distributed_entropy(shards, n, true);
    let (h_sigma, b2) = distributed_entropy(shards, n, false);
    let (h_meet, b3) = distributed_refine_entropy(shards, n);
    (h_pi + h_sigma - h_meet, b1 + b2 + b3)
}

/// Distributed $h(\pi \vee \sigma)$: Protocol 5 (block-graph MapReduce).
///
/// Returns `(join_entropy, total_bytes_transferred)`.
pub fn distributed_coarsen_entropy(shards: &[Shard], n: usize) -> (f64, usize) {
    let msgs: Vec<BlockGraphMsg> = shards.iter().map(|s| map_block_graph(s)).collect();
    let total_bytes: usize = msgs.iter().map(|m| m.bytes_transferred).sum();
    (reduce_coarsen_entropy(&msgs, n), total_bytes)
}

// ---------------------------------------------------------------------------
// Union-Find helpers (internal, shared with streaming.rs pattern)
// ---------------------------------------------------------------------------

fn uf_find(parent: &mut [usize], x: usize) -> usize {
    if parent[x] != x {
        parent[x] = uf_find(parent, parent[x]);
    }
    parent[x]
}

fn uf_union(parent: &mut [usize], rank: &mut [u8], x: usize, y: usize) {
    let rx = uf_find(parent, x);
    let ry = uf_find(parent, y);
    if rx == ry {
        return;
    }
    if rank[rx] < rank[ry] {
        parent[rx] = ry;
    } else if rank[rx] > rank[ry] {
        parent[ry] = rx;
    } else {
        parent[ry] = rx;
        rank[rx] += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn regular_partitions(n: usize, k1: usize, k2: usize) -> (Partition, Partition) {
        let pi = Partition::from_ids((0..n).map(|i| (i % k1) as u32).collect());
        let sigma = Partition::from_ids((0..n).map(|i| (i % k2) as u32).collect());
        (pi, sigma)
    }

    fn random_partitions(n: usize, k1: usize, k2: usize, seed: u64) -> (Partition, Partition) {
        let mut s = seed;
        let mut next = || -> u64 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s >> 33
        };
        let pi = Partition::from_ids((0..n).map(|_| (next() % k1 as u64) as u32).collect());
        let sigma = Partition::from_ids((0..n).map(|_| (next() % k2 as u64) as u32).collect());
        (pi, sigma)
    }

    // ── Protocol 1: h(π) ────────────────────────────────────────────

    #[test]
    fn test_distributed_entropy_exact() {
        let (pi, sigma) = regular_partitions(10_000, 50, 80);
        for m in [2, 4, 8] {
            let shards = shard_universe(&pi, &sigma, m);
            let (dh, _) = distributed_entropy(&shards, pi.size(), true);
            assert!((dh - pi.logical_entropy()).abs() < EPS, "m={m}");
        }
    }

    #[test]
    fn test_distributed_entropy_random() {
        let (pi, sigma) = random_partitions(10_000, 100, 60, 42);
        for m in [2, 3, 5, 7] {
            let shards = shard_universe(&pi, &sigma, m);
            let (dh, _) = distributed_entropy(&shards, pi.size(), true);
            assert!((dh - pi.logical_entropy()).abs() < EPS, "m={m}");
        }
    }

    // ── Protocol 2: h(π ∧ σ) ────────────────────────────────────────

    #[test]
    fn test_distributed_meet_entropy_exact() {
        let (pi, sigma) = regular_partitions(10_000, 50, 80);
        for m in [2, 4, 8] {
            let shards = shard_universe(&pi, &sigma, m);
            let (dh, _) = distributed_refine_entropy(&shards, pi.size());
            let mono = pi.fused_entropy_differentiate(&sigma);
            assert!((dh - mono).abs() < EPS, "m={m}");
        }
    }

    #[test]
    fn test_distributed_meet_entropy_random() {
        let (pi, sigma) = random_partitions(10_000, 100, 60, 123);
        for m in [2, 4, 8] {
            let shards = shard_universe(&pi, &sigma, m);
            let (dh, _) = distributed_refine_entropy(&shards, pi.size());
            let mono = pi.fused_entropy_differentiate(&sigma);
            assert!((dh - mono).abs() < EPS, "m={m}");
        }
    }

    // ── Protocol 3: d_⊕, D_L, I ────────────────────────────────────

    #[test]
    fn test_distributed_d_xor_exact() {
        let (pi, sigma) = regular_partitions(10_000, 50, 80);
        for m in [2, 4, 8] {
            let shards = shard_universe(&pi, &sigma, m);
            let (dd, _) = distributed_ditset_xor_distance(&shards, pi.size());
            assert!((dd - pi.ditset_xor_distance(&sigma)).abs() < EPS, "m={m}");
        }
    }

    #[test]
    fn test_distributed_divergence_exact() {
        let (pi, sigma) = regular_partitions(10_000, 50, 80);
        for m in [2, 4] {
            let shards = shard_universe(&pi, &sigma, m);
            let (dd, _) = distributed_logical_divergence(&shards, pi.size());
            assert!((dd - pi.logical_divergence(&sigma)).abs() < EPS, "m={m}");
        }
    }

    #[test]
    fn test_distributed_mi_exact() {
        let (pi, sigma) = regular_partitions(10_000, 50, 80);
        for m in [2, 4] {
            let shards = shard_universe(&pi, &sigma, m);
            let (dm, _) = distributed_mutual_information(&shards, pi.size());
            assert!((dm - pi.mutual_information(&sigma)).abs() < EPS, "m={m}");
        }
    }

    // ── Protocol 5: h(π ∨ σ) ────────────────────────────────────────

    #[test]
    fn test_distributed_join_entropy_exact() {
        let (pi, sigma) = regular_partitions(10_000, 50, 80);
        for m in [2, 4, 8] {
            let shards = shard_universe(&pi, &sigma, m);
            let (dh, _) = distributed_coarsen_entropy(&shards, pi.size());
            let mono = pi.fused_entropy_integrate(&sigma);
            assert!((dh - mono).abs() < EPS, "m={m}");
        }
    }

    #[test]
    fn test_distributed_join_entropy_random() {
        let (pi, sigma) = random_partitions(10_000, 30, 40, 777);
        for m in [2, 4, 8] {
            let shards = shard_universe(&pi, &sigma, m);
            let (dh, _) = distributed_coarsen_entropy(&shards, pi.size());
            let mono = pi.fused_entropy_integrate(&sigma);
            assert!((dh - mono).abs() < EPS, "m={m}");
        }
    }

    // ── Communication cost analysis ──────────────────────────────────

    #[test]
    fn test_communication_bounded() {
        let n = 100_000;
        let k1 = 100;
        let k2 = 80;
        let (pi, sigma) = regular_partitions(n, k1, k2);
        for m in [2, 4, 8] {
            let shards = shard_universe(&pi, &sigma, m);
            let (_, bytes) = distributed_refine_entropy(&shards, n);
            let data_size = n * 4;
            let ratio = bytes as f64 / data_size as f64;
            assert!(ratio < 0.5, "m={m}: ratio {ratio:.4} >= 0.5 (K2 violation)");
        }
    }

    #[test]
    fn test_communication_weak_scaling() {
        let k1 = 50;
        let k2 = 40;
        let m = 4;
        let mut prev = f64::MAX;
        for &n in &[1_000, 10_000, 100_000] {
            let (pi, sigma) = regular_partitions(n, k1, k2);
            let shards = shard_universe(&pi, &sigma, m);
            let (_, bytes) = distributed_refine_entropy(&shards, n);
            let ratio = bytes as f64 / (n * 4) as f64;
            assert!(ratio < prev, "N={n}: not decreasing");
            prev = ratio;
        }
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn test_single_shard_equals_monolithic() {
        let (pi, sigma) = regular_partitions(1_000, 10, 20);
        let shards = shard_universe(&pi, &sigma, 1);
        let (dh, _) = distributed_entropy(&shards, pi.size(), true);
        let (dm, _) = distributed_refine_entropy(&shards, pi.size());
        let (dj, _) = distributed_coarsen_entropy(&shards, pi.size());
        assert!((dh - pi.logical_entropy()).abs() < EPS);
        assert!((dm - pi.fused_entropy_differentiate(&sigma)).abs() < EPS);
        assert!((dj - pi.fused_entropy_integrate(&sigma)).abs() < EPS);
    }

    #[test]
    fn test_discrete_partition() {
        let n = 100;
        let pi = Partition::discrete(n);
        let sigma = Partition::discrete(n);
        let shards = shard_universe(&pi, &sigma, 4);
        let (dh, _) = distributed_entropy(&shards, n, true);
        assert!((dh - pi.logical_entropy()).abs() < EPS);
    }

    #[test]
    fn test_indiscrete_partition() {
        let n = 100;
        let pi = Partition::indiscrete(n);
        let sigma = Partition::indiscrete(n);
        let shards = shard_universe(&pi, &sigma, 4);
        let (dh, _) = distributed_entropy(&shards, n, true);
        assert!(dh.abs() < EPS, "h(⊤) should be 0");
    }

    #[test]
    fn test_unequal_shard_sizes() {
        let n = 101;
        let (pi, sigma) = regular_partitions(n, 10, 15);
        let shards = shard_universe(&pi, &sigma, 4);
        let total: usize = shards.iter().map(|s| s.len()).sum();
        assert_eq!(total, n);
        let (dm, _) = distributed_refine_entropy(&shards, n);
        let mono = pi.fused_entropy_differentiate(&sigma);
        assert!((dm - mono).abs() < EPS);
    }

    #[test]
    fn test_empty_partition() {
        let pi = Partition::from_ids(vec![]);
        let sigma = Partition::from_ids(vec![]);
        let shards = shard_universe(&pi, &sigma, 1);
        let (dh, _) = distributed_entropy(&shards, 0, true);
        assert_eq!(dh, 0.0);
    }

    #[test]
    fn test_shard_from_raw() {
        let shard = Shard::from_raw(100, vec![0, 0, 1, 1], vec![0, 1, 0, 1]);
        assert_eq!(shard.offset, 100);
        assert_eq!(shard.len(), 4);
        let msg = map_composite_counts(&shard);
        assert_eq!(msg.counts.len(), 4); // all 4 combinations
    }

    // ── Weighted sharding ──────────────────────────────────

    #[test]
    fn test_shard_universe_weighted_equal_weights() {
        // Equal weights must produce identical results to shard_universe()
        let n = 1000;
        let (pi, sigma) = regular_partitions(n, 10, 15);
        let equal = shard_universe(&pi, &sigma, 4);
        let weighted = shard_universe_weighted(&pi, &sigma, &[1.0, 1.0, 1.0, 1.0]);
        for (e, w) in equal.iter().zip(weighted.iter()) {
            assert_eq!(e.len(), w.len(), "equal weights must match equal split");
        }
    }

    #[test]
    fn test_shard_universe_weighted_proportional() {
        // weights [2.0, 1.0] → first worker gets ~67%, second ~33%
        let n = 900;
        let (pi, sigma) = regular_partitions(n, 10, 15);
        let shards = shard_universe_weighted(&pi, &sigma, &[2.0, 1.0]);
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].len(), 600); // 2/3 of 900
        assert_eq!(shards[1].len(), 300); // 1/3 of 900
        assert_eq!(shards[0].len() + shards[1].len(), n);
    }

    #[test]
    fn test_shard_universe_weighted_cores_ratio() {
        // Simulate imac(10 cores) + mini32(8 cores)
        let n = 100_000;
        let (pi, sigma) = regular_partitions(n, 50, 30);
        let shards = shard_universe_weighted(&pi, &sigma, &[10.0, 8.0]);
        assert_eq!(shards.len(), 2);
        // 10/(10+8) = 55.56%, 8/(10+8) = 44.44%
        assert_eq!(shards[0].len() + shards[1].len(), n);
        assert!(shards[0].len() > shards[1].len(), "higher weight gets more");
        let ratio = shards[0].len() as f64 / n as f64;
        assert!(
            (ratio - 10.0 / 18.0).abs() < 0.01,
            "ratio ~55.6%: got {:.3}",
            ratio
        );
    }

    #[test]
    fn test_shard_universe_weighted_accuracy() {
        // Weighted sharding must produce identical entropy results
        let n = 10_000;
        let (pi, sigma) = random_partitions(n, 50, 30, 42);
        let mono = pi.fused_entropy_differentiate(&sigma);

        // Equal split baseline
        let equal_shards = shard_universe(&pi, &sigma, 3);
        let (h_equal, _) = distributed_refine_entropy(&equal_shards, n);

        // Weighted split
        let weighted_shards = shard_universe_weighted(&pi, &sigma, &[3.0, 2.0, 1.0]);
        let (h_weighted, _) = distributed_refine_entropy(&weighted_shards, n);

        assert!(
            (h_equal - mono).abs() < EPS,
            "equal: {} vs mono: {}",
            h_equal,
            mono
        );
        assert!(
            (h_weighted - mono).abs() < EPS,
            "weighted: {} vs mono: {}",
            h_weighted,
            mono
        );
    }

    #[test]
    fn test_shard_universe_weighted_single_worker() {
        let n = 500;
        let (pi, sigma) = regular_partitions(n, 5, 8);
        let shards = shard_universe_weighted(&pi, &sigma, &[1.0]);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].len(), n);
    }
}
