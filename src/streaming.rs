//! Streaming partition operations for large |U|.
//!
//! These functions compute partition lattice metrics without materializing
//! intermediate results, achieving O(K) or O(K₁×K₂) peak memory instead
//! of O(|U|). This enables accurate computation on universes that exceed
//! FastPartition's 8MB BLOCKS_MEMORY_LIMIT.
//!
//! # Supported Operations
//!
//! | Operation | Memory | Accuracy |
//! |-----------|--------|----------|
//! | [`streaming_entropy`] | O(K) | exact |
//! | [`fused_refine_entropy`] | O(K₁×K₂) | exact |
//! | [`streaming_ditset_xor_distance`] | O(K₁+K₂+K₁K₂) | exact |
//! | [`block_graph_coarsen_entropy`] | O(K₁×K₂) | exact |
//!
//! # Mathematical Basis
//!
//! Meet (∧) and Join (∨) share the same K₁×K₂ block co-occurrence matrix M,
//! but interpret it differently:
//! - **Meet blocks** = non-zero cells of M (product)
//! - **Join blocks** = connected components of the bipartite graph induced by M
//!
//! This duality enables O(K₁K₂)-memory computation for both operations,
//! with Union-Find running on K₁+K₂ block nodes instead of N elements.
//!
//! # Example
//!
//! ```
//! use partition_lattice::Partition;
//! use partition_lattice::streaming;
//!
//! let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2]);
//! let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1]);
//!
//! // Compute h(p ∧ q) without materializing the meet
//! let h = streaming::fused_refine_entropy(&p, &q, 2);
//! assert!((h - p.differentiate(&q).logical_entropy()).abs() < 1e-12);
//!
//! // Compute h(p ∨ q) without materializing the join
//! let hj = streaming::block_graph_coarsen_entropy(&p, &q, 2);
//! assert!((hj - p.integrate(&q).logical_entropy()).abs() < 1e-12);
//! ```
//!
//! > **Doctest note.** The per-function examples below are marked `ignore` (illustrative
//! > only). Their numerical correctness is exercised by the `#[test]` suite at the bottom of
//! > this file (`streaming_entropy` / `fused_refine_entropy` / … `_matches_direct`), which uses
//! > the same API. They are `ignore`d because rustdoc compiles one separate binary per runnable
//! > example, and this crate is a wide fan-in dependency — so the workspace doctest phase paid
//! > ~1.7 s × N of link cost (and hit the 60 s "slow test" marker under full-suite load). The
//! > runnable module example above is the compile-checked entry point.

#[cfg(not(feature = "parallel"))]
use crate::par_shim::IntoParShim;
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
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Compute logical entropy via streaming block-size accumulation.
///
/// Iterates through block IDs in chunk-sized windows, accumulating per-block
/// counts, then computes h(π) = 1 - Σ(|Bⱼ|/N)².
///
/// Memory: O(K) where K = number of distinct blocks (independent of |U|).
///
/// # Examples
///
/// ```ignore
/// use partition_lattice::{Partition, streaming};
/// let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 3, 3, 3]);
/// let h = streaming::streaming_entropy(&p, 4);
/// assert!((h - p.logical_entropy()).abs() < 1e-12);
/// ```
/// Compute h(π) directly from raw block ID array without constructing a Partition.
///
/// This avoids the O(N·K) cost of `Partition::from_ids()` (BitVec construction),
/// making it ideal for GPU pipelines where block IDs are already in `Vec<u32>`.
///
/// Cost: O(N) scan + O(K) sum-of-squares. Pure integer arithmetic except final division.
///
/// # Use case
///
/// On Apple M4 unified memory, this function at N=100K runs in **33µs** vs
/// GPU entropy kernel at **473µs** (14× faster), because GPU dispatch overhead
/// (~130µs) exceeds the CPU histogram scan cost.
///
/// # Examples
///
/// ```ignore
/// use partition_lattice::streaming;
/// let ids = vec![0u32, 0, 1, 1, 2, 3, 3, 3];
/// let h = streaming::entropy_from_ids(&ids);
/// assert!((h - 0.71875).abs() < 1e-12); // same as Partition::from_ids(ids).logical_entropy()
/// ```
pub fn entropy_from_ids(ids: &[u32]) -> f64 {
    let n = ids.len();
    if n <= 1 {
        return 0.0;
    }

    // O(N): count block sizes using HashMap for sparse/large IDs (GPU meet output)
    let mut counts: HashMap<u32, u64> = HashMap::new();
    for &id in ids {
        *counts.entry(id).or_default() += 1;
    }

    // O(K): compute sum of squares — integer arithmetic (DCE-T4 compliant)
    let sum_sq: u64 = counts.values().map(|&c| c * c).sum();

    // Final: h = 1 - Σ|B|² / N² (single FP division at boundary, LBP-4)
    let n2 = (n as u64) * (n as u64);
    1.0 - sum_sq as f64 / n2 as f64
}

pub fn streaming_entropy(partition: &Partition, chunk_size: usize) -> f64 {
    let ids = partition.get_ids();
    let n = ids.len();
    if n == 0 {
        return 0.0;
    }

    let mut block_counts: HashMap<u32, usize> = HashMap::new();
    for chunk in ids.chunks(chunk_size.max(1)) {
        for &bid in chunk {
            *block_counts.entry(bid).or_default() += 1;
        }
    }

    let n_f = n as f64;
    let sum_sq: f64 = block_counts.values().map(|&c| (c as f64).powi(2)).sum();
    1.0 - sum_sq / (n_f * n_f)
}

/// Compute h(π ∧ σ) without materializing the meet partition.
///
/// Streams through both partitions in parallel chunks, creating composite
/// block keys (block_π, block_σ) and accumulating counts.
///
/// This is typically **faster** than `p.differentiate(&q).logical_entropy()`
/// because it avoids allocating the meet partition entirely. At large |U|
/// with many blocks, it also avoids FastPartition's 8MB cap lossy reduction.
///
/// Memory: O(K₁ × K₂) where K₁, K₂ are block counts.
///
/// # Examples
///
/// ```ignore
/// use partition_lattice::{Partition, streaming};
/// let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3]);
/// let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1]);
/// let h = streaming::fused_refine_entropy(&p, &q, 4);
/// let direct = p.differentiate(&q).logical_entropy();
/// assert!((h - direct).abs() < 1e-12);
/// ```
///
/// # Panics
///
/// Panics if the two partitions have different universe sizes.
pub fn fused_refine_entropy(p: &Partition, q: &Partition, chunk_size: usize) -> f64 {
    assert_eq!(
        p.size(),
        q.size(),
        "Partitions must have same universe size"
    );
    let n = p.size();
    if n == 0 {
        return 0.0;
    }

    let p_ids = p.get_ids();
    let q_ids = q.get_ids();
    let chunk = chunk_size.max(1);
    let mut meet_counts: HashMap<(u32, u32), usize> = HashMap::new();

    for start in (0..n).step_by(chunk) {
        let end = (start + chunk).min(n);
        for i in start..end {
            let key = (p_ids[i], q_ids[i]);
            *meet_counts.entry(key).or_default() += 1;
        }
    }

    let n_f = n as f64;
    let sum_sq: f64 = meet_counts.values().map(|&c| (c as f64).powi(2)).sum();
    1.0 - sum_sq / (n_f * n_f)
}

/// Compute d_⊕(π, σ) = 2·h(π ∧ σ) − h(π) − h(σ) in a single pass.
///
/// Computes all three entropies simultaneously by accumulating block counts
/// for π, σ, and π∧σ in one traversal. This avoids building any intermediate
/// partitions.
///
/// Memory: O(K₁ + K₂ + K₁×K₂).
///
/// # Examples
///
/// ```ignore
/// use partition_lattice::{Partition, streaming};
/// let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3]);
/// let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1]);
/// let d = streaming::streaming_ditset_xor_distance(&p, &q, 4);
/// let direct = p.ditset_xor_distance(&q);
/// assert!((d - direct).abs() < 1e-12);
/// ```
///
/// # Panics
///
/// Panics if the two partitions have different universe sizes.
pub fn streaming_ditset_xor_distance(p: &Partition, q: &Partition, chunk_size: usize) -> f64 {
    assert_eq!(
        p.size(),
        q.size(),
        "Partitions must have same universe size"
    );
    let n = p.size();
    if n == 0 {
        return 0.0;
    }

    let p_ids = p.get_ids();
    let q_ids = q.get_ids();
    let chunk = chunk_size.max(1);

    let mut p_counts: HashMap<u32, usize> = HashMap::new();
    let mut q_counts: HashMap<u32, usize> = HashMap::new();
    let mut meet_counts: HashMap<(u32, u32), usize> = HashMap::new();

    for start in (0..n).step_by(chunk) {
        let end = (start + chunk).min(n);
        for i in start..end {
            *p_counts.entry(p_ids[i]).or_default() += 1;
            *q_counts.entry(q_ids[i]).or_default() += 1;
            *meet_counts.entry((p_ids[i], q_ids[i])).or_default() += 1;
        }
    }

    let n_sq = (n as f64) * (n as f64);
    let h_p = 1.0 - p_counts.values().map(|&c| (c as f64).powi(2)).sum::<f64>() / n_sq;
    let h_q = 1.0 - q_counts.values().map(|&c| (c as f64).powi(2)).sum::<f64>() / n_sq;
    let h_meet = 1.0
        - meet_counts
            .values()
            .map(|&c| (c as f64).powi(2))
            .sum::<f64>()
            / n_sq;

    2.0 * h_meet - h_p - h_q
}

/// Compute h(π ∨ σ) without materializing the join partition.
///
/// Builds the bipartite block co-occurrence graph in a single pass,
/// then finds connected components via Union-Find on K₁+K₂ block nodes.
///
/// This exploits the meet/join duality: both operations decompose through
/// the same K₁×K₂ block co-occurrence matrix M, but meet uses cells as
/// blocks (product) while join uses connected components (transitive closure).
///
/// Memory: O(K₁ × K₂) — independent of |U|.
///
/// # Examples
///
/// ```ignore
/// use partition_lattice::{Partition, streaming};
/// let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3]);
/// let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1]);
/// let h = streaming::block_graph_coarsen_entropy(&p, &q, 4);
/// let direct = p.integrate(&q).logical_entropy();
/// assert!((h - direct).abs() < 1e-12);
/// ```
///
/// # Panics
///
/// Panics if the two partitions have different universe sizes.
pub fn block_graph_coarsen_entropy(p: &Partition, q: &Partition, chunk_size: usize) -> f64 {
    assert_eq!(
        p.size(),
        q.size(),
        "Partitions must have same universe size"
    );
    let n = p.size();
    if n == 0 {
        return 0.0;
    }

    let p_ids = p.get_ids();
    let q_ids = q.get_ids();
    let chunk = chunk_size.max(1);

    // Single pass: accumulate edge set and π-block sizes
    let mut edges: HashMap<(u32, u32), ()> = HashMap::new();
    let mut p_block_sizes: HashMap<u32, usize> = HashMap::new();

    for start in (0..n).step_by(chunk) {
        let end = (start + chunk).min(n);
        for i in start..end {
            edges.entry((p_ids[i], q_ids[i])).or_default();
            *p_block_sizes.entry(p_ids[i]).or_default() += 1;
        }
    }

    // Collect unique block IDs → contiguous indices
    let p_blocks: Vec<u32> = {
        let mut v: Vec<u32> = p_block_sizes.keys().copied().collect();
        v.sort();
        v
    };
    let q_blocks: Vec<u32> = {
        let mut set: HashMap<u32, ()> = HashMap::new();
        for &(_, qb) in edges.keys() {
            set.entry(qb).or_default();
        }
        let mut v: Vec<u32> = set.keys().copied().collect();
        v.sort();
        v
    };

    let k1 = p_blocks.len();
    let total = k1 + q_blocks.len();

    // Map block IDs → UF indices: π-blocks [0..k1), σ-blocks [k1..total)
    let p_idx: HashMap<u32, usize> = p_blocks.iter().enumerate().map(|(i, &b)| (b, i)).collect();
    let q_idx: HashMap<u32, usize> = q_blocks
        .iter()
        .enumerate()
        .map(|(i, &b)| (b, k1 + i))
        .collect();

    // Union-Find on K₁+K₂ block nodes
    let mut parent: Vec<usize> = (0..total).collect();
    let mut uf_rank: Vec<u8> = vec![0; total];

    for &(pb, qb) in edges.keys() {
        uf_union(&mut parent, &mut uf_rank, p_idx[&pb], q_idx[&qb]);
    }

    // Aggregate join-block sizes from π-block sizes grouped by component
    let mut component_sizes: HashMap<usize, usize> = HashMap::new();
    for (&block_id, &size) in &p_block_sizes {
        let root = uf_find(&mut parent, p_idx[&block_id]);
        *component_sizes.entry(root).or_default() += size;
    }

    let n_f = n as f64;
    let sum_sq: f64 = component_sizes
        .values()
        .map(|&s| (s as f64 / n_f).powi(2))
        .sum();
    1.0 - sum_sq
}

// ── Union-Find helpers (internal) ──────────────────────────────────

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

// ── Parallel streaming ─────────────────────────────────────
//
// Par-shard MapReduce on shared memory: each rayon chunk builds a core-local
// accumulator, then a single reduce merges them. Structurally identical to
// `distributed.rs` protocols P1–P5, but with zero communication overhead.
//
// N_min threshold: parallel path is used only when N ≥ PAR_N_MIN (200K).
// Below this, rayon thread-pool overhead exceeds any parallelism benefit.

/// Minimum universe size for parallel streaming to outperform sequential.
/// Determined experimentally: crossover at N ≈ 200K on 10-core M4.
const PAR_N_MIN: usize = 200_000;

/// Parallel streaming h(π) via per-core block count accumulation.
///
/// Falls back to sequential [`streaming_entropy`] when N < 200K.
///
/// # Examples
///
/// ```ignore
/// use partition_lattice::{Partition, streaming};
/// let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 3, 3, 3]);
/// let h = streaming::par_streaming_entropy(&p);
/// assert!((h - p.logical_entropy()).abs() < 1e-12);
/// ```
pub fn par_streaming_entropy(partition: &Partition) -> f64 {
    let ids = partition.get_ids();
    let n = ids.len();
    if n < PAR_N_MIN {
        return streaming_entropy(partition, (n / 4).max(1));
    }

    let chunk_size = par_chunk_size(n);
    let mapped = make_ranges(n, chunk_size)
        .into_par_iter()
        .map(|(start, end)| {
            let mut local: HashMap<u32, usize> = HashMap::new();
            for i in start..end {
                *local.entry(ids[i]).or_default() += 1;
            }
            local
        });
    // ★ par/seq terminal split (feature="parallel"): rayon reduce(id, op) ≡ sequential
    //   fold(id(), op) — associative merge, bit-exact.
    #[cfg(feature = "parallel")]
    let global_counts: HashMap<u32, usize> = mapped.reduce(HashMap::new, merge_counts);
    #[cfg(not(feature = "parallel"))]
    let global_counts: HashMap<u32, usize> = mapped.fold(HashMap::new(), merge_counts);

    let n_f = n as f64;
    1.0 - global_counts
        .values()
        .map(|&c| (c as f64).powi(2))
        .sum::<f64>()
        / (n_f * n_f)
}

/// Parallel streaming h(π ∧ σ) via per-core composite block count accumulation.
///
/// Falls back to sequential [`fused_refine_entropy`] when N < 200K.
///
/// # Examples
///
/// ```ignore
/// use partition_lattice::{Partition, streaming};
/// let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3]);
/// let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1]);
/// let h = streaming::par_fused_refine_entropy(&p, &q);
/// assert!((h - p.differentiate(&q).logical_entropy()).abs() < 1e-12);
/// ```
///
/// # Panics
///
/// Panics if the two partitions have different universe sizes.
pub fn par_fused_refine_entropy(p: &Partition, q: &Partition) -> f64 {
    assert_eq!(
        p.size(),
        q.size(),
        "Partitions must have same universe size"
    );
    let n = p.size();
    if n < PAR_N_MIN {
        return fused_refine_entropy(p, q, (n / 4).max(1));
    }

    let p_ids = p.get_ids();
    let q_ids = q.get_ids();
    let chunk_size = par_chunk_size(n);

    let mapped = make_ranges(n, chunk_size)
        .into_par_iter()
        .map(|(start, end)| {
            let mut local: HashMap<(u32, u32), usize> = HashMap::new();
            for i in start..end {
                *local.entry((p_ids[i], q_ids[i])).or_default() += 1;
            }
            local
        });
    #[cfg(feature = "parallel")]
    let global_counts: HashMap<(u32, u32), usize> = mapped.reduce(HashMap::new, merge_counts);
    #[cfg(not(feature = "parallel"))]
    let global_counts: HashMap<(u32, u32), usize> = mapped.fold(HashMap::new(), merge_counts);

    let n_f = n as f64;
    1.0 - global_counts
        .values()
        .map(|&c| (c as f64).powi(2))
        .sum::<f64>()
        / (n_f * n_f)
}

/// Parallel streaming d_⊕(π, σ) = 2·h(π∧σ) − h(π) − h(σ) in a single pass.
///
/// Falls back to sequential [`streaming_ditset_xor_distance`] when N < 200K.
///
/// # Examples
///
/// ```ignore
/// use partition_lattice::{Partition, streaming};
/// let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3]);
/// let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1]);
/// let d = streaming::par_streaming_ditset_xor_distance(&p, &q);
/// assert!((d - p.ditset_xor_distance(&q)).abs() < 1e-12);
/// ```
///
/// # Panics
///
/// Panics if the two partitions have different universe sizes.
pub fn par_streaming_ditset_xor_distance(p: &Partition, q: &Partition) -> f64 {
    assert_eq!(
        p.size(),
        q.size(),
        "Partitions must have same universe size"
    );
    let n = p.size();
    if n < PAR_N_MIN {
        return streaming_ditset_xor_distance(p, q, (n / 4).max(1));
    }

    let p_ids = p.get_ids();
    let q_ids = q.get_ids();
    let chunk_size = par_chunk_size(n);

    let mapped = make_ranges(n, chunk_size)
        .into_par_iter()
        .map(|(start, end)| {
            let mut pc: HashMap<u32, usize> = HashMap::new();
            let mut qc: HashMap<u32, usize> = HashMap::new();
            let mut mc: HashMap<(u32, u32), usize> = HashMap::new();
            for i in start..end {
                *pc.entry(p_ids[i]).or_default() += 1;
                *qc.entry(q_ids[i]).or_default() += 1;
                *mc.entry((p_ids[i], q_ids[i])).or_default() += 1;
            }
            (pc, qc, mc)
        });
    // par/seq terminal split — associative 3-tuple merge, bit-exact.
    #[cfg(feature = "parallel")]
    let (p_counts, q_counts, meet_counts) = mapped.reduce(
        || (HashMap::new(), HashMap::new(), HashMap::new()),
        |(mut ap, mut aq, mut am), (bp, bq, bm)| {
            for (k, v) in bp {
                *ap.entry(k).or_default() += v;
            }
            for (k, v) in bq {
                *aq.entry(k).or_default() += v;
            }
            for (k, v) in bm {
                *am.entry(k).or_default() += v;
            }
            (ap, aq, am)
        },
    );
    #[cfg(not(feature = "parallel"))]
    let (p_counts, q_counts, meet_counts) = mapped.fold(
        (
            HashMap::<u32, usize>::new(),
            HashMap::<u32, usize>::new(),
            HashMap::<(u32, u32), usize>::new(),
        ),
        |(mut ap, mut aq, mut am), (bp, bq, bm)| {
            for (k, v) in bp {
                *ap.entry(k).or_default() += v;
            }
            for (k, v) in bq {
                *aq.entry(k).or_default() += v;
            }
            for (k, v) in bm {
                *am.entry(k).or_default() += v;
            }
            (ap, aq, am)
        },
    );

    let n_sq = (n as f64) * (n as f64);
    let h_p = 1.0 - p_counts.values().map(|&c| (c as f64).powi(2)).sum::<f64>() / n_sq;
    let h_q = 1.0 - q_counts.values().map(|&c| (c as f64).powi(2)).sum::<f64>() / n_sq;
    let h_m = 1.0
        - meet_counts
            .values()
            .map(|&c| (c as f64).powi(2))
            .sum::<f64>()
            / n_sq;
    2.0 * h_m - h_p - h_q
}

/// Parallel streaming h(π ∨ σ) via block-graph Union-Find.
///
/// Falls back to sequential [`block_graph_coarsen_entropy`] when N < 200K.
///
/// # Examples
///
/// ```ignore
/// use partition_lattice::{Partition, streaming};
/// let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3]);
/// let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1]);
/// let h = streaming::par_block_graph_coarsen_entropy(&p, &q);
/// assert!((h - p.integrate(&q).logical_entropy()).abs() < 1e-12);
/// ```
///
/// # Panics
///
/// Panics if the two partitions have different universe sizes.
pub fn par_block_graph_coarsen_entropy(p: &Partition, q: &Partition) -> f64 {
    assert_eq!(
        p.size(),
        q.size(),
        "Partitions must have same universe size"
    );
    let n = p.size();
    if n < PAR_N_MIN {
        return block_graph_coarsen_entropy(p, q, (n / 4).max(1));
    }

    let p_ids = p.get_ids();
    let q_ids = q.get_ids();
    let chunk_size = par_chunk_size(n);

    let mapped = make_ranges(n, chunk_size)
        .into_par_iter()
        .map(|(start, end)| {
            let mut e: HashSet<(u32, u32)> = HashSet::new();
            let mut ps: HashMap<u32, usize> = HashMap::new();
            for i in start..end {
                e.insert((p_ids[i], q_ids[i]));
                *ps.entry(p_ids[i]).or_default() += 1;
            }
            (e, ps)
        });
    // par/seq terminal split — associative (edge-set ∪, size-map +) merge, bit-exact.
    #[cfg(feature = "parallel")]
    let (edges, pi_sizes) = mapped.reduce(
        || (HashSet::new(), HashMap::new()),
        |(mut ae, mut ap), (be, bp)| {
            ae.extend(be);
            for (k, v) in bp {
                *ap.entry(k).or_default() += v;
            }
            (ae, ap)
        },
    );
    #[cfg(not(feature = "parallel"))]
    let (edges, pi_sizes) = mapped.fold(
        (HashSet::<(u32, u32)>::new(), HashMap::<u32, usize>::new()),
        |(mut ae, mut ap), (be, bp)| {
            ae.extend(be);
            for (k, v) in bp {
                *ap.entry(k).or_default() += v;
            }
            (ae, ap)
        },
    );

    // Dense index assignment
    let pi_blocks: Vec<u32> = pi_sizes.keys().copied().collect();
    let pi_idx: HashMap<u32, usize> = pi_blocks.iter().enumerate().map(|(i, &b)| (b, i)).collect();
    let k1 = pi_blocks.len();
    let sigma_blocks: HashSet<u32> = edges.iter().map(|&(_, s)| s).collect();
    let sigma_idx: HashMap<u32, usize> = sigma_blocks
        .iter()
        .enumerate()
        .map(|(i, &b)| (b, i))
        .collect();
    let k2 = sigma_blocks.len();

    // Union-Find on K₁ + K₂ nodes
    let total = k1 + k2;
    let mut parent: Vec<usize> = (0..total).collect();
    let mut rank: Vec<u8> = vec![0; total];
    for &(pb, sb) in &edges {
        uf_union(&mut parent, &mut rank, pi_idx[&pb], k1 + sigma_idx[&sb]);
    }

    let mut comp_sizes: HashMap<usize, usize> = HashMap::new();
    for (&block, &size) in &pi_sizes {
        let root = uf_find(&mut parent, pi_idx[&block]);
        *comp_sizes.entry(root).or_default() += size;
    }

    let n_f = n as f64;
    1.0 - comp_sizes
        .values()
        .map(|&s| (s as f64).powi(2))
        .sum::<f64>()
        / (n_f * n_f)
}

/// All seven partition metrics from a single parallel pass.
#[derive(Debug, Clone)]
pub struct AllMetrics {
    /// h(π)
    pub h_pi: f64,
    /// h(σ)
    pub h_sigma: f64,
    /// h(π ∧ σ)
    pub h_meet: f64,
    /// h(π ∨ σ)
    pub h_join: f64,
    /// d_⊕(π, σ) = 2·h(π∧σ) − h(π) − h(σ)
    pub d_xor: f64,
    /// D_L(π ∥ σ) = h(π∧σ) − h(σ)
    pub d_l_pi_sigma: f64,
    /// D_L(σ ∥ π) = h(π∧σ) − h(π)
    pub d_l_sigma_pi: f64,
    /// I(π; σ) = h(π) + h(σ) − h(π∧σ)
    pub mutual_info: f64,
}

/// Compute all 7 partition metrics in a single parallel pass.
///
/// This is the most efficient entry point for computing multiple metrics:
/// one traversal produces all results with per-core accumulation.
///
/// # Examples
///
/// ```ignore
/// use partition_lattice::{Partition, streaming};
/// let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2]);
/// let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1]);
/// let m = streaming::par_all_metrics(&p, &q);
/// assert!(m.h_pi >= 0.0);
/// assert!(m.h_meet >= m.h_pi);
/// ```
///
/// # Panics
///
/// Panics if the two partitions have different universe sizes.
pub fn par_all_metrics(p: &Partition, q: &Partition) -> AllMetrics {
    assert_eq!(
        p.size(),
        q.size(),
        "Partitions must have same universe size"
    );
    let n = p.size();
    if n == 0 {
        return AllMetrics {
            h_pi: 0.0,
            h_sigma: 0.0,
            h_meet: 0.0,
            h_join: 0.0,
            d_xor: 0.0,
            d_l_pi_sigma: 0.0,
            d_l_sigma_pi: 0.0,
            mutual_info: 0.0,
        };
    }

    let p_ids = p.get_ids();
    let q_ids = q.get_ids();
    let chunk_size = par_chunk_size(n);

    let mapped = make_ranges(n, chunk_size)
        .into_par_iter()
        .map(|(start, end)| {
            let mut pc: HashMap<u32, usize> = HashMap::new();
            let mut sc: HashMap<u32, usize> = HashMap::new();
            let mut mc: HashMap<(u32, u32), usize> = HashMap::new();
            let mut e: HashSet<(u32, u32)> = HashSet::new();
            for i in start..end {
                let pid = p_ids[i];
                let qid = q_ids[i];
                *pc.entry(pid).or_default() += 1;
                *sc.entry(qid).or_default() += 1;
                *mc.entry((pid, qid)).or_default() += 1;
                e.insert((pid, qid));
            }
            (pc, sc, mc, e)
        });
    // par/seq terminal split — associative 4-tuple merge, bit-exact.
    #[cfg(feature = "parallel")]
    let (pi_counts, sigma_counts, meet_counts, edges) = mapped.reduce(
        || {
            (
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashSet::new(),
            )
        },
        |(mut ap, mut as_, mut am, mut ae), (bp, bs, bm, be)| {
            for (k, v) in bp {
                *ap.entry(k).or_default() += v;
            }
            for (k, v) in bs {
                *as_.entry(k).or_default() += v;
            }
            for (k, v) in bm {
                *am.entry(k).or_default() += v;
            }
            ae.extend(be);
            (ap, as_, am, ae)
        },
    );
    #[cfg(not(feature = "parallel"))]
    let (pi_counts, sigma_counts, meet_counts, edges) = mapped.fold(
        (
            HashMap::<u32, usize>::new(),
            HashMap::<u32, usize>::new(),
            HashMap::<(u32, u32), usize>::new(),
            HashSet::<(u32, u32)>::new(),
        ),
        |(mut ap, mut as_, mut am, mut ae), (bp, bs, bm, be)| {
            for (k, v) in bp {
                *ap.entry(k).or_default() += v;
            }
            for (k, v) in bs {
                *as_.entry(k).or_default() += v;
            }
            for (k, v) in bm {
                *am.entry(k).or_default() += v;
            }
            ae.extend(be);
            (ap, as_, am, ae)
        },
    );

    let n_sq = (n as f64) * (n as f64);
    let h_pi = 1.0 - pi_counts.values().map(|&c| (c as f64).powi(2)).sum::<f64>() / n_sq;
    let h_sigma = 1.0
        - sigma_counts
            .values()
            .map(|&c| (c as f64).powi(2))
            .sum::<f64>()
            / n_sq;
    let h_meet = 1.0
        - meet_counts
            .values()
            .map(|&c| (c as f64).powi(2))
            .sum::<f64>()
            / n_sq;

    // Join via Union-Find on block graph
    let pi_blocks: Vec<u32> = pi_counts.keys().copied().collect();
    let pi_idx: HashMap<u32, usize> = pi_blocks.iter().enumerate().map(|(i, &b)| (b, i)).collect();
    let k1 = pi_blocks.len();
    let sigma_blocks: HashSet<u32> = edges.iter().map(|&(_, s)| s).collect();
    let sigma_idx: HashMap<u32, usize> = sigma_blocks
        .iter()
        .enumerate()
        .map(|(i, &b)| (b, i))
        .collect();
    let k2 = sigma_blocks.len();

    let total = k1 + k2;
    let mut parent: Vec<usize> = (0..total).collect();
    let mut rank: Vec<u8> = vec![0; total];
    for &(pb, sb) in &edges {
        uf_union(&mut parent, &mut rank, pi_idx[&pb], k1 + sigma_idx[&sb]);
    }

    let mut comp_sizes: HashMap<usize, usize> = HashMap::new();
    for (&block, &size) in &pi_counts {
        let root = uf_find(&mut parent, pi_idx[&block]);
        *comp_sizes.entry(root).or_default() += size;
    }

    let n_f = n as f64;
    let h_join = 1.0
        - comp_sizes
            .values()
            .map(|&s| (s as f64).powi(2))
            .sum::<f64>()
            / (n_f * n_f);

    let d_xor = 2.0 * h_meet - h_pi - h_sigma;
    let d_l_pi_sigma = h_meet - h_sigma;
    let d_l_sigma_pi = h_meet - h_pi;
    let mutual_info = h_pi + h_sigma - h_meet;

    AllMetrics {
        h_pi,
        h_sigma,
        h_meet,
        h_join,
        d_xor,
        d_l_pi_sigma,
        d_l_sigma_pi,
        mutual_info,
    }
}

// ── Parallel helpers ────────────────────────────────────────────────

fn par_chunk_size(n: usize) -> usize {
    #[cfg(feature = "parallel")]
    let threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let threads = 1usize;
    (n / threads).max(1024)
}

fn make_ranges(n: usize, chunk_size: usize) -> Vec<(usize, usize)> {
    (0..n)
        .step_by(chunk_size)
        .map(|start| (start, (start + chunk_size).min(n)))
        .collect()
}

fn merge_counts<K: Eq + core::hash::Hash>(
    mut a: HashMap<K, usize>,
    b: HashMap<K, usize>,
) -> HashMap<K, usize> {
    for (k, v) in b {
        *a.entry(k).or_default() += v;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn test_streaming_entropy_matches_direct() {
        let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 3, 3, 3]);
        let direct = p.logical_entropy();
        for cs in [1, 2, 3, 4, 8] {
            let s = streaming_entropy(&p, cs);
            assert!((direct - s).abs() < EPS, "chunk={cs}: {direct} vs {s}");
        }
    }

    #[test]
    fn test_streaming_entropy_empty() {
        let p = Partition::from_ids(vec![]);
        assert_eq!(streaming_entropy(&p, 10), 0.0);
    }

    #[test]
    fn test_streaming_entropy_discrete() {
        let n = 100;
        let p = Partition::from_ids((0..n as u32).collect());
        let expected = 1.0 - 1.0 / n as f64;
        assert!((streaming_entropy(&p, 10) - expected).abs() < EPS);
    }

    #[test]
    fn test_fused_meet_entropy_matches_direct() {
        let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1]);
        let direct = p.differentiate(&q).logical_entropy();
        for cs in [1, 2, 4, 8] {
            let f = fused_refine_entropy(&p, &q, cs);
            assert!((direct - f).abs() < EPS, "chunk={cs}: {direct} vs {f}");
        }
    }

    #[test]
    fn test_fused_meet_entropy_self() {
        let p = Partition::from_ids(vec![0, 0, 1, 2, 2, 2]);
        let direct = p.logical_entropy();
        let fused = fused_refine_entropy(&p, &p, 3);
        assert!((direct - fused).abs() < EPS);
    }

    #[test]
    fn test_streaming_d_xor_matches_direct() {
        let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1]);
        let direct = p.ditset_xor_distance(&q);
        for cs in [1, 2, 4, 8] {
            let s = streaming_ditset_xor_distance(&p, &q, cs);
            assert!((direct - s).abs() < EPS, "chunk={cs}: {direct} vs {s}");
        }
    }

    #[test]
    fn test_streaming_d_xor_self_is_zero() {
        let p = Partition::from_ids(vec![0, 0, 1, 2, 2, 2]);
        let d = streaming_ditset_xor_distance(&p, &p, 3);
        assert!(d.abs() < EPS, "self d_⊕ should be 0, got {d}");
    }

    // ── block_graph_coarsen_entropy ─────────────────────────────────

    #[test]
    fn test_bg_join_entropy_matches_direct() {
        let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1]);
        let direct = p.integrate(&q).logical_entropy();
        for cs in [1, 2, 4, 8] {
            let bg = block_graph_coarsen_entropy(&p, &q, cs);
            assert!((direct - bg).abs() < EPS, "chunk={cs}: {direct} vs {bg}");
        }
    }

    #[test]
    fn test_bg_join_entropy_self() {
        let p = Partition::from_ids(vec![0, 0, 1, 2, 2, 2]);
        let direct = p.logical_entropy();
        let bg = block_graph_coarsen_entropy(&p, &p, 3);
        assert!((direct - bg).abs() < EPS);
    }

    #[test]
    fn test_bg_join_entropy_transitive_chain() {
        // P = {0,1}{2,3}{4,5}  Q = {0}{1,2}{3,4}{5}
        // P∨Q: chain 0↔1↔2↔3↔4↔5 → single block, h=0
        let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2]);
        let q = Partition::from_ids(vec![0, 1, 1, 2, 2, 3]);
        let bg = block_graph_coarsen_entropy(&p, &q, 2);
        assert!(bg.abs() < EPS, "chain: expected 0, got {bg}");
    }

    #[test]
    fn test_bg_join_entropy_empty() {
        let p = Partition::from_ids(vec![]);
        let q = Partition::from_ids(vec![]);
        assert_eq!(block_graph_coarsen_entropy(&p, &q, 10), 0.0);
    }

    #[test]
    fn test_bg_join_entropy_indiscrete() {
        let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2]);
        let q = Partition::indiscrete(6);
        let bg = block_graph_coarsen_entropy(&p, &q, 3);
        assert!(bg.abs() < EPS, "indiscrete join: {bg}");
    }

    #[test]
    fn test_bg_join_entropy_10k() {
        let n = 10_000;
        let p = Partition::from_ids((0..n as u32).map(|i| i % 20).collect());
        let q = Partition::from_ids((0..n as u32).map(|i| i % 15).collect());
        // block_graph uses exact join entropy, compare with integrate_exact
        let direct = p.integrate_exact(&q).logical_entropy();
        let bg = block_graph_coarsen_entropy(&p, &q, 1_000);
        assert!((direct - bg).abs() < EPS, "10k: {direct} vs {bg}");
    }

    // ── par_streaming ──────────────────────────────────

    #[test]
    fn test_par_fused_meet_entropy_matches_direct() {
        let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1]);
        let direct = p.differentiate(&q).logical_entropy();
        let par = par_fused_refine_entropy(&p, &q);
        assert!((direct - par).abs() < EPS, "par meet: {direct} vs {par}");
    }

    #[test]
    fn test_par_fused_meet_entropy_empty() {
        let p = Partition::from_ids(vec![]);
        let q = Partition::from_ids(vec![]);
        assert_eq!(par_fused_refine_entropy(&p, &q), 0.0);
    }

    #[test]
    fn test_par_fused_meet_entropy_self() {
        let p = Partition::from_ids(vec![0, 0, 1, 2, 2, 2]);
        let direct = p.logical_entropy();
        let par = par_fused_refine_entropy(&p, &p);
        assert!((direct - par).abs() < EPS);
    }

    #[test]
    fn test_par_streaming_entropy_matches_direct() {
        let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 3, 3, 3]);
        let direct = p.logical_entropy();
        let par = par_streaming_entropy(&p);
        assert!((direct - par).abs() < EPS, "par h: {direct} vs {par}");
    }

    #[test]
    fn test_par_streaming_d_xor_matches_direct() {
        let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1]);
        let direct = p.ditset_xor_distance(&q);
        let par = par_streaming_ditset_xor_distance(&p, &q);
        assert!((direct - par).abs() < EPS, "par d_⊕: {direct} vs {par}");
    }

    #[test]
    fn test_par_streaming_d_xor_self_is_zero() {
        let p = Partition::from_ids(vec![0, 0, 1, 2, 2, 2]);
        let d = par_streaming_ditset_xor_distance(&p, &p);
        assert!(d.abs() < EPS, "par self d_⊕ should be 0, got {d}");
    }

    #[test]
    fn test_par_block_graph_join_entropy_matches_direct() {
        let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1]);
        let direct = p.integrate(&q).logical_entropy();
        let par = par_block_graph_coarsen_entropy(&p, &q);
        assert!((direct - par).abs() < EPS, "par join: {direct} vs {par}");
    }

    #[test]
    fn test_par_block_graph_join_entropy_transitive_chain() {
        let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2]);
        let q = Partition::from_ids(vec![0, 1, 1, 2, 2, 3]);
        let par = par_block_graph_coarsen_entropy(&p, &q);
        assert!(par.abs() < EPS, "par chain join: expected 0, got {par}");
    }

    #[test]
    fn test_par_all_metrics_consistency() {
        let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4]);
        let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1, 2, 2]);
        let m = par_all_metrics(&p, &q);

        // Cross-check with sequential
        assert!((m.h_pi - streaming_entropy(&p, 10)).abs() < EPS, "h_pi");
        assert!(
            (m.h_sigma - streaming_entropy(&q, 10)).abs() < EPS,
            "h_sigma"
        );
        assert!(
            (m.h_meet - fused_refine_entropy(&p, &q, 10)).abs() < EPS,
            "h_meet"
        );
        assert!(
            (m.h_join - block_graph_coarsen_entropy(&p, &q, 10)).abs() < EPS,
            "h_join"
        );

        // Algebraic identities
        assert!(
            (m.d_xor - (2.0 * m.h_meet - m.h_pi - m.h_sigma)).abs() < EPS,
            "d_⊕"
        );
        assert!(
            (m.d_l_pi_sigma - (m.h_meet - m.h_sigma)).abs() < EPS,
            "D_L(π||σ)"
        );
        assert!(
            (m.mutual_info - (m.h_pi + m.h_sigma - m.h_meet)).abs() < EPS,
            "I"
        );

        // Lattice ordering
        assert!(m.h_meet >= m.h_pi - EPS, "h(meet) ≥ h(π)");
        assert!(m.h_meet >= m.h_sigma - EPS, "h(meet) ≥ h(σ)");
        assert!(m.h_join <= m.h_pi + EPS, "h(join) ≤ h(π)");
        assert!(m.h_join <= m.h_sigma + EPS, "h(join) ≤ h(σ)");
    }

    #[test]
    fn test_streaming_10k_scale() {
        let n = 10_000;
        let p = Partition::from_ids((0..n as u32).map(|i| i % 30).collect());
        let q = Partition::from_ids((0..n as u32).map(|i| i % 20).collect());

        let direct_h = p.logical_entropy();
        assert!((streaming_entropy(&p, 1_000) - direct_h).abs() < EPS);

        let direct_mh = p.differentiate(&q).logical_entropy();
        assert!((fused_refine_entropy(&p, &q, 1_000) - direct_mh).abs() < EPS);

        let direct_d = p.ditset_xor_distance(&q);
        assert!((streaming_ditset_xor_distance(&p, &q, 1_000) - direct_d).abs() < EPS);
    }

    // ── entropy_from_ids tests ──

    #[test]
    fn test_entropy_from_ids_matches_partition() {
        let ids = vec![0u32, 0, 1, 1, 2, 3, 3, 3];
        let p = Partition::from_ids(ids.clone());
        let from_ids = entropy_from_ids(&ids);
        assert!(
            (from_ids - p.logical_entropy()).abs() < EPS,
            "entropy_from_ids={from_ids} vs Partition::logical_entropy()={}",
            p.logical_entropy()
        );
    }

    #[test]
    fn test_entropy_from_ids_empty() {
        assert_eq!(entropy_from_ids(&[]), 0.0);
    }

    #[test]
    fn test_entropy_from_ids_single() {
        assert_eq!(entropy_from_ids(&[0]), 0.0);
    }

    #[test]
    fn test_entropy_from_ids_discrete() {
        let n = 100u32;
        let ids: Vec<u32> = (0..n).collect();
        let expected = 1.0 - 1.0 / n as f64;
        assert!((entropy_from_ids(&ids) - expected).abs() < EPS);
    }

    #[test]
    fn test_entropy_from_ids_indiscrete() {
        let ids = vec![0u32; 50];
        assert_eq!(entropy_from_ids(&ids), 0.0);
    }

    #[test]
    fn test_entropy_from_ids_large_sparse() {
        // IDs with large gaps (e.g., GPU meet output: a*K+b)
        let ids = vec![0, 100, 200, 100, 0, 300, 200, 0];
        let p = Partition::from_ids(ids.clone());
        let from_ids = entropy_from_ids(&ids);
        assert!((from_ids - p.logical_entropy()).abs() < EPS);
    }
}
