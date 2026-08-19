//! Partition algebra expressiveness
//!
//! Bell number computation, exhaustive partition enumeration, and
//! reachability analysis for Meet/Join closure over generator sets.
//!
//! # Verified Properties (exp50)
//! - Meet/Join closure from generators reaches 100% of Π(U) for N≤5
//! - Bell number B_4=15, B_5=52 confirmed
//! - XOR requires atomic partitions (not just projections)

use crate::unified::Partition;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::FloatExt;
use crate::HashSet;
#[allow(unused_imports)]
use alloc::borrow::ToOwned;
#[allow(unused_imports)]
use alloc::boxed::Box;
use alloc::collections::VecDeque;
#[allow(unused_imports)]
use alloc::string::{String, ToString};
#[allow(unused_imports)]
use alloc::vec::Vec;

/// Compute the n-th Bell number using the Bell triangle.
///
/// B(n) = |Π(U)| where |U| = n.
pub fn bell_number(n: usize) -> u64 {
    if n == 0 {
        return 1;
    }
    let mut prev = vec![1u64];
    for i in 1..n {
        let mut curr = vec![prev[i - 1]];
        for j in 1..=i {
            curr.push(curr[j - 1] + prev[j - 1]);
        }
        prev = curr;
    }
    prev[n - 1]
}

/// Enumerate all partitions of {0, 1, ..., n-1} as block-ID vectors.
///
/// Each partition is represented as `Vec<usize>` where `ids[i]` is element i's block.
/// Warning: grows as B(n) — only feasible for small n (≤ 7).
pub fn enumerate_all_partitions(n: usize) -> Vec<Vec<u32>> {
    let mut results = Vec::new();
    let mut ids = vec![0u32; n];
    enumerate_helper(&mut ids, 0, 0, n, &mut results);
    results
}

fn enumerate_helper(
    ids: &mut Vec<u32>,
    pos: usize,
    max_id: u32,
    n: usize,
    results: &mut Vec<Vec<u32>>,
) {
    if pos == n {
        results.push(ids.clone());
        return;
    }
    for block_id in 0..=max_id {
        ids[pos] = block_id;
        let new_max = if block_id == max_id {
            max_id + 1
        } else {
            max_id
        };
        enumerate_helper(ids, pos + 1, new_max, n, results);
    }
}

/// Result of a Meet/Join reachability analysis.
#[derive(Debug, Clone)]
pub struct ReachabilityResult {
    /// Total partitions in Π(U)
    pub total_partitions: usize,
    /// Partitions reachable via Meet/Join from generators
    pub reachable_count: usize,
    /// Reachability ratio (should be 1.0 with complete generators)
    pub reachability_ratio: f64,
    /// Maximum generation depth needed
    pub max_depth: usize,
}

/// Normalize a partition's IDs via re-encoding through Partition.
fn partition_key(ids: &[u32]) -> Vec<u32> {
    Partition::from_ids(ids.to_vec()).get_ids().to_vec()
}

/// Check what fraction of Π(U) is reachable from generators
/// {discrete, indiscrete, atomic 2-block partitions} via Meet/Join.
///
/// Uses BFS over the partition space. Only feasible for small n (≤ 6).
pub fn check_reachability(n: usize) -> ReachabilityResult {
    let all_partitions = enumerate_all_partitions(n);
    let total = all_partitions.len();

    let all_keys: HashSet<Vec<u32>> = all_partitions
        .iter()
        .map(|ids| partition_key(ids))
        .collect();

    let discrete: Vec<u32> = (0..n as u32).collect();
    let indiscrete = vec![0u32; n];

    let mut reached: HashSet<Vec<u32>> = HashSet::new();
    let mut frontier: VecDeque<(Vec<u32>, usize)> = VecDeque::new();
    let mut max_depth = 0usize;

    // Seed with generators
    let dk = partition_key(&discrete);
    let ik = partition_key(&indiscrete);
    reached.insert(dk.clone());
    reached.insert(ik.clone());
    frontier.push_back((dk, 0));
    frontier.push_back((ik, 0));

    // Atomic 2-block partitions: {i} vs rest
    for i in 0..n {
        let mut ids = vec![0u32; n];
        ids[i] = 1;
        let key = partition_key(&ids);
        if reached.insert(key.clone()) {
            frontier.push_back((key, 0));
        }
    }

    while let Some((current, depth)) = frontier.pop_front() {
        if reached.len() == total {
            break;
        }
        if depth > 2 * n {
            continue;
        }

        let p = Partition::from_ids(current.clone());

        let reached_snapshot: Vec<Vec<u32>> = reached.iter().cloned().collect();
        for other_key in &reached_snapshot {
            let q = Partition::from_ids(other_key.clone());

            let meet = p.differentiate(&q);
            let mk = partition_key(meet.get_ids());
            if all_keys.contains(&mk) && reached.insert(mk.clone()) {
                max_depth = max_depth.max(depth + 1);
                frontier.push_back((mk, depth + 1));
            }

            let join = p.integrate(&q);
            let jk = partition_key(join.get_ids());
            if all_keys.contains(&jk) && reached.insert(jk.clone()) {
                max_depth = max_depth.max(depth + 1);
                frontier.push_back((jk, depth + 1));
            }
        }
    }

    ReachabilityResult {
        total_partitions: total,
        reachable_count: reached.len(),
        reachability_ratio: reached.len() as f64 / total as f64,
        max_depth,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expressiveness_bell_numbers() {
        assert_eq!(bell_number(0), 1);
        assert_eq!(bell_number(1), 1);
        assert_eq!(bell_number(2), 2);
        assert_eq!(bell_number(3), 5);
        assert_eq!(bell_number(4), 15);
        assert_eq!(bell_number(5), 52);
    }

    #[test]
    fn test_expressiveness_enumerate_count_matches_bell() {
        for n in 1..=5 {
            let partitions = enumerate_all_partitions(n);
            assert_eq!(
                partitions.len(),
                bell_number(n) as usize,
                "enumeration count should match B({})",
                n
            );
        }
    }

    #[test]
    fn test_expressiveness_reachability_n4() {
        let result = check_reachability(4);
        assert_eq!(result.total_partitions, 15);
        assert_eq!(result.reachable_count, 15);
        assert!((result.reachability_ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_expressiveness_enumerate_produces_unique_partitions() {
        let partitions = enumerate_all_partitions(4);
        let keys: HashSet<Vec<u32>> = partitions.iter().map(|ids| partition_key(ids)).collect();
        assert_eq!(keys.len(), 15);
    }
}
