//! Join (∨, `integrate`) algorithms for the partition lattice.
//!
//! Currently hosts the CPU ShallowJoin implementation. Extracted from
//! `unified.rs` as a self-contained pure algorithm; the `CoarsenStrategy` enum
//! that selects between Shallow/Exact lives alongside the `Partition` dispatch
//! types in `unified.rs`.

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

/// CPU ShallowJoin with double-buffer symmetric propagation.
///
/// Performs a bounded-depth Join of two partitions by:
/// 1. Building initial labels: elements in the same block (in either partition)
///    point to their block's first-seen element index.
/// 2. Double-buffer label propagation for `depth` rounds:
///    `new[i] = old[old[i]]` — read isolation prevents asymmetric propagation.
/// 3. Normalizing labels to contiguous block IDs.
///
/// # Arguments
/// * `ids_a` - Block IDs of partition π
/// * `ids_b` - Block IDs of partition σ
/// * `depth` - Number of propagation rounds (recommended: 2)
///
/// # Complexity
/// * Time: O(N × depth)
/// * Memory: O(2N) (double buffer)
///
/// # DCE compliance
/// All operations are u32 integer — no FP involved.
/// CPU ShallowJoin with depth-bounded path compression.
///
/// Uses Union-Find to build the correct parent structure, then
/// limits find() to `depth` rounds of path compression instead of
/// converging to root. Double-buffer is applied in the compression
/// phase to ensure symmetric propagation.
///
/// # Arguments
/// * `ids_a` - Block IDs of partition π
/// * `ids_b` - Block IDs of partition σ
/// * `depth` - Number of path compression rounds (recommended: 2)
///
/// # Complexity
/// * Time: O(N) for parent construction + O(N × depth) for compression
/// * Memory: O(2N) (parent + compression buffer)
pub(crate) fn shallow_join_double_buffer(ids_a: &[u32], ids_b: &[u32], depth: u32) -> Vec<u32> {
    let n = ids_a.len();
    debug_assert_eq!(n, ids_b.len(), "Partition sizes must match");
    if n == 0 {
        return vec![];
    }

    let k_a = ids_a.iter().copied().max().unwrap_or(0) as usize + 1;
    let k_b = ids_b.iter().copied().max().unwrap_or(0) as usize + 1;

    // Phase 1: Build parent array via lightweight union (no rank, no full compression).
    // This correctly encodes equivalences from BOTH partitions.
    let mut parent: Vec<u32> = (0..n as u32).collect();

    // Simple union: parent[x] = y (no rank optimization — keep it lightweight)
    #[inline]
    fn shallow_find(parent: &[u32], x: u32) -> u32 {
        // Only 1 level of indirection — NOT full path compression
        let p = parent[x as usize];
        if p != x {
            return p;
        }
        x
    }

    // Partition a: elements in same block get unioned
    let mut first_a = vec![u32::MAX; k_a];
    for i in 0..n {
        let ba = ids_a[i] as usize;
        if first_a[ba] == u32::MAX {
            first_a[ba] = i as u32;
        } else {
            let rep = first_a[ba];
            parent[i] = rep;
        }
    }

    // Partition b: elements in same block get unioned
    // Merge with existing parent structure
    let mut first_b = vec![u32::MAX; k_b];
    for i in 0..n {
        let bb = ids_b[i] as usize;
        if first_b[bb] == u32::MAX {
            first_b[bb] = i as u32;
        } else {
            let rep = first_b[bb];
            // Union: connect the roots of i's tree and rep's tree
            let ri = shallow_find(&parent, i as u32);
            let rr = shallow_find(&parent, rep);
            if ri != rr {
                parent[ri.max(rr) as usize] = ri.min(rr);
            }
        }
    }

    // Phase 2: Double-buffer path compression for `depth` rounds.
    // Each round: dst[i] = src[src[i]] (pointer doubling).
    // This propagates root information without full convergence.
    let mut buf_a = parent;
    let mut buf_b = vec![0u32; n];

    for d in 0..depth {
        let (src, dst) = if d % 2 == 0 {
            (&buf_a[..], &mut buf_b[..])
        } else {
            (&buf_b[..], &mut buf_a[..])
        };
        for i in 0..n {
            let l = src[i] as usize;
            dst[i] = if l < n { src[l] } else { src[i] };
        }
    }

    let final_labels = if depth % 2 == 0 { &buf_a } else { &buf_b };

    // Phase 3: Normalize labels to contiguous block IDs.
    let mut label_map = HashMap::new();
    let mut next_id = 0u32;
    let mut result = vec![0u32; n];
    for i in 0..n {
        let root = final_labels[i];
        let id = label_map.entry(root).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        result[i] = *id;
    }

    result
}
