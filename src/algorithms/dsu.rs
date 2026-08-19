use crate::unified::Partition;
#[allow(unused_imports)]
use alloc::boxed::Box;
#[allow(unused_imports)]
use alloc::vec::Vec;

/// A Disjoint Set Union (DSU) data structure with Path Compression and Union by Rank.
/// Also known as Union-Find.
#[derive(Debug, Clone)]
pub struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl DisjointSet {
    /// Creates a new DSU with `size` elements, each in its own set.
    pub fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    /// Finds the representative (root) of the set containing `i`.
    /// Applies Path Compression optimization.
    pub fn find(&mut self, i: usize) -> usize {
        if self.parent[i] != i {
            self.parent[i] = self.find(self.parent[i]);
        }
        self.parent[i]
    }

    /// Unites the sets containing `i` and `j`.
    /// Applies Union by Rank optimization.
    /// Returns true if the sets were different and merged, false if they were already the same.
    pub fn union(&mut self, i: usize, j: usize) -> bool {
        let root_i = self.find(i);
        let root_j = self.find(j);

        if root_i != root_j {
            match self.rank[root_i].cmp(&self.rank[root_j]) {
                core::cmp::Ordering::Less => self.parent[root_i] = root_j,
                core::cmp::Ordering::Greater => self.parent[root_j] = root_i,
                core::cmp::Ordering::Equal => {
                    self.parent[root_j] = root_i;
                    self.rank[root_i] += 1;
                }
            }
            true
        } else {
            false
        }
    }
}

/// Computes Connected Components using DSU.
/// Time Complexity: O(E * alpha(N)), where alpha is the inverse Ackermann function (nearly constant).
///
/// # Arguments
/// * `size` - The number of nodes in the graph (0..size-1).
/// * `edges` - A list of undirected edges (u, v).
///
/// # Returns
/// A `Partition` representing the connected components.
pub fn fast_connected_components(size: usize, edges: &[(usize, usize)]) -> Partition {
    let mut dsu = DisjointSet::new(size);

    for &(u, v) in edges {
        if u < size && v < size {
            dsu.union(u, v);
        }
    }

    // Map each element to its representative root
    let mut ids: Vec<u32> = Vec::with_capacity(size);
    for i in 0..size {
        ids.push(dsu.find(i) as u32);
    }

    // Partition::from_ids will normalize the IDs
    Partition::from_ids(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dsu_basic() {
        let mut dsu = DisjointSet::new(5);

        // Initially everyone is their own parent
        for i in 0..5 {
            assert_eq!(dsu.find(i), i);
        }

        // Union 0 and 1
        assert!(dsu.union(0, 1));
        assert_eq!(dsu.find(0), dsu.find(1));

        // Union 2 and 3
        assert!(dsu.union(2, 3));
        assert_eq!(dsu.find(2), dsu.find(3));
        assert_ne!(dsu.find(0), dsu.find(2));

        // Union 1 and 3 (merging two sets)
        assert!(dsu.union(1, 3));
        assert_eq!(dsu.find(0), dsu.find(2)); // Now all 0,1,2,3 are connected
        assert_ne!(dsu.find(0), dsu.find(4)); // 4 is still separate
    }

    #[test]
    fn test_fast_cc() {
        let size = 6;
        let edges = vec![(0, 1), (1, 2), (3, 4)];
        // Components: {0, 1, 2}, {3, 4}, {5}

        let partition = fast_connected_components(size, &edges);

        assert_eq!(partition.size(), 6);
        assert_eq!(partition.block_count(), 3);

        let ids = partition.get_ids();
        assert_eq!(ids[0], ids[1]);
        assert_eq!(ids[1], ids[2]);
        assert_eq!(ids[3], ids[4]);
        assert_ne!(ids[0], ids[3]);
        assert_ne!(ids[0], ids[5]);
    }
}
