//! Connected components — building them by coarsening, one edge at a time.
//!
//! An edge says "these two vertices belong together". That is a partition: everything
//! alone except one merged pair. Fold those partitions with `coarsen` and you have the
//! components, because the least upper bound of "u ~ v" facts is exactly their
//! transitive closure.
//!
//! No union-find is written here. The join *is* the union-find.
//!
//! ```text
//! cargo run --example graph_components
//! ```

use partition_lattice::{simplicial, Partition};

/// The partition asserting one edge and nothing else.
fn edge(n: usize, u: usize, v: usize) -> Partition {
    let ids: Vec<u32> = (0..n).map(|i| if i == v { u as u32 } else { i as u32 }).collect();
    Partition::from_ids(ids)
}

/// Components of a graph, as the join of its edges over the discrete partition.
fn components(n: usize, edges: &[[usize; 2]]) -> Partition {
    edges
        .iter()
        .fold(Partition::discrete(n), |acc, e| acc.coarsen(&edge(n, e[0], e[1])))
}

fn show(label: &str, p: &Partition) {
    let blocks: Vec<String> = p
        .to_blocks()
        .iter()
        .map(|b: &Vec<usize>| format!("{{{}}}", b.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(",")))
        .collect();
    println!("  {label:<18} {} components  {}", p.block_count(), blocks.join(" "));
}

fn main() {
    // 8 vertices, three components: a triangle, a path, and an isolated vertex.
    let n = 8;
    let edges = [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [6, 7]];

    println!("graph: {n} vertices, {} edges", edges.len());
    let comp = components(n, &edges);
    show("components", &comp);

    // The crate ships the same computation; agreeing with it is a check on the fold.
    let b0 = simplicial::betti0_via_coarsen(n, &edges);
    println!("  betti_0 = {b0}  (matches the fold: {})", b0 == comp.block_count());

    // ── adding edges only coarsens ────────────────────────────────────────────
    // Components never split as edges arrive, which in the lattice is the statement
    // that each step moves *up*: the previous partition refines the next one.
    println!("\nadding edges one at a time (each step is coarser than the last)");
    let mut acc = Partition::discrete(n);
    for (i, e) in edges.iter().enumerate() {
        let next = acc.coarsen(&edge(n, e[0], e[1]));
        assert!(acc.refines(&next), "adding an edge cannot split a component");
        println!(
            "  +{:?}  {} components, {} distinctions remaining",
            e,
            next.block_count(),
            next.dit_count()
        );
        acc = next;
        let _ = i;
    }

    // ── which edges actually mattered ─────────────────────────────────────────
    // An edge is redundant exactly when adding it changes nothing — an exact integer
    // comparison, not a tolerance.
    println!("\nredundant edges (adding them merges nothing new)");
    let mut acc = Partition::discrete(n);
    for e in &edges {
        let next = acc.coarsen(&edge(n, e[0], e[1]));
        if next.dit_count() == acc.dit_count() {
            println!("  {e:?} is redundant — its endpoints were already connected");
        }
        acc = next;
    }

    // ── comparing two graphs by their components ──────────────────────────────
    let other = components(n, &[[0, 1], [1, 2], [2, 0], [3, 4], [5, 6], [6, 7]]);
    println!("\ncomparing two graphs by component structure");
    show("graph 1", &comp);
    show("graph 2", &other);
    println!("  d_xor between them: {:.4}", comp.ditset_xor_distance(&other));
    show("agreed", &comp.coarsen(&other));
}
