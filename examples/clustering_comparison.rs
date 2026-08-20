//! Comparing two clusterings — and getting more back than a number.
//!
//! Every clustering-comparison index (Rand, Jaccard on pairs, `d_⊕`) answers *how
//! much* two groupings differ. None of them answers *in what way*. The lattice does,
//! because the two clusterings have a meet and a join:
//!
//! - `a.coarsen(b)` — the finest grouping both agree on. The consensus.
//! - `a.refine(b)`  — the coarsest grouping that respects every split either made.
//! - `destr(a, b)`  — distinctions `a` draws that `b` erases.
//! - `creat(a, b)`  — distinctions `b` draws that `a` does not.
//!
//! The last two matter because a scalar conflates them: two comparisons with the same
//! distance can be "a split what b merged" or "a merged what b split", which call for
//! opposite fixes.
//!
//! ```text
//! cargo run --example clustering_comparison
//! ```
//!
//! On metricity: `d_⊕` satisfies the triangle inequality, and so does `1 - rand`.
//! This example checks both rather than asserting either — see the last section.

use partition_lattice::{incidence, Partition};

fn show(label: &str, p: &Partition) {
    let blocks: Vec<String> = p
        .to_blocks()
        .iter()
        .map(|b: &Vec<usize>| format!("{{{}}}", b.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(",")))
        .collect();
    println!("  {label:<12} {} blocks  {}", p.block_count(), blocks.join(" "));
}

fn main() {
    // Ten points. Two algorithms disagree about where the boundaries are.
    //   a: splits {0..3} from {4..6} from {7..9}
    //   b: agrees about 0..3, but merges 4..9 into one cluster and pulls 3 out
    let a = Partition::from_ids(vec![0, 0, 0, 0, 1, 1, 1, 2, 2, 2]);
    let b = Partition::from_ids(vec![0, 0, 0, 3, 1, 1, 1, 1, 1, 1]);

    println!("two clusterings of the same 10 points");
    show("a", &a);
    show("b", &b);

    // ── how much ──────────────────────────────────────────────────────────────
    println!("\nhow much they differ");
    println!("  d_xor          {:.4}   (normalised symmetric difference of the dit sets)", a.ditset_xor_distance(&b));
    println!("  rand agreement {:.4}", a.rand_agreement(&b));
    println!("  jaccard        {:.4}", a.dit_set_jaccard(&b).unwrap_or(f64::NAN));

    // ── in what way ───────────────────────────────────────────────────────────
    // This is what a scalar cannot give you.
    println!("\nin what way");
    show("consensus", &a.coarsen(&b));
    show("combined", &a.refine(&b));
    println!(
        "  a splits what b merges: {} ordered pairs",
        incidence::destr(&a, &b)
    );
    println!(
        "  b splits what a merges: {} ordered pairs",
        incidence::creat(&a, &b)
    );
    println!("  (a scalar distance adds these two together and loses the direction)");

    // ── consensus over many runs ──────────────────────────────────────────────
    // Coarsening is associative, so a consensus over any number of runs is a fold.
    // What survives is exactly what every run agreed on.
    let runs = [
        Partition::from_ids(vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 2]),
        Partition::from_ids(vec![0, 0, 0, 1, 1, 2, 2, 2, 2, 2]),
        Partition::from_ids(vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2]),
    ];
    let consensus = runs.iter().cloned().reduce(|x, y| x.coarsen(&y)).unwrap();
    let strict = runs.iter().cloned().reduce(|x, y| x.refine(&y)).unwrap();
    println!("\nconsensus over 3 runs");
    show("agreed", &consensus);
    show("any-split", &strict);
    println!("  stability: {:.4} of possible distinctions survive all three",
             consensus.logical_entropy() / strict.logical_entropy());

    // ── exactness ─────────────────────────────────────────────────────────────
    // Ranking candidates needs exact counts, not floats that can tie by rounding.
    println!("\nranking runs against a reference, by exact integer");
    let reference = Partition::from_ids(vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 2]);
    let mut scored: Vec<(usize, i64)> = runs
        .iter()
        .enumerate()
        .map(|(i, r)| (i, incidence::dit_xor_count(&reference, r)))
        .collect();
    scored.sort_by_key(|(_, d)| *d);
    for (i, d) in scored {
        println!("  run {i}: {d} differing ordered pairs");
    }

    // ── metricity, measured rather than claimed ───────────────────────────────
    let mut rng: u64 = 0xDEADBEEF;
    let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
    let ps: Vec<Partition> = (0..40)
        .map(|_| {
            let k = 2 + (next() % 4) as u64;
            Partition::from_ids((0..10).map(|_| (next() % k) as u32).collect())
        })
        .collect();
    let (mut dv, mut rv, mut checked) = (0usize, 0usize, 0usize);
    for x in &ps { for y in &ps { for z in &ps {
        checked += 1;
        if x.ditset_xor_distance(z) > x.ditset_xor_distance(y) + y.ditset_xor_distance(z) + 1e-12 { dv += 1; }
        let r = |p: &Partition, q: &Partition| 1.0 - p.rand_agreement(q);
        if r(x, z) > r(x, y) + r(y, z) + 1e-12 { rv += 1; }
    }}}
    println!("\ntriangle inequality over {checked} triples");
    println!("  d_xor      violations: {dv}");
    println!("  1 - rand   violations: {rv}");
    println!("  (both hold; metricity is not what distinguishes them — the algebra is)");
}
