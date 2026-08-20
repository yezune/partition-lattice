//! Finding functional dependencies in a table, using nothing but the lattice order.
//!
//! A functional dependency `X -> Y` says: rows agreeing on `X` agree on `Y`. Group the
//! rows by their value of `X`, group them again by `Y`, and the dependency holds
//! exactly when the first grouping is *finer* than the second:
//!
//! ```text
//!     X -> Y   iff   partition(X).refines(partition(Y))
//! ```
//!
//! That is not an analogy. It is the definition, restated. Which is why this file
//! contains no dependency-checking algorithm: the lattice already is one.
//!
//! Composite left-hand sides come for free. `partition({A, B})` is
//! `partition(A).refine(partition(B))` — the common refinement is precisely "agree on
//! A *and* on B".
//!
//! Run it:
//! ```text
//! cargo run --example functional_dependencies
//! ```

use partition_lattice::Partition;
use std::collections::HashMap;

/// A tiny in-memory table. Real use would read a CSV; the point here is the algebra.
struct Table {
    columns: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    /// Group rows by the value in one column. Equal values share a block; the values
    /// themselves are then irrelevant, which is exactly what a partition records.
    fn partition_of(&self, col: usize) -> Partition {
        let mut seen: HashMap<&str, u32> = HashMap::new();
        let ids: Vec<u32> = self
            .rows
            .iter()
            .map(|r| {
                let next = seen.len() as u32;
                *seen.entry(r[col].as_str()).or_insert(next)
            })
            .collect();
        Partition::from_ids(ids)
    }

    /// The partition induced by a set of columns: the common refinement of each.
    fn partition_of_set(&self, cols: &[usize]) -> Partition {
        cols.iter()
            .map(|&c| self.partition_of(c))
            .reduce(|a, b| a.refine(&b))
            .expect("a determinant needs at least one column")
    }

    fn name_of(&self, cols: &[usize]) -> String {
        let names: Vec<&str> = cols.iter().map(|&c| self.columns[c].as_str()).collect();
        if names.len() == 1 { names[0].to_string() } else { format!("{{{}}}", names.join(", ")) }
    }
}

/// How badly `X -> Y` fails, in exact integers.
///
/// `refine` adds the distinctions `Y` makes that `X` does not; if there are none, the
/// dependency holds. Counting them measures *how far* it is from holding, which a
/// boolean cannot do — and the count is exact, so two near-misses can be ranked
/// without worrying about float ties.
fn violation(lhs: &Partition, rhs: &Partition) -> usize {
    lhs.refine(rhs).dit_count() - lhs.dit_count()
}

fn main() {
    let table = Table {
        columns: ["zip", "city", "country", "currency", "population"]
            .iter().map(|s| s.to_string()).collect(),
        rows: vec![
            ["01", "Seoul", "KR", "KRW", "9.4M"],
            ["02", "Seoul", "KR", "KRW", "9.4M"],
            ["03", "Busan", "KR", "KRW", "3.3M"],
            ["04", "Tokyo", "JP", "JPY", "14M"],
            ["05", "Tokyo", "JP", "JPY", "14M"],
            ["06", "Osaka", "JP", "JPY", "2.7M"],
        ]
        .iter()
        .map(|r| r.iter().map(|s| s.to_string()).collect())
        .collect(),
    };

    let n_cols = table.columns.len();
    println!("table: {} rows x {} columns\n", table.rows.len(), n_cols);

    // ── single-column dependencies ────────────────────────────────────────────
    println!("single-column dependencies");
    for lhs in 0..n_cols {
        for rhs in 0..n_cols {
            if lhs == rhs { continue; }
            let (pl, pr) = (table.partition_of(lhs), table.partition_of(rhs));
            let holds = pl.refines(&pr);
            let mark = if holds { "holds  " } else { "fails  " };
            let extra = if holds {
                String::new()
            } else {
                format!("  ({} rows-pairs disagree)", violation(&pl, &pr))
            };
            if holds || violation(&pl, &pr) <= 4 {
                println!(
                    "  {mark}{:<10} -> {:<10}{extra}",
                    table.columns[lhs], table.columns[rhs]
                );
            }
        }
    }

    // ── a composite determinant ───────────────────────────────────────────────
    // `city` alone does not determine `zip`; nothing here does, since zip is a key.
    // But the refinement of several columns can, and building it is one operation.
    println!("\ncomposite determinant");
    let cols = [1usize, 4]; // city, population
    let lhs = table.partition_of_set(&cols);
    for rhs in [2usize, 3] {
        println!(
            "  {}  {} -> {}",
            if lhs.refines(&table.partition_of(rhs)) { "holds " } else { "fails " },
            table.name_of(&cols),
            table.columns[rhs]
        );
    }

    // ── keys ──────────────────────────────────────────────────────────────────
    // A column is a key exactly when its partition is the discrete one: every row in
    // its own block, so it determines everything.
    println!("\nkeys (partition is discrete, so it determines every column)");
    let discrete = Partition::discrete(table.rows.len());
    for c in 0..n_cols {
        if table.partition_of(c).dit_count() == discrete.dit_count() {
            println!("  {} is a key", table.columns[c]);
        }
    }

    // ── why integers ──────────────────────────────────────────────────────────
    // Ranking near-misses needs exact counts: two candidates that differ by one
    // disagreeing pair must not tie because of float rounding.
    println!("\nranking near-misses by exact violation count");
    let mut near: Vec<(String, String, usize)> = Vec::new();
    for lhs in 0..n_cols {
        for rhs in 0..n_cols {
            if lhs == rhs { continue; }
            let v = violation(&table.partition_of(lhs), &table.partition_of(rhs));
            if v > 0 { near.push((table.columns[lhs].clone(), table.columns[rhs].clone(), v)); }
        }
    }
    near.sort_by_key(|(_, _, v)| *v);
    for (l, r, v) in near.iter().take(5) {
        println!("  {l:<10} -> {r:<10}  {v} disagreeing row-pairs");
    }
}
