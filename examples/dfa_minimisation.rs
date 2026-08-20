//! Minimising a DFA — which is partition refinement and nothing else.
//!
//! Moore's algorithm: start by separating accepting from non-accepting states, then
//! repeatedly split any block whose members disagree about *which block* their
//! transitions land in. Stop when a round changes nothing. The blocks of the fixed
//! point are the states of the minimal DFA.
//!
//! Each round is one `refine`, and "changed nothing" is an exact integer comparison of
//! `dit_count` — no epsilon, no ambiguity about whether the fixed point was reached.
//!
//! ```text
//! cargo run --example dfa_minimisation
//! ```

use partition_lattice::Partition;

/// A DFA over a two-letter alphabet: `delta[state][letter]`.
struct Dfa {
    delta: Vec<[usize; 2]>,
    accepting: Vec<bool>,
}

impl Dfa {
    /// Split states by where their transitions land, given the current partition.
    ///
    /// Two states get the same signature exactly when, for every letter, their targets
    /// are currently indistinguishable. Grouping by signature is itself a partition.
    fn signature_partition(&self, current: &Partition) -> Partition {
        let ids = current.get_ids();
        let mut seen: Vec<[u32; 2]> = Vec::new();
        let sig: Vec<u32> = self
            .delta
            .iter()
            .map(|t| {
                let s = [ids[t[0]], ids[t[1]]];
                match seen.iter().position(|x| *x == s) {
                    Some(i) => i as u32,
                    None => { seen.push(s); (seen.len() - 1) as u32 }
                }
            })
            .collect();
        Partition::from_ids(sig)
    }

    fn minimise(&self) -> (Partition, usize) {
        // Round 0: accepting vs non-accepting. Nothing finer is justified yet.
        let mut current =
            Partition::from_ids(self.accepting.iter().map(|&a| a as u32).collect());
        let mut rounds = 0;
        loop {
            let next = current.refine(&self.signature_partition(&current));
            rounds += 1;
            // Exact: the count either grew or it did not.
            if next.dit_count() == current.dit_count() {
                return (next, rounds);
            }
            current = next;
        }
    }
}

fn main() {
    // A 6-state DFA with redundancy: states 1 and 2 behave identically, as do 3 and 4.
    //          a    b
    //   0 ->   1    2      (start, non-accepting)
    //   1 ->   3    4
    //   2 ->   4    3
    //   3 ->   5    5      (accepting)
    //   4 ->   5    5      (accepting)
    //   5 ->   5    5      (accepting, sink)
    let dfa = Dfa {
        delta: vec![[1, 2], [3, 4], [4, 3], [5, 5], [5, 5], [5, 5]],
        accepting: vec![false, false, false, true, true, true],
    };

    println!("DFA with {} states", dfa.delta.len());
    let (minimal, rounds) = dfa.minimise();
    println!("fixed point reached in {rounds} rounds\n");

    println!("equivalence classes ({} states in the minimal DFA)", minimal.block_count());
    for (i, block) in minimal.to_blocks().iter().enumerate() {
        let acc = dfa.accepting[block[0]];
        println!(
            "  q{i}  {{{}}}{}",
            block.iter().map(|s| format!("s{s}")).collect::<Vec<_>>().join(","),
            if acc { "  accepting" } else { "" }
        );
    }

    // The whole algorithm is a refinement chain, and the chain is an increasing
    // sequence in the lattice order — each round is finer than the last, so it must
    // terminate on a finite universe.
    println!("\nthe refinement chain");
    let mut p = Partition::from_ids(dfa.accepting.iter().map(|&a| a as u32).collect());
    let mut round = 0;
    loop {
        println!(
            "  round {round}: {} blocks, {} distinctions",
            p.block_count(),
            p.dit_count()
        );
        let next = p.refine(&dfa.signature_partition(&p));
        if next.dit_count() == p.dit_count() { break; }
        // Every round is strictly finer than the one before.
        assert!(next.refines(&p), "refinement must move down the lattice");
        p = next;
        round += 1;
    }
    println!("  (each round refines the previous, so termination is the lattice being finite)");
}
