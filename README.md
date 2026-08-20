# partition-lattice

Partition lattice algebra in exact integer arithmetic: **refine / coarsen / logical entropy**, plus
the incidence and dynamics coordinates built on them.

A *partition* of a finite universe `U` is a set of disjoint blocks covering `U`. Partitions
form a lattice under refinement, and this crate implements that lattice directly — the two
operations, the order, and the measures — without going through probabilities or floats where
integers will do.

```rust
use partition_lattice::Partition;

let a = Partition::from_ids(vec![0, 0, 1, 1]);   // {0,1}{2,3}
let b = Partition::from_ids(vec![0, 1, 0, 1]);   // {0,2}{1,3}

let finer   = a.refine(&b);                      // common refinement
let coarser = a.coarsen(&b);                     // common coarsening

assert_eq!(finer.block_count(), 4);              // all singletons
assert_eq!(coarser.block_count(), 1);            // one block
```

## What is here

| Area | Module | What it gives |
|---|---|---|
| Lattice core | `unified`, `partition` | `Partition`, `refine` / `coarsen` (also spelled `differentiate` / `integrate`), refinement order |
| Measure | `partition::logical_entropy` | `h(π) = 1 − Σ (|B|/|U|)²` — the *logical* entropy (Ellerman), exact rational underneath |
| Incidence | `incidence` | `dit_count`, `dit_xor_count`, `dit_intersection_count`, `destr` / `creat`, Möbius cumulants, magnitude homology |
| Curvature | `curvature` | `d_Σ` profile and its moments |
| Dynamics | `endomap`, `orbit`, `pure_pcc` | component / kernel partitions of a finite endomap, group-orbit partitions, value-free refinement steps |
| Backends | `optimized`, `sparse`, `streaming`, `cyclic*` | representations for large or structured universes |

### Two conventions worth knowing before you start

**Order.** `π ⪯ σ` means `π.refines(σ)` — *finer is smaller*. So `Bottom` is the discrete
partition (every element alone) and `Top` is the single-block partition.

**The operations are `refine` and `coarsen`.** Much of the literature — including
Ellerman's papers — calls the common refinement *join*; other sources call it *meet*.
Both are standard, and they are opposite conventions on the same structure, so a name
like `meet` cannot be read without first knowing which convention is in force.
`refine` and `coarsen` say what the operation does, so they read the same either way.

Every public name containing `meet` or `join` was **removed in 0.2.0** — see the
changelog for the mapping. `differentiate` and `integrate` are unchanged and are exact
synonyms of `refine` and `coarsen`: they name a direction of change, not a position in
the lattice, so they were never ambiguous.

```rust
# use partition_lattice::Partition;
# let (a, b) = (Partition::from_ids(vec![0,0,1,1]), Partition::from_ids(vec![0,1,0,1]));
assert_eq!(a.refine(&b).get_ids(),  a.differentiate(&b).get_ids());
assert_eq!(a.coarsen(&b).get_ids(), a.integrate(&b).get_ids());
```

If you are porting a formula from a paper, check which convention it uses and flip
`∧`/`∨` and the direction of `≤` together — or just use `refine` / `coarsen` and
`refines()`, none of which change meaning between conventions.

## `no_std`

`--no-default-features` builds the crate as `#![no_std]` + `alloc`. The lattice core, the
measures, the incidence coordinates and the endomap invariants are all available there. What
is gated out is the machinery that genuinely needs `std`: the sparse (roaring) backend,
serialization, timing, the std RNG, and the analysis modules built on those.

This is checked by the build, not by a comment: the workspace contains a `#![no_std]` crate
whose only job is to depend on this one with `default-features = false`.

## Features

| Feature | Default | Effect |
|---|:-:|---|
| `std` | ✅ | std-only backends, serialization, timing, RNG |
| `parallel` | ✅ | rayon for large universes; the small-N path never engages it, so disabling changes threading only, not results |
| `research` | | measurement ledgers, supply predicates and statistical helpers used by the host project; not lattice algebra, and off by default |
| `slow-tests` | | large exhaustive tests |
| `df-bins` | | one-off measurement binaries; not part of the API |

## Example: functional dependencies

A functional dependency `X -> Y` — rows agreeing on `X` agree on `Y` — is the
refinement order, restated:

```text
X -> Y   iff   partition(X).refines(partition(Y))
```

So finding dependencies in a table needs no dependency-checking algorithm; the lattice
already is one. Composite determinants come for free, since `partition({A, B})` is just
`partition(A).refine(partition(B))`.

```
$ cargo run --example functional_dependencies

single-column dependencies
  holds  zip        -> city
  holds  city       -> country
  fails  city       -> zip         (4 rows-pairs disagree)

composite determinant
  holds  {city, population} -> country

keys (partition is discrete, so it determines every column)
  zip is a key
```

The violation count is exact, so near-misses can be ranked without float ties deciding
which candidate wins — see [`examples/functional_dependencies.rs`](examples/functional_dependencies.rs).

## Exactness

Distinction counts are integers and stay integers. `h` is the only place a division happens,
and its numerator (`dit_count`) is exposed separately so you can compare entropies exactly
rather than through floats.

## License

MIT OR Apache-2.0, at your option.
