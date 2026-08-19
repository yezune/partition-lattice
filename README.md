# partition-lattice

Partition lattice algebra in exact integer arithmetic: **Meet / Join / logical entropy**, plus
the incidence and dynamics coordinates built on them.

A *partition* of a finite universe `U` is a set of disjoint blocks covering `U`. Partitions
form a lattice under refinement, and this crate implements that lattice directly — the two
operations, the order, and the measures — without going through probabilities or floats where
integers will do.

```rust
use partition_lattice::Partition;

let a = Partition::from_ids(vec![0, 0, 1, 1]);   // {0,1}{2,3}
let b = Partition::from_ids(vec![0, 1, 0, 1]);   // {0,2}{1,3}

let meet = a.differentiate(&b);                  // common refinement (finer)
let join = a.integrate(&b);                      // common coarsening (coarser)

assert_eq!(meet.block_count(), 4);               // all singletons
assert_eq!(join.block_count(), 1);               // one block
```

## What is here

| Area | Module | What it gives |
|---|---|---|
| Lattice core | `unified`, `partition` | `Partition`, Meet (`differentiate`), Join (`integrate`), refinement order |
| Measure | `partition::logical_entropy` | `h(π) = 1 − Σ (|B|/|U|)²` — the *logical* entropy (Ellerman), exact rational underneath |
| Incidence | `incidence` | `dit_count`, `dit_xor_count`, `dit_intersection_count`, `destr` / `creat`, Möbius cumulants, magnitude homology |
| Curvature | `curvature` | `d_Σ` profile and its moments |
| Dynamics | `endomap`, `orbit`, `pure_pcc` | component / kernel partitions of a finite endomap, group-orbit partitions, value-free refinement steps |
| Backends | `optimized`, `sparse`, `streaming`, `cyclic*` | representations for large or structured universes |

### Two conventions worth knowing before you start

**Order.** `π ⪯ σ` means `π.refines(σ)` — *finer is smaller*. So `Bottom` is the discrete
partition (every element alone) and `Top` is the single-block partition.

**Meet and Join.** `Meet` = `differentiate` = common **refinement**; `Join` = `integrate` =
common **coarsening**. Much of the literature — including Ellerman's own papers — uses the
**order-dual** convention, where join is the common refinement. Neither is wrong; they are
opposite conventions on the same structure. If you are porting formulas from a paper, check
which one it uses and flip `∧`/`∨` and the direction of `≤` together.

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

## Exactness

Distinction counts are integers and stay integers. `h` is the only place a division happens,
and its numerator (`dit_count`) is exposed separately so you can compare entropies exactly
rather than through floats.

## License

MIT OR Apache-2.0, at your option.
