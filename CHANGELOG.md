# Changelog

## 0.4.1

**`endomap` grows from describing one self-map to comparing two.** Purely additive; nothing
existing changed.

| | |
|---|---|
| `fix_count` | number of fixed points. With `components` it bounds the cycle structure without running an orbit: `fix_count(c) <= components(c).block_count()`, with equality exactly when no cycle is longer than 1 |
| `compose` | `(f∘g)(x) = f(g(x))` — the right factor acts first |
| `organisation` | the conjugacy class of an endomap under renaming of its carrier, as the least conjugate. Label blindness one layer up: it forgets which element is which, for the map itself. Costs `n! · n`, so it is for exhaustive work at small `n` |
| `survives`, `is_mutually_surviving` | whether composing with `e` leaves the organisation of `g` unchanged — on one side, or on both |

The claims in the `fix_count` documentation are checked over every self-map with `n <= 5`
against an oracle that walks the functional graph instead of folding Joins, so agreement is
evidence rather than shared code. The survival counts over all pairs for `n = 1..4`
(mutual `1/5/30/319`, one-sided `1/10/203/6,594`) were first obtained elsewhere and are
recomputed here in this crate's own kernel.

Documentation in the module no longer points at names that mean nothing outside the
repository it was written in — two such references went out in 0.4.0.

## 0.4.0

`Partition::from_ids` chooses differently. Every operation returns the same partition
it did before — the results are unchanged — but which representation carries that
partition has changed, and what it costs in memory has changed a great deal.

### `from_ids` can return `Lattice` for a large universe

It never did above 128 elements, so a partition with many blocks was forced onto
`FastPartition`, which holds one bitvector per block on top of the id vector.
`PartitionLattice` holds only the ids, so its footprint does not grow with the block
count. When Fast's bitvectors would exceed `LATTICE_FALLBACK_BUDGET` (1 GiB),
`from_ids` now hands the input to `Lattice`.

Measured over 276 configurations on two architectures, each machine running its before
and after arms back to back in one session, same commits and the same pinned toolchain:

| | Apple M4 | Ryzen 5 5600U |
|---|---|---|
| peak RSS, worst configuration | 6.79 GB → 2.52 GB | 23.52 GB → 2.55 GB |
| configurations the OOM killer stopped | 6 → 0 | 6 → 0 |
| worst regret against the best backend | 8.29 → 3.70 | 19.07 → 3.95 |
| 95th percentile regret | 3.43 → 3.09 | 3.18 → 2.73 |
| median regret | 1.50 → 1.57 | 1.36 → 1.30 |

Regret is the chosen backend's time over the fastest available one, so 1.0 is optimal.
The median moves in opposite directions on the two architectures, by 4.4% and 4.5%.
That is measured and not explained; whether it says something about the architectures
or about this particular grid is untested.

This is a bound on memory, not a speed-up. It is worth taking because the tail and the
failures improve on both machines while the median moves within a few percent either
way.

### Automatic conversion to `FactorizedFast` withdrawn

`from_ids` used to convert whenever the factorized form looked smaller than `n * 4`
bytes. That comparison was against `PartitionLattice`'s footprint rather than
`FastPartition`'s, and the representation it was really competing with could not be
reached, because `from_ids` could not return `Lattice` at that size. With the fallback
above in place it can be, and factorized loses: across 200 configurations where both
applied, it beat Lattice on time *and* memory in 0 on one machine and 3 on the other
(requiring a 1.02x margin on both axes, the measured noise floor; without that margin,
0 and 7). It was never the fastest backend in any configuration where the choice was
resolvable. On the block-dominant inputs where it actually fired it lost time without
saving memory — at a million elements with 100,000 blocks it was 1.41x slower on Meet
and 7.94x on Join, at 0.95x and 1.47x the peak RSS.

`Partition::FactorizedFast` and `Partition::from_ids_factorized` remain: this withdraws
a default, not a capability. `pick_factor_for_universe` is now public so an explicit
caller can ask for the block size the crate used to choose.

### `PartitionLattice` answers `block_count` and `logical_entropy` in O(1)

Both scanned the id vector on every call, while `FastPartition` has answered both from
stored fields all along. `Partition::diagnose` reads `block_count` on both operands, so
an uncached Lattice made an O(1) dispatch decision O(n): 237 microseconds per call at a
million elements, against a Fast operand too fast to time. The counts are now taken
during construction, where the normalising pass already computed them and discarded
them. The serialized form is unchanged, so payloads written by earlier versions load.

### A Join of two `Lattice` operands is no longer detoured through `Sparse`

The detour exists to stop a *result* `FastPartition` from allocating more than 256 MB
of bitvectors. A Lattice join returns a Lattice, which has no bitvectors, so the case
being guarded against cannot arise. Leaving it in was slower than the Lattice path
(327.7 ms against 21.3 ms at a million elements with 100,000 blocks) and also slower
than using Sparse directly (94.6 ms), since the conversion was added on top of it.

### Compatibility

Correctness and the serialized form are unchanged, and the explicit constructors —
`from_ids_fast`, `from_ids_lattice`, `from_ids_sparse`, `from_ids_factorized` — still
return exactly what they name. Code that matches on the variant `from_ids` returns may
see a different one. The minor version is raised because in `0.x` that is the unit
Cargo treats as compatible.

## 0.3.4

**The indication tier gets a home: `subset`.** Partitions are the categorical dual of
subsets, and this crate has always computed in the dual tier without naming it --
`logical_entropy` is `|dit| / N^2`, `ditset_xor_distance` is the cardinality of a
symmetric difference, `destr` and `creat` are set differences. All of them counted
subsets of `U x U` that were never constructed. Now they can be.

`SubsetU` is a subset of a finite universe with the Boolean operations on it -- union,
intersection, complement, symmetric difference, inclusion -- plus `all_subsets` for
exhaustive work. Nothing existing changed; this is purely an addition.

**And the three maps that connect the two tiers.** Neither lattice embeds in the other
(a partition lattice is non-distributive from `|U| >= 3`, and a sublattice of a
distributive lattice is distributive), so all that connects them are maps:

| | |
|---|---|
| `dit`, `indit` | partition to relation. Carries the common refinement to union *exactly*, and loses the common coarsening |
| `int_` | the right adjoint -- how the lost operation is recovered, and therefore what "subsets only" has to import |
| `cross`, `cross_image` | Spencer-Brown's cross `K(S) = {S, complement S}`. Two-to-one, and what it folds away is which side was marked |

`dit` is built by direct pair enumeration rather than through the lattice operations.
That is slower and deliberate: derived from `refine`, the identity
`dit(refine(a,b)) = dit(a) union dit(b)` would hold because both sides share code, and a
check of it would be a tautology rather than a fact. The two routes are kept separate,
and a test pins the pair-enumeration result against the long-standing cardinality one.

`int_` finds connected components as a Join fold over edge partitions, not as a
hand-written union-find -- the components *are* the transitive closure and Join *is*
that closure.

Compiles under `no_std` as before.

## 0.3.3

**`interpolate_with` no longer panics on swapped arguments.** The function documents
that its first argument is the finer partition, but a caller passing the two the other
way round hit `clamp(lo, hi)` with `lo > hi` and the process died. The bounds are now
ordered: identical behaviour whenever the precondition holds, and a defined result when
it does not -- the function only ever coarsens away from the first argument, so if the
second is the finer of the two there is nothing to coarsen and the first comes back
unchanged. Found when a conformance corpus first exercised `interpolate`.

**Documentation: references a reader cannot follow are gone.** Comments pointed at
private crate names, internal whitepaper identifiers and repository document paths --
21 lines that meant nothing outside the repository they came from. Each is replaced by
the fact it was standing in for, not deleted: a private crate name became the role that
crate plays ("the distributed runtime"), and a pointer to a private metrics module became
the reason it was being warned against ("those presuppose a background distribution").

This was overdue. When 0.3.0 shipped, "no Hangul in the published files" was taken as
evidence that the published surface was clean. It was not the same question, and 21
English lines pointing into a private repository went out.

No API change.

## 0.3.2

Documentation only. The README now leads with what the examples are for: finding
functional dependencies, comparing clusterings, minimising a DFA and finding connected
components use four different standard algorithms, and all four reduce to `refine` and
`coarsen`. The claim is measured rather than asserted — the logic in those examples is
9, 21, 29 and 0 lines respectively, with the rest being printing and sample data.

No code changed.

## 0.3.1

Documentation only: the README now lists all four examples rather than one. The
examples themselves shipped in 0.3.0.


## 0.3.0 — breaking

`Partition::distinction_count` is now **`Partition::dit_count`**.

One value had two names across the two packages: `distinction_count` in Rust,
`dit_count` in Python. Anyone writing both would have to remember which is which for
no reason. `dit_count` won because the rest of the vocabulary already uses *dit* —
`ditset_xor_distance`, `incidence::dit_count` — and because it was the name already
published to PyPI.

`incidence::dit_count(&p)` (a free function) and `p.dit_count()` (a method) now share
a name and agree exactly; that was checked over 2,000 random partitions before the
rename, with 0 disagreements. Rust resolves the two without ambiguity.

No alias was left behind, for the same reason as 0.2.0: a second name for one value is
a question the reader has to answer every time they see it.


## 0.2.0 — breaking

**Every public name containing `meet` or `join` was renamed to `refine` or `coarsen`.**
No aliases were left behind.

| removed | use |
|---|---|
| `complex_meet` | `complex_refine` |
| `cyclic_meet` | `cyclic_refine` |
| `cover_meet` / `cover_join` | `cover_refine` / `cover_coarsen` |
| `fused_join_cover` | `fused_coarsen_cover` |
| `fused_meet_entropy{,_ids}`, `par_fused_meet_entropy` | `fused_refine_entropy{,_ids}`, `par_fused_refine_entropy` |
| `block_graph_join_entropy`, `par_block_graph_join_entropy` | `block_graph_coarsen_entropy`, `par_block_graph_coarsen_entropy` |
| `distributed_meet_entropy` / `distributed_join_entropy` | `distributed_refine_entropy` / `distributed_coarsen_entropy` |
| `reduce_meet_entropy` / `reduce_join_entropy` | `reduce_refine_entropy` / `reduce_coarsen_entropy` |
| `should_use_sparse_meet` / `should_use_sparse_join` | `should_use_sparse_refine` / `should_use_sparse_coarsen` |
| `incremental_meet{,_fast}` | `incremental_refine{,_fast}` |
| `meet_join_closure` | `refine_coarsen_closure` |
| `betti0_via_join` | `betti0_via_coarsen` |
| `fused_meet_refines` | `fused_refine_and_check` |
| `register_meet_entropy_handler`, `MeetEntropyHandler` | `register_refine_entropy_handler`, `RefineEntropyHandler` |
| `JoinStrategy` | `CoarsenStrategy` |
| `DeltaMeetEngine`, `DeltaMeetStats`, module `delta_meet` | `DeltaRefineEngine`, `DeltaRefineStats`, module `delta_refine` |

### Why removal rather than deprecation

The two conventions in the literature disagree about which lattice operation is the
common refinement. In this crate `Meet` was the refinement; in Ellerman's papers the
refinement is the *join*. A caller porting a formula from that side would have reached
for the name that matched their reading, received a valid partition back, and never
learned it was the other operation.

Deprecation does not close that path: a deprecated function still returns an answer.
Taking the name away is the only change that turns a silent wrong result into a
compile error. `refine` and `coarsen` say what the operation does, so they read the
same under either convention.

`differentiate` and `integrate` are unchanged. They were never ambiguous — they name a
direction of change, not a position in the lattice — and `refine` / `coarsen` are their
synonyms, added in 0.1.2.

### The measurement behind this

Renaming is safe, and that is not an assumption. The order-decision procedure was
checked exhaustively for self-duality:

    s <= t  <=>  dual(t) <= dual(s)      6,055 terms, 36,663,025 pairs, 0 mismatches

The two conventions are exact mirrors, so every true statement maps to a true one.
That is also why the *convention* was left alone: switching it would buy convenience
when reading papers and cost every existing caller a silent reversal. The naming was
what needed fixing, not the convention.

Not touched: the GPU crate's Metal kernel names (`fused_meet_count` and friends) are
internal to that crate and are not part of this package's surface.


## 0.1.2

Adds `Partition::refine` and `Partition::coarsen`: the same two operations named for
what they do rather than for their position in the lattice.

`meet` and `join` swap meaning between the two conventions in the literature — this
crate's `Meet` is the common refinement, Ellerman's `join` is — so a formula ported
without checking the convention silently computes the wrong thing. `refine` and
`coarsen` read identically under either one.

Pure delegation to `differentiate` / `integrate`; nothing else changes, and the older
names are not deprecated.

## 0.1.1

Metadata only; the library is byte-identical to 0.1.0.

- `repository` now points at <https://github.com/yezune/partition-lattice>.
  In 0.1.0 it pointed at the private repository this crate is developed in, so the
  link 404'd for everyone and the commit recorded in the package's
  `.cargo_vcs_info.json` referred to a history nobody could read.

That mirror is an output, not a workspace: development happens upstream and each
release lands there as a single commit, tagged `v<version>`. Its history therefore
starts at 0.1.0 rather than reproducing the upstream one.

## 0.1.0

First release.

- Partition lattice core: `Partition`, Meet (`differentiate`), Join (`integrate`),
  and the refinement order. Note the convention: `π ⪯ σ` means *finer*, so `Bottom`
  is the discrete partition. Much of the literature uses the order-dual.
- Logical entropy `h(π) = 1 − Σ (|B|/|U|)²`, with the integer numerator
  (`dit_count`) exposed separately so entropies can be compared exactly.
- Incidence coordinates: `dit_xor_count`, `destr`/`creat`, Möbius cumulants,
  magnitude homology.
- Curvature: the `d_Σ` profile and its moments.
- Dynamics: kernel and component partitions of a finite endomap, group-orbit
  partitions, value-free refinement steps.
- Alternative backends for large or structured universes: bitset, sparse,
  streaming, cyclic, factorized.
- `--no-default-features` builds as `#![no_std]` + `alloc`.
