# Changelog

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
