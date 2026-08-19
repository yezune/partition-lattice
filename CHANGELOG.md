# Changelog

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
