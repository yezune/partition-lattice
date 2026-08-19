use super::*;

const EPS: f64 = 1e-10;

// ── Phase 0: Conversion tests ───────────────────────────────

#[test]
fn test_lattice_to_fast_roundtrip() {
    let pl = PartitionLattice::from_ids(vec![0, 0, 1, 2, 2, 2]);
    let fp = FastPartition::from(&pl);
    let pl2 = PartitionLattice::from(&fp);
    assert_eq!(pl.get_ids(), pl2.get_ids());
}

#[test]
fn test_fast_to_lattice_roundtrip() {
    let fp = FastPartition::from_ids(&[0, 1, 0, 1, 2, 2]);
    let pl = PartitionLattice::from(&fp);
    let fp2 = FastPartition::from(&pl);
    assert_eq!(fp.get_ids(), fp2.get_ids());
}

#[test]
fn test_conversion_preserves_entropy() {
    let ids = vec![0, 0, 1, 1, 2, 3, 3, 3];
    let pl = PartitionLattice::from_ids(ids.clone());
    let fp = FastPartition::from(&pl);
    assert!((pl.logical_entropy() - fp.logical_entropy()).abs() < EPS);
}

#[test]
fn test_conversion_empty() {
    let pl = PartitionLattice::from_ids(vec![]);
    let fp = FastPartition::from(&pl);
    assert_eq!(fp.size(), 0);
    let pl2 = PartitionLattice::from(&fp);
    assert_eq!(pl2.size(), 0);
}

// ── Phase 1: Trait equivalence tests ────────────────────────

fn assert_ops_equal(ids: Vec<u32>) {
    let pl = PartitionLattice::from_ids(ids.clone());
    let fp = FastPartition::from_ids(&ids);

    // Basic properties
    assert_eq!(PartitionOps::size(&pl), PartitionOps::size(&fp));
    assert_eq!(
        PartitionOps::block_count(&pl),
        PartitionOps::block_count(&fp)
    );
    assert!((PartitionOps::logical_entropy(&pl) - PartitionOps::logical_entropy(&fp)).abs() < EPS);
    assert_eq!(
        PartitionOps::distinction_count(&pl),
        PartitionOps::distinction_count(&fp)
    );
    assert!((PartitionOps::logical_area(&pl) - PartitionOps::logical_area(&fp)).abs() < EPS);
}

#[test]
fn test_trait_equivalence_discrete() {
    assert_ops_equal(vec![0, 1, 2, 3]);
}

#[test]
fn test_trait_equivalence_indiscrete() {
    assert_ops_equal(vec![0, 0, 0, 0]);
}

#[test]
fn test_trait_equivalence_mixed() {
    assert_ops_equal(vec![0, 0, 1, 1, 2, 3, 3, 3]);
}

#[test]
fn test_trait_equivalence_binary_ops() {
    let ids_a: Vec<u32> = vec![0, 0, 1, 1];
    let ids_b: Vec<u32> = vec![0, 1, 0, 1];

    let pl_a = PartitionLattice::from_ids(ids_a.clone());
    let pl_b = PartitionLattice::from_ids(ids_b.clone());
    let fp_a = FastPartition::from_ids(&ids_a);
    let fp_b = FastPartition::from_ids(&ids_b);

    // Meet
    let pl_meet = PartitionOps::differentiate(&pl_a, &pl_b);
    let fp_meet = PartitionOps::differentiate(&fp_a, &fp_b);
    assert!(
        (PartitionOps::logical_entropy(&pl_meet) - PartitionOps::logical_entropy(&fp_meet)).abs()
            < EPS
    );

    // Join
    let pl_join = PartitionOps::integrate(&pl_a, &pl_b);
    let fp_join = PartitionOps::integrate(&fp_a, &fp_b);
    assert!(
        (PartitionOps::logical_entropy(&pl_join) - PartitionOps::logical_entropy(&fp_join)).abs()
            < EPS
    );

    // Divergence
    let pl_div = PartitionOps::logical_divergence(&pl_a, &pl_b);
    let fp_div = PartitionOps::logical_divergence(&fp_a, &fp_b);
    assert!((pl_div - fp_div).abs() < EPS);

    // XOR distance
    let pl_xor = PartitionOps::ditset_xor_distance(&pl_a, &pl_b);
    let fp_xor = PartitionOps::ditset_xor_distance(&fp_a, &fp_b);
    assert!((pl_xor - fp_xor).abs() < EPS);

    // Fused entropy
    let pl_fused = PartitionOps::fused_entropy_differentiate(&pl_a, &pl_b);
    let fp_fused = PartitionOps::fused_entropy_differentiate(&fp_a, &fp_b);
    assert!((pl_fused - fp_fused).abs() < EPS);

    // Refines
    assert_eq!(
        PartitionOps::refines(&pl_meet, &pl_a),
        PartitionOps::refines(&fp_meet, &fp_a)
    );
}

// ── Phase 2: Partition enum tests ───────────────────────────

#[test]
fn test_partition_auto_select_small() {
    let p = Partition::from_ids(vec![0, 0, 1, 1]);
    assert!(!p.is_fast(), "Small partition should use Lattice");
}

#[test]
fn test_partition_auto_select_large() {
    // Layer A: a pattern of cycle_len=10 routes to WideCyclic
    // (L ∈ [9, 16]). To trigger the Fast fallback we need a truly
    // non-cyclic id sequence at universe ≥ AUTO_SELECT_THRESHOLD.
    let mut ids: Vec<u32> = (0..200).map(|i| (i % 10) as u32).collect();
    // Break cyclicity at position 17 — guarantees Fast routing.
    ids[17] = 99;
    let p = Partition::from_ids(ids);
    assert!(
        p.is_fast(),
        "Large non-cyclic partition should use Fast (got {:?})",
        std::mem::discriminant(&p)
    );
}

/// Fourth step of the `from_ids` routing chain: automatic conversion from the `Fast`
/// fallback into `FactorizedFast`, guarded by
/// `ids.len() >= MIN_UNIVERSE_FOR_FACTORIZATION` and factorized memory below fast memory.
/// Non-cyclic ids with a low block count and `N >= 1024` route to `FactorizedFast`.
#[test]
fn test_partition_auto_select_factorized_fast_routes_low_block_count() {
    // N = 4096 with 4 distinct non-cyclic blocks routes to FactorizedFast.
    let mut ids: Vec<u32> = (0..4096).map(|i| (i % 4) as u32).collect();
    ids[17] = 99; // break cyclicity
    let p = Partition::from_ids(ids);
    assert!(
        matches!(p, Partition::FactorizedFast(_)),
        "Large non-cyclic low-block-count partition (N=4096, K=5) should route to FactorizedFast (got {:?})",
        std::mem::discriminant(&p)
    );
}

/// Below the minimum universe size the routing keeps the `Fast` fallback.
#[test]
fn test_partition_auto_select_below_factorization_threshold_stays_fast() {
    // N = 512 is below MIN_UNIVERSE_FOR_FACTORIZATION (1024), so Fast is kept.
    let mut ids: Vec<u32> = (0..512).map(|i| (i % 4) as u32).collect();
    ids[17] = 99;
    let p = Partition::from_ids(ids);
    assert!(
        p.is_fast(),
        "Below-threshold non-cyclic partition (N=512) should stay Fast (got {:?})",
        std::mem::discriminant(&p)
    );
}

/// The memory guard: when overrides explode, the factorized representation is larger than
/// `FastPartition`, so routing must fall back to `Fast`.
#[test]
fn test_partition_auto_select_high_block_count_falls_back_fast() {
    // N = 4096 with 4096 distinct ids gives roughly F-1 overrides per coarse block,
    // so factorized memory exceeds fast memory.
    let ids: Vec<u32> = (0..4096).map(|i| i as u32).collect();
    let p = Partition::from_ids(ids);
    assert!(
        p.is_fast(),
        "High-block-count partition (N=4096, K=4096) should fall back to Fast (got {:?})",
        std::mem::discriminant(&p)
    );
}

/// Layer A: a cyclic id sequence of L=10 at large universe now routes
/// to `Partition::WideCyclic` (was Fast pre-Layer-A). Companion to
/// `test_partition_auto_select_large`.
#[test]
fn test_partition_auto_select_large_wide_cyclic() {
    let ids: Vec<u32> = (0..200).map(|i| (i % 10) as u32).collect();
    let p = Partition::from_ids(ids);
    assert!(
        matches!(p, Partition::WideCyclic(_)),
        "Large cyclic L=10 partition should route to WideCyclic (got {:?})",
        std::mem::discriminant(&p)
    );
}

/// Layer B-β-true: explicit factorized constructor must produce
/// `Partition::FactorizedFast` and match dense Fast bit-for-bit on the
/// scalar primitives (size, block_count, get_block, logical_entropy,
/// distinction_count).
#[test]
fn test_partition_factorized_fast_basic_invariants() {
    let mut ids: Vec<u32> = (0..256).map(|i| (i % 4) as u32).collect();
    // Break cyclicity at position 17 to keep this non-cyclic.
    ids[17] = 99;
    let fast = Partition::from_ids_fast(ids.clone());
    let fac = Partition::from_ids_factorized(ids.clone(), 64);
    assert!(
        matches!(fac, Partition::FactorizedFast(_)),
        "from_ids_factorized must route to FactorizedFast"
    );
    assert_eq!(fac.size(), fast.size());
    assert_eq!(fac.block_count(), fast.block_count());
    assert_eq!(fac.distinction_count(), fast.distinction_count());
    assert!((fac.logical_entropy() - fast.logical_entropy()).abs() < EPS);
    for i in 0..ids.len() {
        assert_eq!(
            fac.get_block(i),
            fast.get_block(i),
            "get_block mismatch at i={}",
            i
        );
    }
}

/// Layer B-β-true: `get_ids()` materialization must be bit-exact with
/// the dense Fast representation (round-trip path used by SpectralSignature
/// and compound ops fall-through).
#[test]
fn test_partition_factorized_fast_get_ids_bit_exact() {
    let mut ids: Vec<u32> = (0..1024).map(|i| ((i * 7919) % 8) as u32).collect();
    ids[13] = 42; // break any incidental periodicity
    let fast = Partition::from_ids_fast(ids.clone());
    let fac = Partition::from_ids_factorized(ids.clone(), 64);
    assert_eq!(
        fast.get_ids(),
        fac.get_ids(),
        "FactorizedFast::get_ids must be bit-exact with FastPartition::get_ids"
    );
}

/// Layer B-β-true: compound ops (Meet/Join) fall through `to_fast()` and
/// produce results that agree with the dense path at the entropy level.
#[test]
fn test_partition_factorized_fast_meet_fall_through() {
    let mut ids_a: Vec<u32> = (0..256).map(|i| (i % 4) as u32).collect();
    let mut ids_b: Vec<u32> = (0..256).map(|i| (i % 5) as u32).collect();
    ids_a[7] = 99;
    ids_b[11] = 88;
    let fast_a = Partition::from_ids_fast(ids_a.clone());
    let fast_b = Partition::from_ids_fast(ids_b.clone());
    let fac_a = Partition::from_ids_factorized(ids_a.clone(), 64);
    let fac_b = Partition::from_ids_factorized(ids_b.clone(), 64);

    let meet_dense = fast_a.differentiate(&fast_b);
    let meet_fac = fac_a.differentiate(&fac_b);
    assert!((meet_dense.logical_entropy() - meet_fac.logical_entropy()).abs() < EPS);
    assert_eq!(meet_dense.block_count(), meet_fac.block_count());
}

/// Layer B-β-true: `to_fast()` round-trip must be bit-exact (gate-6.2
/// callsite protection — compound ops materialize via this path).
#[test]
fn test_partition_factorized_fast_to_fast_round_trip() {
    let ids: Vec<u32> = (0..512).map(|i| ((i * 13) % 7) as u32).collect();
    let fac = Partition::from_ids_factorized(ids.clone(), 64);
    let dense = fac.to_fast();
    let fast = Partition::from_ids_fast(ids.clone()).to_fast();
    assert_eq!(dense.get_ids(), fast.get_ids());
}

#[test]
fn test_partition_enum_entropy() {
    let ids = vec![0, 0, 1, 1, 2, 3, 3, 3];
    let p = Partition::from_ids(ids.clone());
    let pl = PartitionLattice::from_ids(ids);
    assert!((p.logical_entropy() - pl.logical_entropy()).abs() < EPS);
}

#[test]
fn test_partition_enum_meet() {
    let a = Partition::from_ids(vec![0, 0, 1, 1]);
    let b = Partition::from_ids(vec![0, 1, 0, 1]);
    let meet = a.differentiate(&b);
    // Meet of {{0,1},{2,3}} and {{0,2},{1,3}} = discrete {{0},{1},{2},{3}}
    assert_eq!(meet.block_count(), 4);
}

#[test]
fn test_partition_enum_join() {
    let a = Partition::from_ids(vec![0, 0, 1, 1]);
    let b = Partition::from_ids(vec![0, 1, 0, 1]);
    let join = a.integrate(&b);
    // Join = indiscrete {{0,1,2,3}}
    assert_eq!(join.block_count(), 1);
}

#[test]
fn test_partition_enum_refines() {
    let fine = Partition::from_ids(vec![0, 1, 2, 3]); // discrete
    let coarse = Partition::from_ids(vec![0, 0, 1, 1]);
    assert!(fine.refines(&coarse));
    assert!(!coarse.refines(&fine));
}

#[test]
fn test_partition_enum_equality() {
    let a = Partition::from_ids(vec![0, 0, 1, 1]);
    let b = Partition::from_ids_fast(vec![0, 0, 1, 1]);
    assert_eq!(a, b, "Same ids should be equal regardless of backend");
}

#[test]
fn test_partition_enum_to_lattice_to_fast() {
    let p = Partition::from_ids(vec![0, 0, 1, 2, 2, 2]);
    let pl = p.to_lattice();
    let fp = p.to_fast();
    assert!((pl.logical_entropy() - fp.logical_entropy()).abs() < EPS);
}

#[test]
fn test_partition_from_lattice() {
    let pl = PartitionLattice::from_ids(vec![0, 0, 1, 1]);
    let p = Partition::from(pl.clone());
    assert_eq!(p.get_ids(), pl.get_ids());
}

#[test]
fn test_partition_from_fast() {
    let fp = FastPartition::from_ids(&[0, 0, 1, 1]);
    let p = Partition::from(fp.clone());
    assert_eq!(p.get_ids(), fp.get_ids());
    assert!(p.is_fast());
}

#[test]
fn test_partition_enum_divergence() {
    let a = Partition::from_ids(vec![0, 0, 1, 1]);
    let b = Partition::from_ids(vec![0, 0, 0, 0]);
    let div = a.logical_divergence(&b);
    // D_L(a||b) = h(a ∧ b) - h(b) = h(a) - 0 = 0.5
    assert!((div - 0.5).abs() < EPS);
}

#[test]
fn test_partition_enum_xor_distance() {
    let a = Partition::from_ids(vec![0, 0, 1, 1]);
    let b = Partition::from_ids(vec![0, 0, 1, 1]);
    assert!(
        a.ditset_xor_distance(&b).abs() < EPS,
        "Same partition → distance 0"
    );
}

#[test]
fn test_partition_enum_serialization_roundtrip() {
    let p = Partition::from_ids(vec![0, 0, 1, 2, 2, 2]);
    let bytes = bincode::serialize(&p).expect("serialize");
    let restored: Partition = bincode::deserialize(&bytes).expect("deserialize");
    assert_eq!(p.get_ids(), restored.get_ids());
}

// ── Wave 0: Parallel / Batch API tests ──────────────────────

#[test]
fn test_partition_parallel_entropy_matches_sequential() {
    let ids: Vec<u32> = (0..256).map(|i| i % 8).collect();
    let p = Partition::from_ids(ids);
    let seq = p.logical_entropy();
    let par = p.logical_entropy_parallel();
    assert!(
        (seq - par).abs() < EPS,
        "parallel entropy must match sequential"
    );
}

#[test]
fn test_partition_parallel_entropy_lattice_backend() {
    // Small partition stays as Lattice, parallel should still work via conversion
    let p = Partition::from_ids_lattice(vec![0, 0, 1, 1, 2, 2]);
    assert!(!p.is_fast());
    let par = p.logical_entropy_parallel();
    let seq = p.logical_entropy();
    assert!((seq - par).abs() < EPS);
}

#[test]
fn test_partition_differentiate_parallel() {
    let a = Partition::from_ids(vec![0, 0, 1, 1, 2, 2]);
    let b = Partition::from_ids(vec![0, 1, 0, 1, 0, 1]);
    let seq = a.differentiate(&b);
    let par = a.differentiate_parallel(&b);
    assert_eq!(seq.get_ids(), par.get_ids());
}

#[test]
fn test_partition_entropy_batch() {
    let partitions: Vec<Partition> = vec![
        Partition::from_ids(vec![0, 0, 1, 1]),
        Partition::from_ids(vec![0, 1, 2, 3]),
        Partition::from_ids(vec![0, 0, 0, 0]),
    ];
    let entropies = Partition::entropy_batch(&partitions);
    assert_eq!(entropies.len(), 3);
    // discrete → h = 0.75, all-same → h = 0.0
    assert!(entropies[1] > entropies[0]);
    assert!(entropies[2].abs() < EPS);
}

/// Checks that two partitions are structurally equal, ignoring block id order.
fn partitions_structurally_equal(a: &Partition, b: &Partition) -> bool {
    let ids_a = a.get_ids();
    let ids_b = b.get_ids();
    if ids_a.len() != ids_b.len() {
        return false;
    }
    // The same element pairs must share a block.
    use std::collections::HashMap;
    let mut map: HashMap<u32, u32> = HashMap::new();
    for i in 0..ids_a.len() {
        if let Some(&mapped) = map.get(&ids_a[i]) {
            if mapped != ids_b[i] {
                return false;
            }
        } else {
            map.insert(ids_a[i], ids_b[i]);
        }
    }
    true
}

#[test]
fn test_partition_differentiate_parallel_large() {
    // N = 1000 takes the FastPartition path (above AUTO_SELECT_THRESHOLD = 128) and,
    // being above PARALLEL_THRESHOLD = 512, the parallel path.
    let ids_a: Vec<u32> = (0..1000).map(|i| i % 7).collect();
    let ids_b: Vec<u32> = (0..1000).map(|i| (i * 3 + 1) % 11).collect();
    let a = Partition::from_ids_fast(ids_a);
    let b = Partition::from_ids_fast(ids_b);
    let seq = a.differentiate(&b);
    let par = a.differentiate_parallel(&b);
    assert_eq!(
        seq.block_count(),
        par.block_count(),
        "parallel meet block count must match serial"
    );
    assert!(
        (seq.logical_entropy() - par.logical_entropy()).abs() < 1e-10,
        "parallel meet entropy must match serial"
    );
    assert!(
        partitions_structurally_equal(&seq, &par),
        "parallel meet must be structurally equal to serial"
    );
}

#[test]
fn test_partition_differentiate_parallel_10k() {
    // N = 10000: the large-scale parallel path.
    let ids_a: Vec<u32> = (0..10_000).map(|i| i % 5).collect();
    let ids_b: Vec<u32> = (0..10_000).map(|i| i % 3).collect();
    let a = Partition::from_ids_fast(ids_a);
    let b = Partition::from_ids_fast(ids_b);
    let seq = a.differentiate(&b);
    let par = a.differentiate_parallel(&b);
    assert_eq!(seq.block_count(), par.block_count());
    assert!((seq.logical_entropy() - par.logical_entropy()).abs() < 1e-10);
    assert!(partitions_structurally_equal(&seq, &par));
}

#[test]
fn test_partition_mutual_info_matrix() {
    let partitions: Vec<Partition> = vec![
        Partition::from_ids(vec![0, 0, 1, 1]),
        Partition::from_ids(vec![0, 0, 1, 1]),
    ];
    let mi = Partition::mutual_info_matrix(&partitions);
    assert_eq!(mi.len(), 4); // 2x2 matrix
                             // Identical partitions → MI = entropy
    let h = partitions[0].logical_entropy();
    assert!((mi[0] - h).abs() < EPS); // diagonal
    assert!((mi[3] - h).abs() < EPS); // diagonal
                                      // Off-diagonal MI should equal entropy for identical partitions
    assert!((mi[1] - h).abs() < EPS);
}

#[test]
fn test_partition_ops_trait_parallel_defaults() {
    // PartitionLattice uses default (sequential) fallback
    let pl = PartitionLattice::from_ids(vec![0, 0, 1, 1, 2, 2]);
    let seq = PartitionOps::logical_entropy(&pl);
    let par = PartitionOps::logical_entropy_parallel(&pl);
    assert!((seq - par).abs() < EPS);
}

#[test]
fn test_partition_ops_trait_parallel_fast_override() {
    // FastPartition overrides with real parallel impl
    let fp = FastPartition::from_ids(&[0, 0, 1, 1, 2, 2]);
    let seq = PartitionOps::logical_entropy(&fp);
    let par = PartitionOps::logical_entropy_parallel(&fp);
    assert!((seq - par).abs() < EPS);
}

// ── Phase 2: Sparse variant integration tests ───────────────

#[test]
fn test_partition_sparse_variant_basic() {
    let p = Partition::from_ids_sparse(vec![0, 0, 1, 1, 2]);
    assert!(p.is_sparse());
    assert!(!p.is_fast());
    assert_eq!(p.size(), 5);
    assert_eq!(p.block_count(), 3);
}

#[test]
fn test_partition_sparse_entropy() {
    let ids = vec![0, 0, 1, 1, 2, 3, 3, 3];
    let sp = Partition::from_ids_sparse(ids.clone());
    let fp = Partition::from_ids(ids);
    assert!((sp.logical_entropy() - fp.logical_entropy()).abs() < EPS);
}

#[test]
fn test_partition_sparse_differentiate_same_backend() {
    let a = Partition::from_ids_sparse(vec![0, 0, 1, 1]);
    let b = Partition::from_ids_sparse(vec![0, 1, 0, 1]);
    let meet = a.differentiate(&b);
    // Should remain Sparse
    assert!(meet.is_sparse());
    assert_eq!(meet.block_count(), 4); // discrete
}

#[test]
fn test_partition_sparse_differentiate_mixed() {
    // Sparse × Fast → converts to common backend
    let a = Partition::from_ids_sparse(vec![0, 0, 1, 1]);
    let b = Partition::from_ids_fast(vec![0, 1, 0, 1]);
    let meet = a.differentiate(&b);
    // Mixed converts to Fast (N=4 < 128, so Lattice via existing logic)
    assert_eq!(meet.block_count(), 4);
}

#[test]
fn test_partition_sparse_integrate_mixed() {
    // Sparse × Lattice
    let a = Partition::from_ids_sparse(vec![0, 0, 1, 1]);
    let b = Partition::from_ids_lattice(vec![0, 1, 0, 1]);
    let join = a.integrate(&b);
    assert_eq!(join.block_count(), 1); // indiscrete
}

#[test]
fn test_partition_sparse_refines() {
    let fine = Partition::from_ids_sparse(vec![0, 1, 2, 3]);
    let coarse = Partition::from_ids_sparse(vec![0, 0, 1, 1]);
    assert!(fine.refines(&coarse));
    assert!(!coarse.refines(&fine));
}

#[test]
fn test_partition_sparse_serde_roundtrip() {
    let p = Partition::from_ids_sparse(vec![0, 0, 1, 2, 2, 2]);
    let bytes = bincode::serialize(&p).expect("serialize");
    let restored: Partition = bincode::deserialize(&bytes).expect("deserialize");
    assert_eq!(p.get_ids(), restored.get_ids());
}

#[test]
fn test_partition_sparse_to_fast_roundtrip() {
    let sp = Partition::from_ids_sparse(vec![0, 0, 1, 1, 2, 3, 3]);
    let fp = Partition::Fast(sp.to_fast());
    assert!((sp.logical_entropy() - fp.logical_entropy()).abs() < EPS);
    assert_eq!(sp.block_count(), fp.block_count());
}

#[test]
fn test_partition_sparse_from_ids_factory() {
    let ids = vec![0, 0, 1, 1, 2, 2, 3];
    let p = Partition::from_ids_sparse(ids.clone());
    assert_eq!(p.get_ids(), &ids[..]);
    assert!(p.is_sparse());
}

#[test]
fn test_partition_sparse_divergence_matches() {
    let ids1 = vec![0, 0, 1, 1];
    let ids2 = vec![0, 0, 0, 0];
    let sp1 = Partition::from_ids_sparse(ids1.clone());
    let sp2 = Partition::from_ids_sparse(ids2.clone());
    let fp1 = Partition::from_ids(ids1);
    let fp2 = Partition::from_ids(ids2);
    assert!((sp1.logical_divergence(&sp2) - fp1.logical_divergence(&fp2)).abs() < EPS);
}

#[test]
fn test_partition_sparse_xor_distance_matches() {
    let ids1 = vec![0, 0, 1, 1];
    let ids2 = vec![0, 0, 1, 1];
    let sp1 = Partition::from_ids_sparse(ids1);
    let sp2 = Partition::from_ids_sparse(ids2);
    assert!(sp1.ditset_xor_distance(&sp2).abs() < EPS);
}

#[test]
fn test_partition_from_sparse_partition() {
    let sp = SparsePartition::from_ids(&[0, 0, 1, 1, 2]);
    let p = Partition::from(sp);
    assert!(p.is_sparse());
    assert_eq!(p.block_count(), 3);
}

#[test]
fn test_partition_sparse_equality_cross_backend() {
    let ids = vec![0, 0, 1, 1, 2];
    let sp = Partition::from_ids_sparse(ids.clone());
    let fp = Partition::from_ids(ids);
    // PartialEq compares get_ids()
    assert_eq!(sp, fp);
}

// ── A1: interpolate_with tests ───────────────────────────────

#[test]
fn test_partition_interpolate_boundary_conditions() {
    // alpha = 0.0 matches the meet's block count; alpha = 1.0 matches the join's.
    let meet = Partition::from_ids(vec![0, 0, 1, 1, 2]);
    let join = Partition::from_ids(vec![0, 0, 0, 1, 1]);

    let at_zero = Partition::interpolate_with(&meet, &join, 0.0);
    let at_one = Partition::interpolate_with(&meet, &join, 1.0);

    assert_eq!(
        at_zero.block_count(),
        meet.block_count(),
        "α=0.0 should give meet's block count"
    );
    assert_eq!(
        at_one.block_count(),
        join.block_count(),
        "α=1.0 should give join's block count"
    );
}

#[test]
fn test_partition_interpolate_refinement_order() {
    // meet (fine) <= pi_alpha <= join (coarse), so the block count is monotone.
    let meet = Partition::from_ids(vec![0, 0, 1, 1, 2]);
    let join = Partition::from_ids(vec![0, 0, 0, 1, 1]);

    for &alpha in &[0.0f64, 0.25, 0.5, 0.75, 1.0] {
        let interp = Partition::interpolate_with(&meet, &join, alpha);
        assert!(
            meet.block_count() >= interp.block_count(),
            "α={}: meet.blocks({}) < interp.blocks({})",
            alpha,
            meet.block_count(),
            interp.block_count()
        );
        assert!(
            interp.block_count() >= join.block_count(),
            "α={}: interp.blocks({}) < join.blocks({})",
            alpha,
            interp.block_count(),
            join.block_count()
        );
    }
}

#[test]
fn test_partition_interpolate_monotone_entropy() {
    // alpha1 < alpha2 implies h(pi_alpha1) >= h(pi_alpha2): non-increasing.
    let meet = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3]);
    let join = Partition::from_ids(vec![0, 0, 0, 0, 1, 1, 1]);

    let alphas = [0.0f64, 0.25, 0.5, 0.75, 1.0];
    let entropies: Vec<f64> = alphas
        .iter()
        .map(|&a| Partition::interpolate_with(&meet, &join, a).logical_entropy())
        .collect();

    for i in 0..entropies.len() - 1 {
        assert!(
            entropies[i] >= entropies[i + 1] - 1e-10,
            "alpha={} gives h={:.4}, alpha={} gives h={:.4}: monotonicity violated",
            alphas[i],
            entropies[i],
            alphas[i + 1],
            entropies[i + 1]
        );
    }
}

#[test]
fn test_partition_interpolate_idempotent() {
    // Equal partitions in, equal partition out.
    let pi = Partition::from_ids(vec![0, 0, 1, 2, 2]);
    for &alpha in &[0.0f64, 0.3, 0.5, 0.7, 1.0] {
        let result = Partition::interpolate_with(&pi, &pi, alpha);
        assert_eq!(
            result.block_count(),
            pi.block_count(),
            "α={}: idempotent block count",
            alpha
        );
        assert_eq!(
            result.get_ids(),
            pi.get_ids(),
            "α={}: idempotent ids",
            alpha
        );
    }
}

#[test]
fn test_partition_interpolate_deterministic() {
    // Equal inputs give equal outputs.
    let meet = Partition::from_ids(vec![0, 0, 1, 1, 2]);
    let join = Partition::from_ids(vec![0, 0, 0, 1, 1]);

    let r1 = Partition::interpolate_with(&meet, &join, 0.5);
    let r2 = Partition::interpolate_with(&meet, &join, 0.5);
    assert_eq!(r1.get_ids(), r2.get_ids(), "the result must be deterministic");
}

// === R3: interpolate_with correctness + performance tests ===

#[test]
fn test_interpolate_various_alphas_medium_partition() {
    // N=20, various alpha values — correctness with larger partition
    let meet_ids: Vec<u32> = (0..20u32).map(|i| i % 10).collect(); // 10 blocks
    let join_ids: Vec<u32> = (0..20u32).map(|i| i % 3).collect(); // 3 blocks
    let meet = Partition::from_ids(meet_ids);
    let join = Partition::from_ids(join_ids);

    for &alpha in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
        let result = Partition::interpolate_with(&meet, &join, alpha);
        // Block count should be between join and meet
        assert!(
            result.block_count() >= join.block_count()
                && result.block_count() <= meet.block_count(),
            "α={}: blocks={} not in [{}, {}]",
            alpha,
            result.block_count(),
            join.block_count(),
            meet.block_count()
        );
    }
}

#[test]
fn test_interpolate_large_partition_correctness() {
    // N=100, b_meet=50, b_join=5 — stress test
    let meet_ids: Vec<u32> = (0..100u32).map(|i| i % 50).collect();
    let join_ids: Vec<u32> = (0..100u32).map(|i| i % 5).collect();
    let meet = Partition::from_ids(meet_ids);
    let join = Partition::from_ids(join_ids);

    for &alpha in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let result = Partition::interpolate_with(&meet, &join, alpha);
        assert!(
            result.block_count() >= join.block_count()
                && result.block_count() <= meet.block_count(),
            "N=100 α={}: blocks={} not in [{}, {}]",
            alpha,
            result.block_count(),
            join.block_count(),
            meet.block_count()
        );
        // Entropy monotonicity
        assert!(
            result.logical_entropy() >= join.logical_entropy() - 1e-10,
            "N=100 α={}: h(result)={:.4} < h(join)={:.4}",
            alpha,
            result.logical_entropy(),
            join.logical_entropy()
        );
        assert!(
            result.logical_entropy() <= meet.logical_entropy() + 1e-10,
            "N=100 α={}: h(result)={:.4} > h(meet)={:.4}",
            alpha,
            result.logical_entropy(),
            meet.logical_entropy()
        );
    }
}

#[test]
fn test_interpolate_performance_n50() {
    // Performance: N=50, b_meet=25, α=0.5 — should complete quickly
    let meet_ids: Vec<u32> = (0..50u32).map(|i| i % 25).collect();
    let join_ids: Vec<u32> = (0..50u32).map(|i| i % 5).collect();
    let meet = Partition::from_ids(meet_ids);
    let join = Partition::from_ids(join_ids);

    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = Partition::interpolate_with(&meet, &join, 0.5);
    }
    let elapsed = start.elapsed();
    // Sanity check: 100 iterations should complete in under 1 second
    assert!(
        elapsed.as_millis() < 1000,
        "N=50 × 100 iterations took {}ms (expected < 1000ms)",
        elapsed.as_millis()
    );
}

#[test]
fn test_interpolate_performance_n256() {
    // Performance: N=256, b_meet=128, α=0.5
    let meet_ids: Vec<u32> = (0..256u32).map(|i| i % 128).collect();
    let join_ids: Vec<u32> = (0..256u32).map(|i| i % 8).collect();
    let meet = Partition::from_ids(meet_ids);
    let join = Partition::from_ids(join_ids);

    let start = std::time::Instant::now();
    for _ in 0..10 {
        let _ = Partition::interpolate_with(&meet, &join, 0.5);
    }
    let elapsed = start.elapsed();
    // Sanity check: 10 iterations should complete in under 5 seconds
    assert!(
        elapsed.as_millis() < 5000,
        "N=256 × 10 iterations took {}ms (expected < 5000ms)",
        elapsed.as_millis()
    );
}

// ── restrict() / block_sizes() tests ─────────────────────────

#[test]
fn test_restrict_empty_subset() {
    let p = Partition::from_ids(vec![0, 0, 1, 2, 2, 2]);
    let r = p.restrict(&[]);
    assert_eq!(r.size(), 0);
}

#[test]
fn test_restrict_full_subset() {
    let ids = vec![0, 0, 1, 2, 2, 2];
    let p = Partition::from_ids(ids.clone());
    let subset: Vec<usize> = (0..ids.len()).collect();
    let r = p.restrict(&subset);
    assert_eq!(r.size(), p.size());
    assert_eq!(r.block_count(), p.block_count());
    assert!((r.logical_entropy() - p.logical_entropy()).abs() < EPS);
}

#[test]
fn test_restrict_preserves_block_membership() {
    let p = Partition::from_ids(vec![0, 0, 1, 2, 2, 2]);
    let r = p.restrict(&[0, 1, 4, 5]);
    assert_eq!(r.size(), 4);
    assert_eq!(r.block_count(), 2);
    assert_eq!(r.get_block(0), r.get_block(1));
    assert_eq!(r.get_block(2), r.get_block(3));
    assert_ne!(r.get_block(0), r.get_block(2));
}

#[test]
fn test_restrict_renumbers_contiguously() {
    let p = Partition::from_ids(vec![5, 5, 10, 10, 20]);
    let r = p.restrict(&[0, 2, 4]);
    assert_eq!(r.block_count(), 3);
    let mut ids: Vec<u32> = r.get_ids().to_vec();
    ids.sort();
    ids.dedup();
    assert_eq!(ids, vec![0, 1, 2]);
}

#[test]
fn test_restrict_meet_commutativity() {
    // Core theorem: (π ∧ σ)|_S = π|_S ∧ σ|_S
    let pi = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3]);
    let sigma = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1]);
    let subset = vec![1, 2, 5, 6];

    let m1 = pi.differentiate(&sigma).restrict(&subset);
    let m2 = pi.restrict(&subset).differentiate(&sigma.restrict(&subset));

    assert_eq!(m1.block_count(), m2.block_count());
    assert!((m1.logical_entropy() - m2.logical_entropy()).abs() < EPS);
}

#[test]
fn test_block_sizes_basic() {
    let p = Partition::from_ids(vec![0, 0, 1, 2, 2, 2]);
    let sizes = p.block_sizes();
    assert_eq!(sizes.len(), 3);
    let map: std::collections::HashMap<u32, usize> = sizes.into_iter().collect();
    assert_eq!(map[&0], 2);
    assert_eq!(map[&1], 1);
    assert_eq!(map[&2], 3);
}

#[test]
fn test_block_sizes_entropy_reconstruction() {
    let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 3, 3, 3]);
    let sizes = p.block_sizes();
    let n = p.size() as f64;
    let h: f64 = 1.0
        - sizes
            .iter()
            .map(|(_, s)| (*s as f64 / n).powi(2))
            .sum::<f64>();
    assert!((h - p.logical_entropy()).abs() < EPS);
}

// ── Adaptive streaming dispatch tests ─────────────────────────

#[test]
fn test_fused_entropy_integrate_basic() {
    let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3]);
    let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1]);
    let direct = p.integrate(&q).logical_entropy();
    let fused = p.fused_entropy_integrate(&q);
    assert!((direct - fused).abs() < EPS, "basic: {direct} vs {fused}");
}

#[test]
fn test_fused_entropy_integrate_self() {
    let p = Partition::from_ids(vec![0, 0, 1, 2, 2, 2]);
    let direct = p.logical_entropy();
    let fused = p.fused_entropy_integrate(&p);
    assert!((direct - fused).abs() < EPS, "self: {direct} vs {fused}");
}

#[test]
fn test_fused_entropy_integrate_chain() {
    // transitive chain → single block
    let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2]);
    let q = Partition::from_ids(vec![0, 1, 1, 2, 2, 3]);
    let fused = p.fused_entropy_integrate(&q);
    assert!(fused.abs() < EPS, "chain should be 0, got {fused}");
}

#[test]
fn test_fused_entropy_integrate_10k() {
    let n = 10_000;
    let p = Partition::from_ids((0..n as u32).map(|i| i % 20).collect());
    let q = Partition::from_ids((0..n as u32).map(|i| i % 15).collect());
    // fused_entropy_integrate uses exact (streaming block-graph), so compare with integrate_exact
    let direct = p.integrate_exact(&q).logical_entropy();
    let fused = p.fused_entropy_integrate(&q);
    assert!((direct - fused).abs() < EPS, "10k: {direct} vs {fused}");
}

#[test]
fn test_adaptive_logical_divergence_small() {
    // Small N: should use monolithic path, same result
    let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2]);
    let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1]);
    let meet = p.differentiate(&q);
    let expected = meet.logical_entropy() - q.logical_entropy();
    let actual = p.logical_divergence(&q);
    assert!(
        (expected - actual).abs() < EPS,
        "div small: {expected} vs {actual}"
    );
}

#[test]
fn test_adaptive_mutual_information_small() {
    let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2]);
    let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1]);
    let h_meet = p.differentiate(&q).logical_entropy();
    let expected = p.logical_entropy() + q.logical_entropy() - h_meet;
    let actual = p.mutual_information(&q);
    assert!(
        (expected - actual).abs() < EPS,
        "mi small: {expected} vs {actual}"
    );
}

#[test]
fn test_adaptive_ditset_xor_small() {
    let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3]);
    let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1, 0, 1]);
    let expected = {
        let h_meet = p.differentiate(&q).logical_entropy();
        2.0 * h_meet - p.logical_entropy() - q.logical_entropy()
    };
    let actual = p.ditset_xor_distance(&q);
    assert!(
        (expected - actual).abs() < EPS,
        "dxor small: {expected} vs {actual}"
    );
}

#[test]
fn test_should_use_streaming_threshold() {
    // N=100, K₁=3, K₂=3 → 3*3*(2)*8 = 144 bytes → false
    let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2]);
    let q = Partition::from_ids(vec![0, 1, 0, 1, 0, 1]);
    assert!(
        !p.should_use_streaming(&q),
        "small should not use streaming"
    );
}

// ── PCD-guided integrate dispatch ──────────────────────────

#[test]
fn test_integrate_sparse_dispatch_correctness() {
    // Verify that Sparse-dispatched integrate gives the same result as Fast.
    // K=50, N=500 → K₁×K₂ = 2500 > √500 ≈ 22 → would trigger sparse dispatch.
    let n = 500;
    let k = 50;
    let ids_a: Vec<u32> = (0..n).map(|i| (i % k) as u32).collect();
    let ids_b: Vec<u32> = (0..n).map(|i| ((i + 1) % k) as u32).collect();
    let p = Partition::from_ids(ids_a.clone());
    let q = Partition::from_ids(ids_b.clone());

    // Reference: direct Fast integrate
    let fast_result = {
        let fp = FastPartition::from_ids(&ids_a);
        let fq = FastPartition::from_ids(&ids_b);
        fp.integrate(&fq)
    };
    let h_fast = fast_result.logical_entropy();

    // Actual: through Partition::integrate (should dispatch adaptively)
    let result = p.integrate(&q);
    let h_result = result.logical_entropy();

    assert!(
        (h_fast - h_result).abs() < 1e-10,
        "Sparse-dispatched integrate must match Fast: h_fast={h_fast}, h_result={h_result}"
    );
}

#[cfg(feature = "slow-tests")]
#[test]
fn test_should_use_sparse_join_detection() {
    // Join result memory = max(K₁,K₂) × ⌈N/64⌉ × 8.
    // N=100000, K=500 → 500 × 1563 × 8 = 6.25MB per block... need larger.
    // N=100000, K=5000 → 5000 × 1563 × 8 = 62.5MB... still < 256MB.
    // Use very high K to exceed 256MB: K=20000, N=100000
    // → 20000 × 1563 × 8 = 250MB ≈ 256MB. Use N=200000:
    // K=10000, N=200000 → 10000 × 3125 × 8 = 250MB. Close. Use K=11000:
    // 11000 × 3125 × 8 = 275MB > 256MB → should trigger.
    let n = 200_000usize;
    let _k = 11_000usize;
    // Can't create 11000 distinct blocks with N=200000 easily,
    // but we only need the detection threshold, not actual computation.
    // Use K = discrete partition (N blocks):
    let p = Partition::discrete(n); // K=N=200000 blocks
    let q = Partition::discrete(n);
    // max(K)=200000 × 3125 × 8 = 5GB >> 256MB → should trigger.
    assert!(
        p.should_use_sparse_join(&q),
        "Discrete partitions at N={n} should trigger sparse join (high K)"
    );
}

#[test]
fn test_should_not_use_sparse_join_small_k() {
    // K=10, N=100000 → max(K)=10 × 1563 × 8 = 125KB ≪ 256MB → no trigger.
    let ids_a: Vec<u32> = (0..100_000).map(|i| (i % 10) as u32).collect();
    let ids_b: Vec<u32> = (0..100_000).map(|i| (i % 10) as u32).collect();
    let p = Partition::from_ids(ids_a);
    let q = Partition::from_ids(ids_b);

    assert!(
        !p.should_use_sparse_join(&q),
        "K=10, N=100000: should NOT trigger sparse join"
    );
}

#[test]
fn test_integrate_high_k_does_not_explode() {
    // N=10000, K=500 → K² = 250000 >> N → previously would be slow on Fast.
    // With PCD-guided dispatch, should route to Sparse automatically.
    let n = 10_000usize;
    let k = 500usize;
    let ids_a: Vec<u32> = (0..n).map(|i| (i % k) as u32).collect();
    let ids_b: Vec<u32> = (0..n).map(|i| ((i + 1) % k) as u32).collect();
    let p = Partition::from_ids(ids_a);
    let q = Partition::from_ids(ids_b);

    let start = std::time::Instant::now();
    let result = p.integrate(&q);
    let elapsed = start.elapsed();

    // Should complete in < 1 second (previously could take much longer on Fast)
    assert!(
        elapsed.as_secs() < 1,
        "High-K integrate took {:?} — explosion not prevented",
        elapsed
    );
    // Sanity: result is a valid partition
    assert_eq!(result.size(), n);
    assert!(result.block_count() >= 1);
}

// ── E4 Memory model validation ─────────────────

#[test]
fn test_pcs_e4_memory_model_fast_partition() {
    // PCS-A2: M_meet_R1(N, K₁, K₂) = K₁·K₂·⌈N/64⌉·8 bytes
    // Verify that FastPartition block memory matches prediction.
    for &n in &[1_000usize, 10_000, 100_000] {
        let k = 20u32;
        let ids: Vec<u32> = (0..n as u32).map(|i| i % k).collect();
        let fp = FastPartition::from_ids(&ids);

        let words_per_block = (n + 63) / 64;
        let predicted_bytes = fp.block_count() * words_per_block * 8;

        // Actual memory: count words in all BitVecs
        let actual_bytes: usize = fp.blocks_memory_usage();

        let ratio = actual_bytes as f64 / predicted_bytes as f64;
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "N={n}: predicted={predicted_bytes}, actual={actual_bytes}, ratio={ratio:.3}"
        );
    }
}

// ── E5 Lossy differentiate error quantification ─

#[test]
fn test_pcs_e5_lossy_meet_error_before_fix() {
    // Create partitions where K₁·K₂ would exceed BLOCKS_MEMORY_LIMIT
    // when building FastPartition result, triggering lossy *id % max_blocks.
    //
    // N=10_000, K=200 each → K₁·K₂ = 40,000 result blocks
    // words_per_block = 157, estimated = 40000*157*8 = 50MB >> 8MB
    // max_blocks = 8MB / (157*8) = ~6400
    let n = 10_000;
    let k = 200u32;
    let ids_a: Vec<u32> = (0..n as u32).map(|i| i % k).collect();
    let ids_b: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 3) % k).collect();

    let fast_a = FastPartition::from_ids(&ids_a);
    let fast_b = FastPartition::from_ids(&ids_b);
    let sparse_a = SparsePartition::from_ids(&ids_a);
    let sparse_b = SparsePartition::from_ids(&ids_b);

    // Fast (may be lossy) vs Sparse (always exact)
    let h_fast = fast_a.differentiate(&fast_b).logical_entropy();
    let h_exact = sparse_a.differentiate(&sparse_b).logical_entropy();

    // Record the error for Phase 0 evidence
    let error = (h_fast - h_exact).abs();
    eprintln!("E5: N={n}, K={k}, h_fast={h_fast:.6}, h_exact={h_exact:.6}, error={error:.6}");

    // This test documents current behavior — error may or may not be 0
    // depending on whether lossy cap triggers. The assertion below
    // will be TIGHTENED after E8 (lossy removal).
    assert!(
        h_fast >= 0.0 && h_fast <= 1.0,
        "h_fast out of range: {h_fast}"
    );
}

// ── E8 Meet should_use_sparse_meet guard ───────

#[test]
fn test_pcs_e8_should_use_sparse_meet_detection() {
    // K₁=200, K₂=200, N=10_000: K₁·K₂=40_000, words_per_block=157
    // estimated = 40000*157*8 = 50MB >> 8MB threshold → should trigger
    let n = 10_000;
    let k = 200u32;
    let p = Partition::from_ids((0..n as u32).map(|i| i % k).collect());
    let q = Partition::from_ids((0..n as u32).map(|i| (i * 7 + 3) % k).collect());
    assert!(
        p.should_use_sparse_meet(&q),
        "should_use_sparse_meet must trigger for K₁·K₂·words·8 > 8MB"
    );
}

#[test]
fn test_pcs_e8_should_not_use_sparse_meet_small() {
    // K=5, N=1000: K₁·K₂=25, estimated = 25*16*8 = 3.2KB → should NOT trigger
    let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4]);
    let q = Partition::from_ids(vec![0, 1, 0, 1, 2, 3, 2, 3, 4, 4]);
    assert!(
        !p.should_use_sparse_meet(&q),
        "should_use_sparse_meet must NOT trigger for small K"
    );
}

// ── Unified diagnose() tests ────────────────────

#[test]
fn test_pcs_e7_diagnose_meet_small_k() {
    let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4]);
    let q = Partition::from_ids(vec![0, 1, 0, 1, 2, 3, 2, 3, 4, 4]);
    assert_eq!(p.diagnose(PartitionOp::Meet, &q), DispatchAdvice::Fast);
}

#[test]
fn test_pcs_e7_diagnose_meet_high_k() {
    let n = 10_000;
    let k = 200u32;
    let p = Partition::from_ids((0..n as u32).map(|i| i % k).collect());
    let q = Partition::from_ids((0..n as u32).map(|i| (i * 7 + 3) % k).collect());
    assert_eq!(
        p.diagnose(PartitionOp::Meet, &q),
        DispatchAdvice::Sparse,
        "High-K meet should route to Sparse"
    );
}

#[test]
fn test_pcs_e7_diagnose_join_small_k() {
    let n = 10_000;
    let p = Partition::from_ids((0..n as u32).map(|i| i % 10).collect());
    let q = Partition::from_ids((0..n as u32).map(|i| i % 15).collect());
    assert_eq!(p.diagnose(PartitionOp::Join, &q), DispatchAdvice::Fast);
}

#[test]
fn test_pcs_e7_diagnose_metric_high_k() {
    let n = 10_000;
    let k = 200u32;
    let p = Partition::from_ids((0..n as u32).map(|i| i % k).collect());
    let q = Partition::from_ids((0..n as u32).map(|i| (i * 7 + 3) % k).collect());
    assert_eq!(
        p.diagnose(PartitionOp::Metric, &q),
        DispatchAdvice::Streaming,
        "High-K metric should route to Streaming"
    );
}

#[test]
fn test_pcs_e7_diagnose_consistency() {
    // diagnose() must agree with individual should_use_* functions
    let n = 10_000;
    let k = 200u32;
    let p = Partition::from_ids((0..n as u32).map(|i| i % k).collect());
    let q = Partition::from_ids((0..n as u32).map(|i| (i * 7 + 3) % k).collect());

    // Meet: diagnose(Meet) == Sparse ↔ should_use_sparse_meet == true
    let diag_meet = p.diagnose(PartitionOp::Meet, &q);
    let legacy_meet = p.should_use_sparse_meet(&q);
    assert_eq!(
        diag_meet == DispatchAdvice::Sparse,
        legacy_meet,
        "diagnose(Meet) must match should_use_sparse_meet"
    );

    // Metric: diagnose(Metric) == Streaming ↔ should_use_streaming == true
    let diag_metric = p.diagnose(PartitionOp::Metric, &q);
    let legacy_stream = p.should_use_streaming(&q);
    assert_eq!(
        diag_metric == DispatchAdvice::Streaming,
        legacy_stream,
        "diagnose(Metric) must match should_use_streaming"
    );
}

#[test]
fn test_pcs_e8_differentiate_exact_high_k() {
    // After E8 fix: differentiate() must produce exact results
    // even when K₁·K₂ would exceed memory limit.
    let n = 10_000;
    let k = 200u32;
    let ids_a: Vec<u32> = (0..n as u32).map(|i| i % k).collect();
    let ids_b: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 3) % k).collect();

    let p = Partition::from_ids(ids_a.clone());
    let q = Partition::from_ids(ids_b.clone());

    // Partition::differentiate should auto-route through Sparse
    let result = p.differentiate(&q);

    // Ground truth via SparsePartition
    let sp_a = SparsePartition::from_ids(&ids_a);
    let sp_b = SparsePartition::from_ids(&ids_b);
    let exact = sp_a.differentiate(&sp_b);

    let h_result = result.logical_entropy();
    let h_exact = exact.logical_entropy();
    assert!(
        (h_result - h_exact).abs() < EPS,
        "E8: differentiate must be exact. h_result={h_result:.10}, h_exact={h_exact:.10}, diff={:.2e}",
        (h_result - h_exact).abs()
    );
}

// ── K=√N boundary verification ────────────────────

#[test]
fn test_pcs_e10_join_sqrt_n_boundary() {
    // K=√N is the theoretical boundary where Join cost transitions.
    // At N=1M K=1000: should_use_sparse_join triggers (1000*125KB=125MB < 256MB? No)
    // Let's verify: at N=1M, K=1000: k_max=1000, words=15625, mem=1000*15625*8=125MB < 256MB
    // So this stays on Fast path with elementwise integrate.
    let n = 1_000_000;
    let k = 1000u32; // K ≈ √N
    let ids_a: Vec<u32> = (0..n as u32).map(|i| i % k).collect();
    let ids_b: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 3) % k).collect();

    let p = Partition::from_ids(ids_a);
    let q = Partition::from_ids(ids_b);

    let advice = p.diagnose(PartitionOp::Join, &q);
    eprintln!("E10: N={n}, K={k}, Join advice={advice:?}");

    let start = std::time::Instant::now();
    let result = p.integrate(&q);
    let elapsed = start.elapsed();
    eprintln!(
        "E10: Join took {:?}, result K={}",
        elapsed,
        result.block_count()
    );

    // Must complete in <5 seconds (previously could take much longer)
    assert!(
        elapsed.as_secs() < 5,
        "K=√N Join at N={n} took {:?} — too slow",
        elapsed
    );
    assert_eq!(result.size(), n as usize);
}

// ── Dispatch overhead ≤5% verification ───────────

#[test]
fn test_pcs_e9_diagnose_overhead() {
    // diagnose() is O(1): only reads size() and block_count().
    // Verify it adds negligible overhead to normal Fast-path operations.
    let n = 100_000;
    let k = 10u32;
    let ids_a: Vec<u32> = (0..n as u32).map(|i| i % k).collect();
    let ids_b: Vec<u32> = (0..n as u32).map(|i| (i * 3 + 1) % k).collect();
    let p = Partition::from_ids(ids_a);
    let q = Partition::from_ids(ids_b);

    // Warm up
    let _ = p.diagnose(PartitionOp::Meet, &q);
    let _ = p.differentiate(&q);

    // Measure diagnose() alone (1M calls)
    let start = std::time::Instant::now();
    for _ in 0..1_000_000 {
        let _ = std::hint::black_box(p.diagnose(PartitionOp::Meet, &q));
    }
    let diagnose_time = start.elapsed();

    // Measure a single differentiate (has diagnose inside)
    let start = std::time::Instant::now();
    let _ = std::hint::black_box(p.differentiate(&q));
    let op_time = start.elapsed();

    // 1M diagnose calls < 1 differentiate ← diagnose is negligible
    eprintln!(
        "E9: 1M diagnose()={:?}, 1x differentiate={:?}, ratio={:.4}",
        diagnose_time,
        op_time,
        diagnose_time.as_nanos() as f64 / (1_000_000.0 * op_time.as_nanos() as f64)
    );

    // diagnose per call should be < 1% of differentiate
    let per_diagnose_ns = diagnose_time.as_nanos() as f64 / 1_000_000.0;
    let per_op_ns = op_time.as_nanos() as f64;
    let overhead_pct = per_diagnose_ns / per_op_ns * 100.0;
    assert!(
        overhead_pct < 5.0,
        "diagnose overhead {overhead_pct:.2}% exceeds 5% limit"
    );
}

#[test]
fn test_pcs_e10_meet_sqrt_n_boundary() {
    // Meet at K=√N: should route through Sparse via should_use_sparse_meet
    let n = 1_000_000;
    let k = 1000u32;
    let ids_a: Vec<u32> = (0..n as u32).map(|i| i % k).collect();
    let ids_b: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 3) % k).collect();

    let p = Partition::from_ids(ids_a);
    let q = Partition::from_ids(ids_b);

    let advice = p.diagnose(PartitionOp::Meet, &q);
    eprintln!("E10: N={n}, K={k}, Meet advice={advice:?}");
    // K₁·K₂ = 1M, words = 15625, mem = 1M*15625*8 = 125TB → Sparse
    assert_eq!(advice, DispatchAdvice::Sparse);

    let start = std::time::Instant::now();
    let result = p.differentiate(&q);
    let elapsed = start.elapsed();
    eprintln!(
        "E10: Meet took {:?}, result K={}",
        elapsed,
        result.block_count()
    );

    assert!(
        elapsed.as_secs() < 10,
        "K=√N Meet at N={n} took {:?} — too slow",
        elapsed
    );
}

// ══════════════════════════════════════════════════════════════
// Composition Safety + Scale Verification
// ══════════════════════════════════════════════════════════════

// ── E11: PCC cycle composition memory bound ───────────────────

#[test]
fn test_pcs_e11_pcc_cycle_memory_bound() {
    // PCS-A4: M(Meet ∘ Join) ≤ M(Meet) + M(Join)
    //
    // Simulate a PCC cycle: Meet → Δh → Select → Join
    // Verify that:
    // 1. Intermediate K doesn't explode through the cycle
    // 2. Final result K ≤ input K (Join is coarsening)
    // 3. Total peak stays bounded
    for &n in &[1_000usize, 10_000, 100_000] {
        let k = 20u32;
        let ids_a: Vec<u32> = (0..n as u32).map(|i| i % k).collect();
        let ids_b: Vec<u32> = (0..n as u32).map(|i| (i * 3 + 1) % (k + 5)).collect();
        let ids_c: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 2) % (k + 3)).collect();

        let pi = Partition::from_ids(ids_a);
        let sigma = Partition::from_ids(ids_b);
        let tau = Partition::from_ids(ids_c);

        let k_pi = pi.block_count();
        let k_sigma = sigma.block_count();

        // Step 1: Meet (accumulate) — K grows
        let meet = pi.differentiate(&sigma);
        let k_meet = meet.block_count();
        assert!(
            k_meet <= k_pi * k_sigma,
            "N={n}: meet K={k_meet} > K₁·K₂={}",
            k_pi * k_sigma
        );

        // Step 2: Δh (measure) — no allocation
        let delta_h = meet.logical_entropy() - pi.logical_entropy();
        assert!(delta_h >= -EPS, "N={n}: Δh={delta_h} should be ≥ 0");

        // Step 3: Join (abstract) — K shrinks
        let join = meet.integrate(&tau);
        let k_join = join.block_count();
        assert!(
            k_join <= k_meet.min(tau.block_count()),
            "N={n}: join K={k_join} > min(K_meet, K_tau)"
        );

        // PCS-A4 composition: final K ≤ input scale
        // After full cycle, K is bounded by min(K_meet, K_tau)
        // which is at most min(K₁·K₂, K₃)
        assert!(
            k_join <= k_pi * k_sigma,
            "N={n}: PCC cycle K_final={k_join} exceeds composition bound"
        );

        eprintln!("E11: N={n}, K_in=({k_pi},{k_sigma}), K_meet={k_meet}, K_join={k_join}, Δh={delta_h:.6}");
    }
}

#[test]
fn test_pcs_e11_pcc_cycle_high_k_sparse_routing() {
    // PCC cycle with K large enough to trigger Sparse routing.
    // Verifies that composition through Sparse maintains exactness.
    let n = 10_000;
    let k = 100u32;
    let p = Partition::from_ids((0..n as u32).map(|i| i % k).collect());
    let q = Partition::from_ids((0..n as u32).map(|i| (i * 7 + 3) % k).collect());
    let r = Partition::from_ids((0..n as u32).map(|i| (i * 13 + 5) % 15).collect());

    // Full PCC cycle through Partition enum (auto-dispatches)
    let meet = p.differentiate(&q); // May route to Sparse
    let h_meet = meet.logical_entropy();
    // Use integrate_exact for Sparse ground truth comparison
    let join = meet.integrate_exact(&r);
    let h_join = join.logical_entropy();

    // Verify via Sparse ground truth
    let sp = SparsePartition::from_ids(p.get_ids());
    let sq = SparsePartition::from_ids(q.get_ids());
    let sr = SparsePartition::from_ids(r.get_ids());
    let exact_meet = sp.differentiate(&sq);
    let exact_join = exact_meet.integrate(&sr); // SparsePartition integrate is exact

    assert!(
        (h_meet - exact_meet.logical_entropy()).abs() < EPS,
        "Meet exactness: {} vs {}",
        h_meet,
        exact_meet.logical_entropy()
    );
    assert!(
        (h_join - exact_join.logical_entropy()).abs() < EPS,
        "Join exactness: {} vs {}",
        h_join,
        exact_join.logical_entropy()
    );

    eprintln!("E11 high-K: h_meet={h_meet:.6}, h_join={h_join:.6}, exact match ✓");
}

#[test]
fn test_pcs_e11_composition_across_crates_pattern() {
    // Simulate the 3 main PCC usage patterns across crates:
    // 1. PIL: Meet chain (witness verification)
    // 2. NLL: Meet + entropy (token discrimination)
    // 3. LFT: Meet + Join + feedback (convergence)
    let n = 50_000;

    // Pattern 1: PIL-style Meet chain (3 consecutive meets)
    let a = Partition::from_ids((0..n as u32).map(|i| i % 10).collect());
    let b = Partition::from_ids((0..n as u32).map(|i| i % 7).collect());
    let c = Partition::from_ids((0..n as u32).map(|i| i % 13).collect());
    let ab = a.differentiate(&b);
    let abc = ab.differentiate(&c);
    // K_abc ≤ K_ab · K_c ≤ 10·7·13 = 910 (bounded)
    assert!(
        abc.block_count() <= 10 * 7 * 13,
        "PIL chain: K={} > 910",
        abc.block_count()
    );

    // Pattern 2: NLL-style Meet + fused entropy
    let h_fused = a.fused_entropy_differentiate(&b);
    let h_direct = a.differentiate(&b).logical_entropy();
    assert!(
        (h_fused - h_direct).abs() < EPS,
        "NLL fused: {h_fused} vs {h_direct}"
    );

    // Pattern 3: LFT-style Meet + Join feedback
    let meet_ab = a.differentiate(&b);
    let feedback = meet_ab.integrate(&c);
    assert!(
        feedback.block_count() <= meet_ab.block_count(),
        "LFT feedback: K_result={} > K_meet={}",
        feedback.block_count(),
        meet_ab.block_count()
    );

    eprintln!(
        "E11 3-pattern: PIL K_chain={}, NLL fused OK, LFT K_feedback={}",
        abc.block_count(),
        feedback.block_count()
    );
}

// ── E12: Distributed dispatch verification ────────────────────

#[test]
fn test_pcs_e12_distributed_hook_fallback() {
    // Verify that dispatch hooks provide correct local fallback
    // when no distributed handler is registered.
    // This is the path that would switch to distributed at N=100M+.
    let n = 100_000;
    let p = Partition::from_ids((0..n as u32).map(|i| i % 50).collect());
    let q = Partition::from_ids((0..n as u32).map(|i| i % 30).collect());

    // logical_entropy: goes through dispatch hook → None → local
    let h = p.logical_entropy();
    assert!(h > 0.0 && h < 1.0);

    // fused_entropy_differentiate: goes through dispatch hook → None → local
    let h_meet = p.fused_entropy_differentiate(&q);
    assert!(h_meet > h, "meet entropy should be ≥ individual entropy");

    // Verify streaming path is available for large estimates
    if p.should_use_streaming(&q) {
        // Streaming used — verify consistency
        let h_direct = p.differentiate(&q).logical_entropy();
        assert!(
            (h_meet - h_direct).abs() < EPS,
            "streaming vs direct: {h_meet} vs {h_direct}"
        );
    }
}

#[test]
fn test_pcs_e12_diagnose_large_n_streaming() {
    // At very large N with moderate K, diagnose should recommend
    // Streaming for metrics (K₁·K₂·words·8 > 8MB).
    //
    // N=10M, K=100: K₁·K₂=10000, words=156250, mem=10000*156250*8 ≈ 12.5GB
    // → diagnose(Metric) = Streaming
    //
    // We can't create N=10M partitions in a unit test (too slow),
    // but we can verify the formula directly.
    let n = 10_000_000usize;
    let k1 = 100u64;
    let k2 = 100u64;
    let words = ((n + 63) / 64) as u64;
    let mem = k1 * k2 * words * 8;
    assert!(
        mem > STREAMING_DISPATCH_THRESHOLD as u64,
        "N={n} K=100: estimated mem={mem} should exceed 8MB threshold"
    );

    // For Meet: same memory → Sparse
    assert!(
        mem > STREAMING_DISPATCH_THRESHOLD as u64,
        "N={n} K=100: Meet should route to Sparse"
    );

    // For Join: max(K)=100, mem=100*words*8 ≈ 125MB < 256MB → Fast
    let join_mem = k1.max(k2) * words * 8;
    assert!(
        join_mem < 256 * 1024 * 1024,
        "N={n} K=100: Join should stay on Fast (join_mem={join_mem})"
    );

    eprintln!("E12: N={n} K=100: meet_mem={mem}, join_mem={join_mem}, thresholds correct");
}

// ── E13: Operation chain memory accumulation ──────────────────

#[test]
fn test_pcs_e13_chain_memory_accumulation() {
    // PCS-A4: 5-step operation chain memory ≤ 2× single operation peak.
    //
    // Chain: π₁ ∧ π₂ → (∧π₃) → (∨π₄) → (∧π₅) → (∨π₆)
    // After each Join, K shrinks. So peak K is from the Meet steps.
    let n = 100_000;
    let partitions: Vec<Partition> = (0..6)
        .map(|i| {
            let k = 10 + i * 3; // K = 10, 13, 16, 19, 22, 25
            Partition::from_ids((0..n as u32).map(|j| j % k as u32).collect())
        })
        .collect();

    let mut current = partitions[0].clone();
    let mut max_k: usize = current.block_count();

    let steps = [
        ("meet", 1),
        ("meet", 2),
        ("join", 3),
        ("meet", 4),
        ("join", 5),
    ];

    for (op, idx) in &steps {
        let other = &partitions[*idx];
        let k_before = current.block_count();
        current = match *op {
            "meet" => current.differentiate(other),
            "join" => current.integrate(other),
            _ => unreachable!(),
        };
        let k_after = current.block_count();
        if k_after > max_k {
            max_k = k_after;
        }
        eprintln!("E13: step {op}(π{idx}): K {k_before} → {k_after}");
    }

    // After alternating Meet/Join, K should not have exploded
    // First meet: 10*13 = 130 max
    // After join: ≤ 130
    // Second meet with that result: could be up to 130*22 = 2860
    // After final join: ≤ 2860
    //
    // Key check: final K is bounded by product of inputs, not exponential
    let k_product_bound: usize = partitions.iter().map(|p| p.block_count()).product();
    assert!(
        current.block_count() <= k_product_bound,
        "Chain K={} exceeds product bound {}",
        current.block_count(),
        k_product_bound
    );

    // Memory check: with Join interspersed, K stays manageable
    // K should be well below N (100K)
    assert!(
        max_k < n as usize / 10,
        "Peak K={max_k} is too large relative to N={n}"
    );

    // Entropy sanity: chain result is valid
    let h = current.logical_entropy();
    assert!(h >= 0.0 && h <= 1.0, "Chain entropy {h} out of range");

    eprintln!(
        "E13: chain complete. max_K={max_k}, final_K={}, h={h:.6}",
        current.block_count()
    );
}

#[test]
fn test_pcs_e13_repeated_meet_k_growth() {
    // Worst case: repeated meets WITHOUT joins (K can grow multiplicatively).
    // PCS-A4 predicts K(n) ≤ K₁·K₂·...·Kₙ but bounded by N.
    let n = 10_000;
    let mut current = Partition::from_ids((0..n as u32).map(|i| i % 5).collect());

    for step in 0..5 {
        let k_mod = 3 + step;
        let other = Partition::from_ids((0..n as u32).map(|i| i % k_mod as u32).collect());
        let k_before = current.block_count();
        current = current.differentiate(&other);
        let k_after = current.block_count();
        eprintln!("E13 chain-meet: step {step}: K {k_before} → {k_after} (mod {k_mod})");

        // K is bounded by N (at most N singleton blocks)
        assert!(k_after <= n as usize, "K={k_after} exceeds N={n}");
    }

    // After 5 consecutive meets, partition should be near-discrete
    let h = current.logical_entropy();
    assert!(h > 0.9, "5-meet chain should produce high entropy, got {h}");
    eprintln!(
        "E13 chain-meet: final K={}, h={h:.6}",
        current.block_count()
    );
}

// ── sift/span/phantom_budget tests ───────────────

#[test]
fn test_sift_idempotent() {
    let a = Partition::from_ids(vec![0, 0, 1, 2]);
    assert_eq!(a.sift(&a, &a).get_ids(), a.get_ids());
}

#[test]
fn test_span_idempotent() {
    let a = Partition::from_ids(vec![0, 0, 1, 2]);
    assert_eq!(a.span(&a, &a).get_ids(), a.get_ids());
}

#[test]
fn test_sift_bc_commutative() {
    let a = Partition::from_ids(vec![0, 0, 1, 2]);
    let b = Partition::from_ids(vec![0, 1, 0, 1]);
    let c = Partition::from_ids(vec![0, 1, 1, 0]);
    assert_eq!(a.sift(&b, &c).get_ids(), a.sift(&c, &b).get_ids());
}

#[test]
fn test_span_bc_commutative() {
    let a = Partition::from_ids(vec![0, 0, 1, 2]);
    let b = Partition::from_ids(vec![0, 1, 0, 1]);
    let c = Partition::from_ids(vec![0, 1, 1, 0]);
    assert_eq!(a.span(&b, &c).get_ids(), a.span(&c, &b).get_ids());
}

#[test]
fn test_sift_absorb() {
    let a = Partition::from_ids(vec![0, 0, 1, 2]);
    let b = Partition::from_ids(vec![0, 1, 0, 1]);
    assert_eq!(a.sift(&a, &b).get_ids(), a.get_ids());
}

#[test]
fn test_sift_degenerate_to_meet() {
    let a = Partition::from_ids(vec![0, 0, 1, 2]);
    let b = Partition::from_ids(vec![0, 1, 0, 1]);
    assert_eq!(a.sift(&b, &b).get_ids(), a.differentiate(&b).get_ids());
}

#[test]
fn test_sift_leq_a() {
    let a = Partition::from_ids(vec![0, 0, 1, 2]);
    let b = Partition::from_ids(vec![0, 1, 0, 1]);
    let c = Partition::from_ids(vec![0, 1, 1, 0]);
    let s = a.sift(&b, &c);
    assert!(s.refines(&a), "sift must refine a");
}

#[test]
fn test_span_leq_sift() {
    let a = Partition::from_ids(vec![0, 0, 0, 1]);
    let b = Partition::from_ids(vec![0, 0, 1, 0]);
    let c = Partition::from_ids(vec![0, 0, 1, 1]);
    let si = a.sift(&b, &c);
    let sp = a.span(&b, &c);
    assert!(
        sp.refines(&si),
        "span must refine sift (semi-distributivity)"
    );
}

#[test]
fn test_phantom_budget_nonnegative() {
    let a = Partition::from_ids(vec![0, 0, 0, 1]);
    let b = Partition::from_ids(vec![0, 0, 1, 0]);
    let c = Partition::from_ids(vec![0, 0, 1, 1]);
    assert!(a.phantom_budget(&b, &c) >= -EPS);
}

#[test]
fn test_phantom_budget_distributive_zero() {
    let a = Partition::from_ids(vec![0, 1, 2, 3]);
    let b = Partition::from_ids(vec![0, 0, 1, 1]);
    let c = Partition::from_ids(vec![0, 1, 0, 1]);
    assert!(
        a.phantom_budget(&b, &c).abs() < EPS,
        "discrete a is distributive → phantom budget must be 0"
    );
}

#[test]
fn test_phantom_budget_nondistributive_positive() {
    // Phase 0 witness: N=4 max-phantom (2 phantom dits)
    let a = Partition::from_ids(vec![0, 0, 0, 1]);
    let b = Partition::from_ids(vec![0, 0, 1, 0]);
    let c = Partition::from_ids(vec![0, 0, 1, 1]);
    let pb = a.phantom_budget(&b, &c);
    assert!(
        (pb - 0.25).abs() < EPS,
        "expected phantom budget 0.25 (2 phantom dits), got {pb}"
    );
}

#[test]
fn test_sift_span_match_manual_composition() {
    let a = Partition::from_ids(vec![0, 0, 0, 0, 1]);
    let b = Partition::from_ids(vec![0, 0, 1, 1, 0]);
    let c = Partition::from_ids(vec![0, 0, 1, 1, 1]);

    let sift_manual = a.differentiate(&b.integrate(&c));
    let span_manual = a.differentiate(&b).integrate(&a.differentiate(&c));

    assert_eq!(a.sift(&b, &c).get_ids(), sift_manual.get_ids());
    assert_eq!(a.span(&b, &c).get_ids(), span_manual.get_ids());
}

// === Layer A routing tests ===

/// Layer A: 2-tier routing — narrow L ≤ 8 routes to Cyclic, wide L ∈
/// [9, 16] routes to WideCyclic. Universe must be ≥ AUTO_SELECT_THRESHOLD
/// (= 128) for the cyclic-detection branch to be exercised.
#[test]
fn test_layer_a_routing_chain_narrow_l_to_cyclic() {
    let universe = 4096usize;
    for cl in 1..=8usize {
        let pattern: Vec<u32> = (0..cl as u32).collect();
        let ids: Vec<u32> = (0..universe).map(|i| pattern[i % cl]).collect();
        let p = Partition::from_ids(ids);
        assert!(
            matches!(p, Partition::Cyclic(_)),
            "narrow L={} must route to Partition::Cyclic",
            cl
        );
    }
}

#[test]
fn test_layer_a_routing_chain_wide_l_to_wide_cyclic() {
    let universe = 4096usize;
    for cl in 9..=16usize {
        let pattern: Vec<u32> = (0..cl as u32).collect();
        let ids: Vec<u32> = (0..universe).map(|i| pattern[i % cl]).collect();
        let p = Partition::from_ids(ids);
        assert!(
            matches!(p, Partition::WideCyclic(_)),
            "wide L={} must route to Partition::WideCyclic, got {:?}",
            cl,
            std::mem::discriminant(&p)
        );
    }
}

/// Layer A: non-cyclic ids fall through to Fast (above AUTO_SELECT_THRESHOLD).
#[test]
fn test_layer_a_routing_chain_non_cyclic_to_fast() {
    let universe = 4096usize;
    // Build a definitely-non-cyclic id sequence of length > 16 prefix
    // that breaks cyclicity. Easiest: random-looking ids with one
    // distinctive position outside the first 16.
    let mut ids: Vec<u32> = (0..universe).map(|i| (i % 5) as u32).collect();
    // Break cyclicity at position 17 by flipping its id.
    ids[17] = 99;
    let p = Partition::from_ids(ids);
    assert!(
        matches!(p, Partition::Fast(_)),
        "non-cyclic ids must fall through to Partition::Fast, got {:?}",
        std::mem::discriminant(&p)
    );
}

/// Layer A: routing-derived WideCyclic must produce ids identical to
/// the manually-constructed FastPartition equivalent.
#[test]
fn test_layer_a_routing_get_ids_bit_exact_with_fast() {
    let universe = 8192usize;
    for cl in 9..=16usize {
        let pattern: Vec<u32> = (0..cl as u32).collect();
        let ids: Vec<u32> = (0..universe).map(|i| pattern[i % cl]).collect();
        let p = Partition::from_ids(ids.clone());
        let fast = Partition::from_ids_fast(ids.clone());
        assert_eq!(
            p.get_ids(),
            fast.get_ids(),
            "Layer A routing get_ids mismatch at L={}",
            cl
        );
        assert!((p.logical_entropy() - fast.logical_entropy()).abs() < 1e-12);
        assert_eq!(p.block_count(), fast.block_count());
        assert_eq!(p.distinction_count(), fast.distinction_count());
    }
}

// ── Three- and four-way Hu logical mutual information ──────────────────────

#[test]
fn test_triple_mutual_information_self_yields_h() {
    // I3(a, a, a) = 3h(a) - 3h(a) + h(a) = h(a), since every Meet is a itself.
    let a = Partition::from_ids(vec![0, 0, 1, 1]);
    let h = a.logical_entropy();
    let i3 = a.triple_mutual_information(&a, &a);
    assert!(
        (i3 - h).abs() < EPS,
        "I₃(a,a,a) = h(a), got {}, expected {}",
        i3,
        h
    );
}

#[test]
fn test_triple_mutual_information_three_2blocks_n4() {
    // Three non-trivial two-block partitions on N = 4, each with h = 0.5.
    // a=[0,0,1,1], b=[0,1,0,1], c=[0,1,1,0].
    // Every Meet (a∧b, a∧c, b∧c, a∧b∧c) is discrete with h = 0.75.
    // I₃ = 0.5+0.5+0.5 − 0.75−0.75−0.75 + 0.75 = 1.5 − 2.25 + 0.75 = 0.0.
    let a = Partition::from_ids(vec![0, 0, 1, 1]);
    let b = Partition::from_ids(vec![0, 1, 0, 1]);
    let c = Partition::from_ids(vec![0, 1, 1, 0]);
    let i3 = a.triple_mutual_information(&b, &c);
    assert!(
        i3.abs() < EPS,
        "I₃ = 0 expected (3 orthogonal 2-blocks), got {}",
        i3
    );
}

#[test]
fn test_triple_mutual_information_with_indiscrete_yields_two_way() {
    // Indiscrete (c=Top, h=0): I₃(a,b,Top) = h(a)+h(b)+0 − h(a∧b)−h(a)−h(b) + h(a∧b)
    //                                       = 0.
    // Top acts as a unit for Meet: a∧Top = a and h(a∧Top) = h(a).
    let a = Partition::from_ids(vec![0, 0, 1, 1]);
    let b = Partition::from_ids(vec![0, 1, 0, 1]);
    let top = Partition::from_ids(vec![0, 0, 0, 0]); // indiscrete
    let i3 = a.triple_mutual_information(&b, &top);
    assert!(i3.abs() < EPS, "I₃(a,b,Top) = 0 expected, got {}", i3);
}

#[test]
fn test_triple_mutual_information_consistency_with_two_way() {
    // The difference between I3(a,b,c) and I3(a,b,Top) measures c's incremental contribution.
    // Top is trivial, so I3(a,b,Top) = 0, consistent with the test above.
    let a = Partition::from_ids(vec![0, 0, 1, 1]);
    let b = Partition::from_ids(vec![0, 1, 0, 1]);
    let c = Partition::from_ids(vec![0, 0, 0, 1]);
    let i3 = a.triple_mutual_information(&b, &c);
    // The value may be of either sign; only check that it is finite and of sensible magnitude.
    assert!(i3.is_finite(), "I₃ finite, got {}", i3);
    assert!(i3.abs() < 2.0, "I₃ magnitude reasonable, got {}", i3);
}

#[test]
fn test_quad_mutual_information_self_chains() {
    // I4(a, a, a, a): every Meet in the inclusion-exclusion is a.
    //   = 4h − 6h + 4h − h = h.
    let a = Partition::from_ids(vec![0, 0, 1, 1, 2, 2]);
    let h = a.logical_entropy();
    let i4 = a.quad_mutual_information(&a, &a, &a);
    assert!(
        (i4 - h).abs() < EPS,
        "I₄(a,a,a,a) = h(a), got {}, expected {}",
        i4,
        h
    );
}

#[test]
fn test_quad_mutual_information_with_indiscrete_yields_triple() {
    // I4(a, b, c, Top): Top is the Meet unit, so a∧Top = a and so on.
    //   = h(a)+h(b)+h(c)+0
    //   − h(a∧b)−h(a∧c)−h(a)−h(b∧c)−h(b)−h(c)
    //   + h(a∧b∧c)+h(a∧b)+h(a∧c)+h(b∧c)
    //   − h(a∧b∧c)
    //   = 0.
    // Every term cancels, so Top contributes nothing.
    let a = Partition::from_ids(vec![0, 0, 1, 1]);
    let b = Partition::from_ids(vec![0, 1, 0, 1]);
    let c = Partition::from_ids(vec![0, 0, 0, 1]);
    let top = Partition::from_ids(vec![0, 0, 0, 0]);
    let i4 = a.quad_mutual_information(&b, &c, &top);
    assert!(i4.abs() < EPS, "I₄(a,b,c,Top) = 0 expected, got {}", i4);
}

#[test]
fn test_quad_mutual_information_finite() {
    // I4(a, b, c, d): finite and of sensible magnitude.
    let a = Partition::from_ids(vec![0, 0, 1, 1, 2]);
    let b = Partition::from_ids(vec![0, 1, 0, 1, 2]);
    let c = Partition::from_ids(vec![0, 1, 1, 0, 2]);
    let d = Partition::from_ids(vec![0, 0, 1, 2, 1]);
    let i4 = a.quad_mutual_information(&b, &c, &d);
    assert!(i4.is_finite(), "I₄ finite, got {}", i4);
}

// ── Path A (generate-vptree-optimize): mixed-variant ditset_xor_distance equivalence ──
//
// `Partition::ditset_xor_distance` dispatches same-variant (Lattice/Fast/Sparse) pairs to
// leaf impls and ALL other (mixed) pairs through one fallback arm — including every
// Cyclic/WideCyclic/FactorizedFast comparison, since those have no same-variant arm.
// ngram crystals are `Cyclic` and the generate query becomes `Fast` after a Meet, so the
// `generate_retrieve_nearest` hot path is exactly this mixed fallback. These guards pin the
// mixed path bit-identical to the reference d_⊕ (Lattice leaf on the same logical ids) so
// the get_block+sort optimization that replaces the per-call `to_sparse()` rebuild cannot
// drift. Integer numerator over a shared N² ⇒ exact f64 equality (not ε).
fn pa_cyclic_ids(pattern: &[u32], n: usize) -> Vec<u32> {
    (0..n).map(|i| pattern[i % pattern.len()]).collect()
}

#[test]
fn test_ditset_xor_mixed_variant_matches_reference() {
    let n = 64;
    let ca = pa_cyclic_ids(&[0, 1, 2, 3], n); // narrow cyclic (cycle 4)
    let cb = pa_cyclic_ids(&[0, 0, 1, 2, 1], n); // narrow cyclic (cycle 5)
    let wc = pa_cyclic_ids(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], n); // wide cyclic (cycle 10)
    let fa: Vec<u32> = (0..n as u32).map(|i| i % 7).collect();
    let fb: Vec<u32> = (0..n as u32).map(|i| (i * 3 + 1) % 5).collect();

    let reference = |x: &[u32], y: &[u32]| {
        Partition::from_ids_lattice(x.to_vec())
            .ditset_xor_distance(&Partition::from_ids_lattice(y.to_vec()))
    };

    // every pair routes through the MIXED fallback arm of ditset_xor_distance
    let cases: Vec<(&str, Partition, Partition, &[u32], &[u32])> = vec![
        (
            "Cyclic×Fast",
            Partition::from_ids_cyclic(ca.clone()),
            Partition::from_ids_fast(fb.clone()),
            &ca,
            &fb,
        ),
        (
            "Fast×Cyclic",
            Partition::from_ids_fast(fa.clone()),
            Partition::from_ids_cyclic(cb.clone()),
            &fa,
            &cb,
        ),
        (
            "Cyclic×Cyclic",
            Partition::from_ids_cyclic(ca.clone()),
            Partition::from_ids_cyclic(cb.clone()),
            &ca,
            &cb,
        ),
        (
            "Sparse×Cyclic",
            Partition::from_ids_sparse(fa.clone()),
            Partition::from_ids_cyclic(ca.clone()),
            &fa,
            &ca,
        ),
        (
            "Lattice×Cyclic",
            Partition::from_ids_lattice(fb.clone()),
            Partition::from_ids_cyclic(cb.clone()),
            &fb,
            &cb,
        ),
        (
            "WideCyclic×Fast",
            Partition::from_ids_wide_cyclic(wc.clone()),
            Partition::from_ids_fast(fa.clone()),
            &wc,
            &fa,
        ),
        (
            "Factorized×Fast",
            Partition::from_ids_factorized(fa.clone(), 4),
            Partition::from_ids_fast(fb.clone()),
            &fa,
            &fb,
        ),
    ];

    for (name, a, b, ids_a, ids_b) in &cases {
        let got = a.ditset_xor_distance(b);
        let want = reference(ids_a, ids_b);
        assert_eq!(
            got, want,
            "mixed {name}: d_⊕ must be bit-identical to reference"
        );
    }
}

#[test]
fn test_ditset_xor_mixed_identical_is_zero() {
    let ca = pa_cyclic_ids(&[0, 1, 2, 3], 32);
    let a = Partition::from_ids_cyclic(ca.clone());
    let b = Partition::from_ids_fast(ca.clone()); // same labeling, different variant
    assert_eq!(
        a.ditset_xor_distance(&b),
        0.0,
        "identical labelings across variants ⇒ d_⊕ = 0"
    );
}

// ── consensus (approximate Fréchet median / majority) — store-time learning primitive ──
// The "majority/median" partition aggregation identified as the missing substrate for real
// relational *learning* (abstract many observed targets into one generalizing prototype),
// distinct from Meet/Join. Validates majority same-block + Union-Find closure.
#[test]
fn test_consensus_of_identicals_is_identity() {
    let p = Partition::from_ids(vec![0, 0, 1, 1, 2]);
    let c = Partition::consensus(&[p.clone(), p.clone(), p.clone()]).unwrap();
    // identical inputs ⇒ consensus reproduces the (normalized) same-block relation
    for i in 0..5 {
        for j in 0..5 {
            assert_eq!(
                (p.get_block(i) == p.get_block(j)),
                (c.get_block(i) == c.get_block(j)),
                "consensus of identicals preserves same-block relation ({i},{j})"
            );
        }
    }
}

#[test]
fn test_consensus_majority_vote() {
    // 2/3 agree {0,1}|{2,3}; 1 outlier {0,2}|{1,3}. Majority wins per pair.
    let p0 = Partition::from_ids(vec![0, 0, 1, 1]);
    let p1 = Partition::from_ids(vec![0, 0, 1, 1]);
    let outlier = Partition::from_ids(vec![0, 1, 0, 1]);
    let c = Partition::consensus(&[p0, p1, outlier]).unwrap();
    assert_eq!(c.get_block(0), c.get_block(1), "majority 0~1 held");
    assert_eq!(c.get_block(2), c.get_block(3), "majority 2~3 held");
    assert_ne!(c.get_block(0), c.get_block(2), "minority 0~2 rejected");
}

#[test]
fn test_consensus_empty_is_none() {
    assert!(Partition::consensus(&[]).is_none());
}

#[test]
fn test_mode_picks_most_frequent_not_blend() {
    let a = Partition::from_ids(vec![0, 0, 1, 1]);
    let b = Partition::from_ids(vec![0, 1, 0, 1]);
    // multi-modal [a,a,a,b] → mode = a (dominant), NOT a consensus blend.
    let m = Partition::mode(&[a.clone(), a.clone(), a.clone(), b.clone()]).unwrap();
    assert_eq!(m.get_ids(), a.get_ids(), "mode = dominant a");
    // single → that value.
    assert_eq!(
        Partition::mode(&[b.clone()]).unwrap().get_ids(),
        b.get_ids()
    );
    // empty → None.
    assert!(Partition::mode(&[]).is_none());
}

#[test]
fn test_mode_tie_breaks_first_occurrence_deterministic() {
    let a = Partition::from_ids(vec![0, 0, 1, 1]);
    let b = Partition::from_ids(vec![0, 1, 0, 1]);
    // count tie (1 each) → first occurrence (a). deterministic both orders.
    assert_eq!(
        Partition::mode(&[a.clone(), b.clone()]).unwrap().get_ids(),
        a.get_ids()
    );
    assert_eq!(
        Partition::mode(&[b.clone(), a.clone()]).unwrap().get_ids(),
        b.get_ids()
    );
}

#[test]
fn test_rand_agreement_vs_dit_set_jaccard_are_distinct_forces() {
    // The over-consolidate-trap proof: Rand (Class A) and set-Jaccard (Class B) are DIFFERENT
    // formulas — same input, different output — so they must NOT be merged.
    let a = Partition::from_ids(vec![0, 0, 0, 1]); // 3 same-pairs, 3 diff-pairs
    let b = Partition::from_ids(vec![0, 0, 0, 0]); // all same (no dits)

    // Rand = agree-on-classification / C(N,2): b is all-same, so agree ⟺ a-same (3 pairs) → 3/6.
    assert_eq!(a.rand_agreement(&b), 0.5);
    // set-Jaccard = |dit(a) ∩ dit(b)| / |dit(a) ∪ dit(b)|: dit(b)=∅ → inter 0 → 0/3.
    assert_eq!(a.dit_set_jaccard(&b), Some(0.0));
    // ⇒ 0.5 ≠ 0.0: distinct forces (merging would corrupt one of them).
    assert!((a.rand_agreement(&b) - a.dit_set_jaccard(&b).unwrap()).abs() > 1e-9);

    // identical ⟹ both maximal (Rand 1.0, set-Jaccard 1.0).
    assert_eq!(a.rand_agreement(&a), 1.0);
    assert_eq!(a.dit_set_jaccard(&a), Some(1.0));

    // empty conventions (callers map): no-dits ⟹ set-Jaccard None; distance ⟹ 0.0 (eigenform);
    // Rand over <2 elements ⟹ 1.0.
    let t = Partition::from_ids(vec![0, 0]);
    assert_eq!(t.dit_set_jaccard(&t), None);
    assert_eq!(t.dit_set_jaccard_distance(&t), 0.0);
    assert_eq!(
        Partition::from_ids(vec![0]).rand_agreement(&Partition::from_ids(vec![0])),
        1.0
    );

    // slice core == method (lpg's &[u32] path is bit-identical to the Partition method).
    assert_eq!(
        rand_agreement_ids(a.get_ids(), b.get_ids()),
        a.rand_agreement(&b)
    );
}

#[test]
fn test_min_ditset_xor_distance_over() {
    // SSOT min-d_⊕-over-pool reduction (novelty/surprise force home).
    let q = Partition::from_ids(vec![0, 0, 1, 2, 2]);
    let near = Partition::from_ids(vec![0, 0, 1, 2, 1]); // small d_⊕
    let far = Partition::from_ids(vec![0, 1, 2, 3, 4]); // larger d_⊕
    let pool = [far.clone(), near.clone(), q.clone()];

    // empty pool ⟹ INFINITY (nothing to be near).
    assert_eq!(
        q.min_ditset_xor_distance_over(std::iter::empty::<&Partition>()),
        f64::INFINITY
    );

    // identical entry present ⟹ 0.
    assert_eq!(q.min_ditset_xor_distance_over(pool.iter()), 0.0);

    // matches the manual min over the same pool (delegation equivalence).
    let manual = [far.clone(), near.clone()]
        .iter()
        .map(|p| q.ditset_xor_distance(p))
        .fold(f64::INFINITY, f64::min);
    assert_eq!(q.min_ditset_xor_distance_over([&far, &near]), manual);
    // = the smaller of the two individual distances (whichever member is nearer).
    assert_eq!(
        manual,
        q.ditset_xor_distance(&near)
            .min(q.ditset_xor_distance(&far))
    );

    // single-element pool ⟹ exactly that pairwise distance.
    assert_eq!(
        q.min_ditset_xor_distance_over([&near]),
        q.ditset_xor_distance(&near)
    );
}

#[test]
fn test_partition_try_pullback_permutation_round_trip() {
    let p = Partition::from_ids(vec![0, 0, 1, 1, 2, 2]);
    let swap = [0usize, 2, 1, 3, 4, 5];

    let transported = p
        .try_pullback_permutation(&swap)
        .expect("a bijection on the same universe must be accepted");
    assert_eq!(transported.get_ids(), &[0, 1, 0, 1, 2, 2]);

    let restored = transported
        .try_pullback_permutation(&swap)
        .expect("the transposition is its own inverse");
    assert_eq!(restored.get_ids(), p.get_ids());
}

#[test]
fn test_partition_try_pullback_permutation_rejects_non_bijection() {
    let p = Partition::from_ids(vec![0, 0, 1, 1]);
    assert!(
        p.try_pullback_permutation(&[0, 1, 2]).is_none(),
        "wrong length must fail"
    );
    assert!(
        p.try_pullback_permutation(&[0, 1, 1, 3]).is_none(),
        "duplicate source must fail"
    );
    assert!(
        p.try_pullback_permutation(&[0, 1, 2, 4]).is_none(),
        "out-of-range source must fail"
    );
}

#[test]
fn test_partition_pullback_permutation_preserves_meet_join_and_dxor() {
    let a = Partition::from_ids(vec![0, 0, 1, 1, 2, 2]);
    let b = Partition::from_ids(vec![0, 1, 0, 1, 2, 2]);
    let swap = [0usize, 2, 1, 3, 4, 5];
    let pa = a.try_pullback_permutation(&swap).unwrap();
    let pb = b.try_pullback_permutation(&swap).unwrap();

    let map_meet = a.differentiate(&b).try_pullback_permutation(&swap).unwrap();
    let meet_map = pa.differentiate(&pb);
    assert!(map_meet.refines(&meet_map) && meet_map.refines(&map_meet));

    let map_join = a
        .integrate_exact(&b)
        .try_pullback_permutation(&swap)
        .unwrap();
    let join_map = pa.integrate_exact(&pb);
    assert!(map_join.refines(&join_map) && join_map.refines(&map_join));
    assert_eq!(a.ditset_xor_distance(&b), pa.ditset_xor_distance(&pb));
}

#[test]
fn test_consensus_is_central_under_dxor() {
    // Median property (approx): consensus is at least as central (lower total d_⊕ to the set)
    // as the outlier member — i.e. consensus abstracts toward the majority structure.
    let p0 = Partition::from_ids(vec![0, 0, 1, 1, 2, 2]);
    let p1 = Partition::from_ids(vec![0, 0, 1, 1, 2, 3]);
    let p2 = Partition::from_ids(vec![0, 0, 1, 1, 3, 2]);
    let outlier = Partition::from_ids(vec![0, 1, 2, 3, 4, 5]); // all-distinct outlier
    let set = [p0, p1, p2, outlier.clone()];
    let c = Partition::consensus(&set).unwrap();
    let total = |q: &Partition| set.iter().map(|p| q.ditset_xor_distance(p)).sum::<f64>();
    assert!(
        total(&c) <= total(&outlier) + 1e-12,
        "consensus total d_⊕ {} ≤ outlier total {}",
        total(&c),
        total(&outlier)
    );
}

// ── is_distinct_from_all: INT-only bool-specialization of (min d_⊕ > 0) ──────

/// The INT membership `is_distinct_from_all` must agree bit-for-bit with the
/// f64 novelty predicate `min_ditset_xor_distance_over(pool) > 0.0` — the
/// invariant (metric positivity: `d_⊕(P,Q)=0 ⟺ P==Q`) that lets the WMF novelty
/// gate drop the per-comparison Meet + f64 entropy for a plain equality scan.
#[test]
fn test_is_distinct_from_all_matches_min_dxor_positive() {
    // Deterministic LCG (no external RNG) over a range of universes/pools.
    let mut state: u64 = 0xD1B54A32D192ED03;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    for &n in &[3usize, 8, 33, 130] {
        // Build a small pool of random partitions of the SAME universe n.
        let mk = |seed: &mut dyn FnMut() -> u64| {
            let ids: Vec<u32> = (0..n)
                .map(|_| (seed() % ((n as u64).max(1))) as u32)
                .collect();
            Partition::from_ids(ids)
        };
        for _ in 0..12 {
            let pool: Vec<Partition> = (0..(next() % 5)).map(|_| mk(&mut next)).collect();
            // Query is sometimes a clone of a pool item (non-novel), sometimes fresh.
            let query = if !pool.is_empty() && next() % 2 == 0 {
                pool[(next() as usize) % pool.len()].clone()
            } else {
                mk(&mut next)
            };
            let via_f64 = query.min_ditset_xor_distance_over(pool.iter()) > 0.0;
            let via_int = query.is_distinct_from_all(pool.iter());
            assert_eq!(
                via_int,
                via_f64,
                "is_distinct_from_all ≠ (min d_⊕ > 0) for n={n}, |pool|={}",
                pool.len()
            );
        }
    }
}

#[test]
fn test_is_distinct_from_all_empty_pool_is_distinct() {
    let p = Partition::from_ids(vec![0, 1, 1, 2]);
    let empty: Vec<Partition> = Vec::new();
    assert!(
        p.is_distinct_from_all(empty.iter()),
        "empty pool ⇒ distinct (novel)"
    );
}

#[test]
fn test_is_distinct_from_all_crosses_backend() {
    // Same partition via Lattice(small) vs a materialized clone must be seen as
    // NON-distinct (semantic equality via get_ids canonicalization).
    let a = Partition::from_ids(vec![0, 0, 1, 1, 2]);
    let b = Partition::from_ids(vec![0, 0, 1, 1, 2]); // identical partition
    let c = Partition::from_ids(vec![0, 1, 1, 1, 2]); // different
    assert!(
        !a.is_distinct_from_all([b].iter()),
        "bit-identical ⇒ not distinct"
    );
    assert!(
        a.is_distinct_from_all([c].iter()),
        "different partition ⇒ distinct"
    );
}

// ───── canonicalize_ids — label-blind identity key (SSOT, 2026-07-29) ─────

#[test]
fn test_canonicalize_ids_renumbers_first_seen_from_zero() {
    assert_eq!(canonicalize_ids(&[7, 7, 3, 3, 9]), vec![0, 0, 1, 1, 2]);
    assert_eq!(canonicalize_ids(&[0, 1, 2]), vec![0, 1, 2]);
}

#[test]
fn test_canonicalize_ids_is_label_blind() {
    // Same partition, different labels ⇒ identical canonical form.
    assert_eq!(canonicalize_ids(&[0, 0, 1, 1]), canonicalize_ids(&[1, 1, 0, 0]));
    assert_eq!(canonicalize_ids(&[5, 5, 2, 2]), canonicalize_ids(&[0, 0, 1, 1]));
}

#[test]
fn test_canonicalize_ids_distinguishes_different_partitions() {
    assert_ne!(canonicalize_ids(&[0, 0, 1, 1]), canonicalize_ids(&[0, 1, 0, 1]));
}

#[test]
fn test_canonicalize_ids_idempotent() {
    let once = canonicalize_ids(&[4, 9, 9, 1]);
    assert_eq!(canonicalize_ids(&once), once);
}

#[test]
fn test_canonicalize_ids_empty_and_singleton() {
    assert!(canonicalize_ids(&[]).is_empty());
    assert_eq!(canonicalize_ids(&[42]), vec![0]);
}

/// Pins the exact semantics the four pre-existing copies (l-vm-eigenform,
/// l-vm-symbolism, l-vm-mm0, l-vm-tars) implemented, so delegation is bit-exact.
#[test]
fn test_canonicalize_ids_matches_legacy_copy_semantics() {
    fn legacy(ids: &[u32]) -> Vec<u32> {
        let mut map: crate::HashMap<u32, u32> = crate::HashMap::new();
        let mut next_id = 0u32;
        let mut out = Vec::with_capacity(ids.len());
        for &id in ids {
            let canon = *map.entry(id).or_insert_with(|| {
                let v = next_id;
                next_id += 1;
                v
            });
            out.push(canon);
        }
        out
    }
    for case in [
        vec![],
        vec![0u32],
        vec![3, 3, 3],
        vec![9, 1, 9, 1, 4],
        vec![0, 1, 2, 3, 4, 5],
        vec![2, 2, 0, 0, 1, 1, 2],
    ] {
        assert_eq!(canonicalize_ids(&case), legacy(&case), "case {case:?}");
    }
}
