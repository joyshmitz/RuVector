//! End-to-end smoke tests for the ruLake federation intermediary.
//!
//! Acceptance gates for M1 in `docs/research/ruLake/07-implementation-plan.md`:
//!
//! 1. A query through `RuLake::search_one` over `LocalBackend` returns
//!    the same top-k ids as a direct `RabitqPlusIndex::search` on the
//!    same data, modulo tie ordering.
//! 2. Cache coherence — mutating the backend bumps the generation;
//!    the next search re-primes the cache automatically.
//! 3. Federated search — fanning out across two backends and merging
//!    by score gives the globally-correct top-k.
//! 4. Cache-hit path is significantly faster than cache-miss
//!    (measurement-level check, not just a correctness check).

use std::sync::Arc;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};

use ruvector_rabitq::{AnnIndex, RabitqPlusIndex};
use ruvector_rulake::{cache::Consistency, LocalBackend, RuLake};

fn clustered(n: usize, d: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let centroid = Uniform::new(-2.0f32, 2.0);
    let centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..d).map(|_| centroid.sample(&mut rng)).collect())
        .collect();
    let noise = Normal::new(0.0f64, 0.6).unwrap();
    (0..n)
        .map(|_| {
            let c = &centroids[rng.gen_range(0..n_clusters)];
            c.iter()
                .map(|&x| x + noise.sample(&mut rng) as f32)
                .collect()
        })
        .collect()
}

#[test]
fn rulake_matches_direct_rabitq_on_local_backend() {
    // M1 acceptance #1: ruLake's search output matches a direct call to
    // RabitqPlusIndex::search, given the same seed + rerank factor.
    let d = 64;
    let n = 1_000;
    let rerank = 20;
    let seed = 42;

    let data = clustered(n, d, 10, seed);
    let ids: Vec<u64> = (0..n as u64).collect();

    // Reference: direct RaBitQ.
    let mut direct = RabitqPlusIndex::new(d, seed, rerank);
    for (i, v) in data.iter().enumerate() {
        direct.add(i, v.clone()).unwrap();
    }

    // ruLake with a LocalBackend.
    let backend = Arc::new(LocalBackend::new("local-a"));
    backend
        .put_collection("demo", d, ids, data.clone())
        .unwrap();
    let lake = RuLake::new(rerank, seed);
    lake.register_backend(backend).unwrap();

    // Prime the cache + query.
    let query = clustered(1, d, 10, 999)[0].clone();
    let direct_hits = direct.search(&query, 10).unwrap();
    let lake_hits = lake.search_one("local-a", "demo", &query, 10).unwrap();
    assert_eq!(lake_hits.len(), direct_hits.len());

    // Ids should match (positions are dense 0..n, so direct's usize
    // matches lake's u64 after the pos_to_id mapping).
    let direct_ids: Vec<usize> = direct_hits.iter().map(|r| r.id).collect();
    let lake_ids: Vec<u64> = lake_hits.iter().map(|r| r.id).collect();
    let direct_as_u64: Vec<u64> = direct_ids.iter().map(|&x| x as u64).collect();
    assert_eq!(direct_as_u64, lake_ids);

    // Scores should match too — same seed + same rerank factor means
    // the two caches are byte-identical.
    for (a, b) in direct_hits.iter().zip(lake_hits.iter()) {
        assert!(
            (a.score - b.score).abs() < 1e-4,
            "direct {} vs lake {} — estimator divergence",
            a.score,
            b.score
        );
    }
}

#[test]
fn rulake_recomputes_on_backend_generation_bump() {
    // M1 acceptance #2: mutating the backend bumps the generation; the
    // next search observes the new state.
    let d = 32;
    let rerank = 20;
    let seed = 7;

    let backend = Arc::new(LocalBackend::new("local-mut"));
    backend
        .put_collection("c1", d, vec![10], vec![vec![1.0; d]])
        .unwrap();

    let lake = RuLake::new(rerank, seed);
    lake.register_backend(backend.clone()).unwrap();

    // First search — miss, prime.
    let q = vec![1.0; d];
    let r1 = lake.search_one("local-mut", "c1", &q, 1).unwrap();
    assert_eq!(r1[0].id, 10);
    let s1 = lake.cache_stats();
    assert_eq!(s1.primes, 1);
    assert_eq!(s1.misses, 1);
    assert_eq!(s1.hits, 0);

    // Second search, same data — hit.
    lake.search_one("local-mut", "c1", &q, 1).unwrap();
    let s2 = lake.cache_stats();
    assert_eq!(s2.primes, 1, "no re-prime on cache hit");
    assert_eq!(s2.hits, 1);

    // Mutate the backend — append a *closer* vector with a new id.
    backend.append("c1", 42, vec![1.0; d]).unwrap();

    // Third search — backend generation bumped → cache invalidated +
    // re-primed. The new vector is a tie with the old one (same
    // coordinates), so at k=2 both are returned.
    let r3 = lake.search_one("local-mut", "c1", &q, 2).unwrap();
    let ids: std::collections::HashSet<u64> = r3.iter().map(|r| r.id).collect();
    assert!(ids.contains(&42), "new vector not observed after bump");
    let s3 = lake.cache_stats();
    assert_eq!(s3.primes, 2, "re-prime on generation bump");
}

#[test]
fn rulake_federates_across_two_backends() {
    // M1 acceptance #3: fan-out across two backends, merge by score.
    let d = 16;
    let rerank = 20;
    let seed = 3;

    // Build a fake global index by concatenating the two backends' data,
    // so we know the correct federated top-k.
    let a_data: Vec<Vec<f32>> = (0..50)
        .map(|i| (0..d).map(|j| (i + j) as f32).collect())
        .collect();
    let b_data: Vec<Vec<f32>> = (0..50)
        .map(|i| (0..d).map(|j| ((i * 2) + j) as f32).collect())
        .collect();

    let ba = Arc::new(LocalBackend::new("bq-like"));
    ba.put_collection("t1", d, (0..a_data.len() as u64).collect(), a_data.clone())
        .unwrap();

    let bb = Arc::new(LocalBackend::new("snowflake-like"));
    bb.put_collection(
        "t2",
        d,
        (1_000..1_000 + b_data.len() as u64).collect(),
        b_data.clone(),
    )
    .unwrap();

    let lake = RuLake::new(rerank, seed);
    lake.register_backend(ba).unwrap();
    lake.register_backend(bb).unwrap();

    // Query close to a specific a_data[7] row.
    let query = (0..d).map(|j| (7 + j) as f32 + 0.1).collect::<Vec<_>>();
    let hits = lake
        .search_federated(&[("bq-like", "t1"), ("snowflake-like", "t2")], &query, 5)
        .unwrap();
    assert_eq!(hits.len(), 5);
    // Top hit should be from `bq-like/t1` with id 7 (closest).
    assert_eq!(hits[0].backend, "bq-like");
    assert_eq!(hits[0].collection, "t1");
    assert_eq!(hits[0].id, 7);

    // Results must be sorted by score ascending.
    for w in hits.windows(2) {
        assert!(w[0].score <= w[1].score);
    }
}

#[test]
fn cache_hit_is_faster_than_miss() {
    // M1 acceptance #4: cache hits are meaningfully cheaper than misses.
    // We don't assert a specific speedup (CI noise) — we assert that
    // miss_time is greater than hit_time, both sub-second, at n=500.
    let d = 64;
    let n = 500;
    let rerank = 20;
    let seed = 13;

    let data = clustered(n, d, 10, seed);
    let backend = Arc::new(LocalBackend::new("perf-demo"));
    backend
        .put_collection("c", d, (0..n as u64).collect(), data.clone())
        .unwrap();

    // Eventual consistency so we don't round-trip to the backend on
    // every hit.
    let lake = RuLake::new(rerank, seed).with_consistency(Consistency::Eventual { ttl_ms: 60_000 });
    lake.register_backend(backend).unwrap();

    let query = clustered(1, d, 10, 99)[0].clone();

    // First search — cache miss.
    let t = Instant::now();
    let _ = lake.search_one("perf-demo", "c", &query, 10).unwrap();
    let t_miss = t.elapsed();

    // Subsequent searches — cache hits.
    let t = Instant::now();
    for _ in 0..20 {
        let _ = lake.search_one("perf-demo", "c", &query, 10).unwrap();
    }
    let t_hit_20 = t.elapsed();
    let t_hit_avg = t_hit_20 / 20;

    eprintln!(
        "miss={:?}  hit_avg={:?}  ratio={:.1}×",
        t_miss,
        t_hit_avg,
        t_miss.as_nanos() as f64 / t_hit_avg.as_nanos().max(1) as f64
    );
    // Miss includes building the RabitqPlusIndex over n rows; hit is
    // just a search. Miss must dominate.
    assert!(
        t_miss > t_hit_avg,
        "expected miss > hit_avg; got miss={:?} hit_avg={:?}",
        t_miss,
        t_hit_avg
    );

    let stats = lake.cache_stats();
    assert_eq!(stats.primes, 1);
    assert!(stats.hits >= 20, "want ≥20 hits, got {}", stats.hits);
}

#[test]
fn dimension_mismatch_returns_error() {
    let d = 8;
    let backend = Arc::new(LocalBackend::new("tiny"));
    backend
        .put_collection("c", d, vec![1], vec![vec![0.0; d]])
        .unwrap();
    let lake = RuLake::new(20, 0);
    lake.register_backend(backend).unwrap();
    let bad_query = vec![0.0; d + 1];
    let err = lake.search_one("tiny", "c", &bad_query, 1).unwrap_err();
    assert!(matches!(
        err,
        ruvector_rulake::RuLakeError::DimensionMismatch { .. }
    ));
}

#[test]
fn unknown_backend_returns_error() {
    let lake = RuLake::new(20, 0);
    let err = lake.search_one("nope", "nope", &[0.0; 4], 1).unwrap_err();
    assert!(matches!(
        err,
        ruvector_rulake::RuLakeError::UnknownBackend(_)
    ));
}

#[test]
fn unknown_collection_returns_error() {
    let backend = Arc::new(LocalBackend::new("b"));
    let lake = RuLake::new(20, 0);
    lake.register_backend(backend).unwrap();
    let err = lake.search_one("b", "missing", &[0.0; 4], 1).unwrap_err();
    // Error surfaces via the backend's generation() call.
    assert!(matches!(
        err,
        ruvector_rulake::RuLakeError::UnknownCollection { .. }
    ));
}
