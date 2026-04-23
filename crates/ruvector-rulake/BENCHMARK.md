# ruvector-rulake — Benchmarks

All numbers produced by a **single reproducible run** of

```bash
cargo run --release -p ruvector-rulake --bin rulake-demo
```

on a commodity Ryzen-class laptop, release build, single thread. Seeds
deterministic; reruns bit-identical.

## Headline (LocalBackend, same dataset as `ruvector-rabitq`)

Clustered Gaussian, D = 128, 100 clusters, rerank×20, 300 queries per
row (warm-cache; prime time reported separately).

### Intermediary tax is ~0× on a local backend

| n       | direct RaBitQ+ (QPS) | ruLake Fresh (QPS) | ruLake Eventual (QPS) | tax (Fresh/Eventual) |
|--------:|---------------------:|-------------------:|----------------------:|---------------------:|
|   5 000 |              17,311  |            17,874  |             17,858    | 0.97× / 0.97×        |
|  50 000 |               5,162  |             5,123  |              5,050    | 1.01× / 1.02×        |
| 100 000 |               3,122  |             3,117  |              3,114    | 1.00× / 1.00×        |

Interpretation:
- **Cache-hit path in `RuLake::search_one` costs effectively nothing** vs
  calling `RabitqPlusIndex::search` directly. The pos→id lookup + the
  HashMap get are in the noise.
- `Fresh` mode calls `LocalBackend::generation()` on every search (one
  hash-map read here). On a real backend this becomes a network RPC —
  **expect materially higher tax on BigQuery / Snowflake / S3-Parquet**.
  `Eventual { ttl_ms }` amortises it.
- Measured "prime" time is ≈ the `RabitqPlusIndex` build time on the
  pulled batch (210 ms / 50 k rows, 420 ms / 100 k rows, scales linearly).

### Federation — sequential fan-out (v1)

| n       | single-shard QPS | 2 shards QPS | 4 shards QPS | 4-shard efficiency |
|--------:|-----------------:|-------------:|-------------:|-------------------:|
|   5 000 |          17,874  |      10,953  |       6,933  |              0.39× |
|  50 000 |           5,123  |       3,808  |       2,671  |              0.52× |
| 100 000 |           3,117  |       2,470  |       1,781  |              0.57× |
                     
Federation-mode splits N vectors across K backends; each backend holds
N/K rows. v1 runs searches **sequentially** across backends; hence the
QPS drops sub-linearly in K. v2 adds parallel fan-out via `rayon` — see
ADR-155 §Consequences. Efficiency is QPS(fed) / (QPS(single) / K); 0.57
at 4 shards @ n=100k means fan-out merge overhead is real but not
catastrophic.

## Acceptance checks (M1)

The 7 smoke tests under `tests/federation_smoke.rs` gate M1 from
`docs/research/ruLake/07-implementation-plan.md`:

| # | Test | What it proves |
|---|---|---|
| 1 | `rulake_matches_direct_rabitq_on_local_backend` | Federation path is byte-exact vs direct RaBitQ at the same seed + rerank factor |
| 2 | `rulake_recomputes_on_backend_generation_bump` | Cache coherence protocol works — backend mutation is observed on next search |
| 3 | `rulake_federates_across_two_backends` | Multi-backend fan-out + score merge produces the globally-correct top-k |
| 4 | `cache_hit_is_faster_than_miss` | Cache prime-then-serve path beats uncached (measurement-level sanity) |
| 5 | `dimension_mismatch_returns_error` | Error type surfaces on bad inputs |
| 6 | `unknown_backend_returns_error` | Error type surfaces on misconfiguration |
| 7 | `unknown_collection_returns_error` | Error type surfaces on wrong collection name |

```
cargo test -p ruvector-rulake --release
  → 7 passed / 0 failed
```

## What's NOT benchmarked (v1 scope)

- **Real-backend network latency.** `LocalBackend::pull_vectors` is an in-process
  HashMap read; the Fresh-mode tax reported above is the floor, not the ceiling.
  Real backends (Parquet on S3, BigQuery via Storage Read API) add 10-100 ms
  per prime. Measured numbers land in M2.
- **Recall regressions vs direct RaBitQ.** The test suite confirms byte-exact
  ordering + scores at the same seed. Formal recall sweeps across n / D /
  rerank_factor reuse `ruvector-rabitq::BENCHMARK.md` — ruLake doesn't change
  recall, only the distribution layer.
- **Push-down paths.** ADR-155 §Decision 4 defers backend-native vector ops
  to Tier-2 per-adapter. Not measured in v1.
- **Concurrent multi-client throughput.** Bench is single-thread. `RuLake` is
  `Send + Sync`; multi-threaded scaling is an M3 measurement.
- **Cache memory footprint vs backend size.** The cache currently primes the
  entire collection; LRU eviction is M3.

## Reproduce

```bash
cargo test  -p ruvector-rulake --release                   # 7 passed
cargo run   -p ruvector-rulake --release --bin rulake-demo # ~30 s on n=100k
cargo run   -p ruvector-rulake --release --bin rulake-demo -- --fast  # ~5 s
```

Dataset generator + seeds in `src/bin/rulake-demo.rs::clustered`.
