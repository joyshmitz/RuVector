# ADR-155: ruLake — Vector-Native Federation Intermediary on RVF

## Status

**Accepted (M1)** — core abstraction + LocalBackend + FsBackend shipped
and measured on branch `research/rulake-datalake-analysis`. Backend
adapters (Parquet, BigQuery, Iceberg, Delta) are M2+.

## Date

2026-04-23 (v2 — intermediary reframe)
2026-04-22 (v3 — M1 measured + cache-first reframe)

## Authors

ruv.io · RuVector research. Spike output on branch
`research/rulake-datalake-analysis`.

## Relates To

- ADR-154 — RaBitQ rotation-based 1-bit quantization for ANNS
- ADR-057 — Federated RVF transfer learning (PII stripping, DP accounting)
- ADR-006 — Unified Memory Service (AgentDB context)
- Research: [`docs/research/ruLake/`](../research/ruLake/) (8 companion docs)
- Research: [`docs/research/rvf/spec/`](../research/rvf/spec/) (RVF Four Laws)

---

## Context

The RVF ecosystem (`crates/rvf/`, 22 crates) ships:

- Append-only segment model with a tail manifest (RVF Four Laws).
- Progressive HNSW indexing (Layer A/B/C).
- Temperature-tiered quantization (scalar, PQ, binary).
- **RaBitQ 1-bit rotation-based quantization** (`crates/ruvector-rabitq`,
  merged 2026-04-23 as `2c028aee3`). 100 % recall@10 at 957 QPS
  single-thread on n = 100 k, D = 128, rerank×20 — 3.13× over exact flat
  (see `crates/ruvector-rabitq/BENCHMARK.md`).
- Witness chains (SHAKE-256, Ed25519, ML-DSA-65 PQ).
- Federation primitives: PII stripping, differential privacy, RDP
  accounting, FedAvg (`rvf-federation`).
- An HTTP/SSE streaming server (`rvf-server`) that already speaks RVF
  wire over a non-JSON protocol.

Enterprise customers don't buy formats. They buy *reachability into
their existing datalake* — BigQuery, Snowflake, Databricks, Iceberg on
S3, Delta on ADLS. The first cut of this ADR proposed one
deep per-lake integration (BigQuery Tier-1 via external tables + remote
functions) and treated the others as Tier-2 follow-ups. That shape works
but buys less leverage than the alternative proposed here.

## Decision

**Build ruLake as a cache-first vector execution fabric.** The cache —
RaBitQ-compressed hot codes with deterministic rotation + witness — *is*
the product; federation is the cache's refill mechanism. An
app or agent speaks the RVF wire to `rvf-server`; ruLake answers from
the cache whenever the backend's coherence token is current. On miss
or drift, the corresponding backend adapter (Parquet-on-GCS, BigQuery,
Snowflake, Iceberg, Delta, local) pulls fresh vectors, rebuilds the
cache entry, and returns the result. The cache-hit path matches direct
`RabitqPlusIndex::search` at ~957 QPS / 100 % recall@10
(`ruvector-rabitq::BENCHMARK.md`) and measures a 1.00× intermediary
tax on a local backend (`ruvector-rulake::BENCHMARK.md`).

**Why cache-first, not federation-first**, per the decision matrix in
[`06-positioning.md`](../research/ruLake/06-positioning.md) §"Decision matrix":

| Axis | Federation-first | Cache-first |
|------|-----------------:|------------:|
| Latency | 2 | **5** |
| Simplicity | 2 | 4 |
| Governance | 5 | 4 |
| Adoption friction | 2 | **4** |
| Differentiation | 3 | **5** |

Cache-first wins 4 of 5 axes. The one federation wins (governance)
still scores 4/5 under cache-first because the cache is the choke point.

Key shape decisions:

1. **Backend-adapter trait is the contract.** A new crate
   `rvf-lake` (workspace member under `crates/rvf/`) defines the
   `BackendAdapter` trait: `list_collections`, `pull_vectors`,
   `supports_pushdown`, `push_down_topk`. Adapters ship as sub-crates
   so customers with proprietary lakes can implement their own without
   forking ruLake.

2. **Cache is RaBitQ-native.** Pulled vectors are immediately compressed
   into 1-bit RaBitQ codes via the existing `ruvector-rabitq::RabitqPlusIndex`.
   Hot queries answer from the cache (957 QPS, 21× compression); cold
   queries pull from the backend and prime the cache. Cache coherence is
   via manifest-generation numbers carried by each backend (see §Consequences
   for the staleness trade-off).

3. **Governance is a single choke point** between the wire and the
   planner (L4 in the 4-layer diagram). RBAC, column masking, lineage,
   GDPR phase-1 delete, PII classification, and audit log all live here,
   so adding a backend does not multiply the governance work.

4. **BigQuery-native compute (remote functions, external tables) is a
   push-down optimization inside the `BigQueryBackend` adapter, not a
   new product shape.** When BQ Vector Search or a BQ remote function
   can do the work, the adapter pushes down; otherwise the adapter pulls
   vectors and the intermediary's cache answers. The wire-protocol API
   never changes.

5. **RVF is the lingua franca both upstream and downstream.** Apps speak
   RVF wire to `rvf-server`; adapters emit RVF segments into the cache
   (`rvf-runtime::segment::Segment`), reusing the same type system that
   the rest of the ecosystem already depends on.

6. **The portable unit is the bundle sidecar `table.rulake.json`,
   not the UDF.** Implemented in `ruvector_rulake::bundle::RuLakeBundle`:
   carries `(data_ref, dim, rotation_seed, rerank_factor, generation,
   rvf_witness, pii_policy, lineage_id)` with a SHAKE-256 witness over
   the preceding fields. Two instances of ruLake observing the same
   bundle from different backends cache-share safely because the
   witness is the cache-key anchor. This explicitly addresses the
   "cache invalidation drift" failure mode — the witness changes iff
   the underlying compressed codes would change. `Generation` is an
   opaque union (`Num(u64)` for mtimes + versions; `Opaque(String)`
   for UUIDs and hashes) so Iceberg/Snowflake tokens fit.

### Minimum viable scope (12 weeks, 20.5 engineer-weeks)

See [`docs/research/ruLake/07-implementation-plan.md`](../research/ruLake/07-implementation-plan.md)
for the full breakdown.

- **M1 (weeks 1–2) — shipped on branch, measured:** the
  `ruvector-rulake` crate scaffold with `BackendAdapter` trait,
  `LocalBackend` + `FsBackend`, RaBitQ-cache glue, witness-addressed
  cache with cross-backend sharing, LRU eviction over unpinned
  entries, rayon parallel fan-out across shards, and the bundle
  sidecar (`table.rulake.json`) with atomic FS persistence. 24 tests
  passing (9 bundle + 3 fs_backend + 12 federation incl. concurrent
  hammer). Acceptance numbers from `crates/ruvector-rulake/BENCHMARK.md`:

  - Intermediary tax on LocalBackend: 1.00× at n=100k (2,991 QPS
    cache-hit vs 3,050 QPS direct RaBitQ).
  - Cache-hit path in `RuLake::search_one` byte-exact vs direct
    `RabitqPlusIndex::search` at the same `(seed, rerank_factor)`.
  - Rayon parallel fan-out prime-time speedups: 2-shard 1.97×,
    4-shard 3.86× at n=100k.
  - Recall@10 > 90% on clustered D=128 rerank×20 (gate test
    `rulake_recall_at_10_above_90pct_vs_brute_force`).
  - Witness-addressed cache: two `LocalBackend`s with identical data
    share one compressed entry; `shared_hits` counter bumps on the
    install (test `two_backends_share_cache_when_witness_matches`).
  - Send+Sync under contention: 8 threads × 50 queries,
    mixed single-shard + federated, zero primes after warmup (test
    `concurrent_searches_are_safe_and_correct`).

  **Original M1 acceptance (preserved):** end-to-end query against
  an in-memory collection returns top-k identical to direct
  `RabitqPlusIndex::search`. Met.

- **M2 (weeks 3–5)**: `ParquetBackend` (read vectors from S3-Parquet or
  local Parquet via the `arrow` crate). Cache coherence via Parquet
  file-mtime or Iceberg snapshot id. **Acceptance:** ingest a 100 k-row
  Parquet file; query latency ≤ 2× the equivalent `RabitqPlusIndex`
  standalone on the cache-hit path.

- **M3 (weeks 6–8)**: `BigQueryBackend` — pull path via storage-read
  API + Tier-2 push-down via BQ Vector Search for backends that don't
  benefit from RaBitQ. **Acceptance:** end-to-end query against a
  10 M-row BQ table returns correct top-k and respects a row-level
  access-control policy.

- **M4 (weeks 9–10)**: Governance MVP — RBAC via OIDC/JWT claims,
  PII classification passthrough (reusing `rvf-federation::pii`), lineage
  events into OpenLineage format. **Acceptance:** a query against a
  masked column returns masked values; lineage trace is complete across
  the federation hop.

- **M5 (weeks 11–12)**: Second backend adapter (`DeltaBackend` or
  `IcebergBackend`, customer-driven). **Acceptance:** a query federated
  across BigQuery + Delta returns correctly-merged top-k under the same
  wire call.

### Non-goals

- **Not a vector database.** ruLake does not own storage. Customers who
  want a standalone managed vector DB stay on Pinecone / Weaviate /
  Milvus / LanceDB.
- **Not a replacement for BigQuery/Snowflake.** These are backends to
  ruLake, not competitors.
- **Not a storage engine.** We ride S3 / GCS / ADLS / local.
- **Not sub-millisecond.** Cache-hit path targets ≤ 2 ms p99; federated
  cold path is network-bound.
- **Not GDPR-compliant out of the box.** v1 supports phase-1 logical
  delete with 30-day phase-2 backend delete. Crypto-shredding (same-day)
  is v2.
- **Not a SQL dialect.** Queries are structured RVF wire — vector + filter
  predicates — not ANSI SQL. We do NOT try to reimplement Trino.

## Alternatives considered

### A. Plug-in-per-lake (the first cut of this spike)

Deep BigQuery integration as Tier-1 (external tables + remote function
with RaBitQ kernel), Snowflake as Tier-2, etc. Rejected because:

- Each backend is a multi-E-wk integration with its own governance,
  lineage, and auth story — work multiplies linearly with backends.
- Users who have vectors *spread across* BigQuery and S3 still can't
  federate; they get a per-lake experience.
- The RaBitQ compression story lives N times instead of once.

Preserved as an *option inside* the intermediary shape: the `BigQueryBackend`
adapter can push operators down into BQ when it helps (see ADR Decision §4
and [`03-bigquery-integration.md`](../research/ruLake/03-bigquery-integration.md)).

### B. Standalone vector-DB with Parquet import

Rejected: competes head-on with Pinecone/Weaviate/Milvus/LanceDB without
a clear 10× moat. The "where the data already lives" moat is strong
only if we meet the data where it is.

### C. Pure Iceberg table-format extension

Propose a vector-extension to the Iceberg spec and let every engine that
reads Iceberg pick it up "for free". Real but slow — Iceberg spec
evolution is measured in years. Kept as a v2 contribution upstream.

### D. Trino/Presto connector

Implement a Trino connector that answers vector queries. Rejected as
Tier-1 because Trino assumes SQL; a vector ANN query doesn't fit the
SQL shape cleanly without UDFs. Kept as v2.

### E. JVM intermediary in Java/Scala

Rejected: core RVF and RaBitQ are Rust. Shipping a JVM intermediary
duplicates the hot path in a slower runtime. A JVM *client* for ruLake
is v2.

### F. Run RVF purely inside a customer's notebook

The status quo for many customers — load Parquet into `RabitqPlusIndex`,
query from Python. Works for single-machine single-user; the intermediary
unlocks multi-user, multi-backend, and governance.

### G. Push-through-only (no cache)

Every query always goes to the backend. Simpler coherence, but throughput
is gated by the slowest backend and RaBitQ's 3× speedup is wasted. Kept
as a mode flag for customers who cannot tolerate cache staleness.

## Consequences

### Positive

- **One governance choke point** across all backends — a single
  RBAC/PII/lineage/audit story, not N.
- **RaBitQ compression pays off across the fleet.** Compress once in the
  cache, serve from any backend.
- **Additive backend support.** Shipping DeltaBackend or SnowflakeBackend
  adds to the reachable market without changing the wire protocol.
- **Clean contract for partners.** Proprietary-lake customers implement
  `BackendAdapter` themselves; ruLake stays maintainable.

### Negative

- **Cache coherence is a real problem.** Backend updates don't notify
  ruLake by default. Mitigations per backend:
  - Parquet: file-mtime + filename hash.
  - Iceberg: snapshot id on the table manifest.
  - BigQuery: `INFORMATION_SCHEMA.TABLE_STORAGE.last_modified_time`.
  - Delta: `_delta_log` transaction version.
  - Snowflake: `SYSTEM$CLUSTERING_INFORMATION` / change streams.
  Customers with strict consistency requirements run in
  push-through-only mode (alternative G) and accept the QPS hit.
- **Latency hop.** Even on cache-hit the RVF-wire round trip adds
  1–5 ms over direct library use. Customers who call RaBitQ in-process
  stay in-process.
- **Owns a new surface** — the planner, the router, the cache eviction
  policy, the per-backend coherence protocols. Real engineering weight.
- **BigQuery's own native Vector Search competes** for pure-BQ
  customers. ruLake's value is cross-backend + governance + RaBitQ
  determinism; for a customer with one lake and no governance needs,
  the native path may win.

### Neutral

- `rvf-federation` expands semantically: it already meant "aggregate
  across untrusted nodes"; now it also means "aggregate across
  heterogeneous backends". The crate name keeps.
- `rvf-server` grows a backend-registry endpoint and a cache-status
  endpoint. API is additive; existing callers are undisturbed.
- Wire-protocol additions (adapter identity, backend id, coherence
  token) ride the existing RVF segment type system — no breaking
  changes.

## Open questions

### Resolved in M1

- ~~**Cache sizing.**~~ **Resolved (partial):** LRU eviction over
  unpinned entries is implemented via `RuLake::with_max_cache_entries(n)`
  + `CacheEntry::last_used`. M2 measures the tail-working-set ratio
  where the tax becomes unattractive.
- ~~**Consistency SLA.**~~ **Resolved:** `Consistency::{Fresh, Eventual{ttl_ms}}`
  is a per-`RuLake` setting with `Fresh` as the default. Per-backend
  override deferred — no customer has asked and the surface is easy
  to add later.
- ~~**Per-collection vs per-backend cache.**~~ **Resolved:** One pool,
  witness-addressed. Per-collection quotas are an M4 governance
  feature if a customer needs "hot collection" guarantees.
- ~~**Crate placement.**~~ **Resolved:** `crates/ruvector-rulake/` — a
  top-level workspace member (not under `crates/rvf/`). Keeps rvf core
  `no_std`-friendly, matches the rabitq crate's location.

### Still open (M2+)

1. **Push-down negotiation.** When a backend supports native push-down
   (BQ Vector Search, Snowflake Cortex), at what point does the planner
   prefer it over the cache? Probably when cache hit-rate < 20% for the
   collection — needs a policy, not a constant.
2. **JVM client for the wire protocol.** Enterprise customers want a
   maintained Java client. Spec'd as v2 but the enterprise-pipeline
   customer will ask about it in week one.
3. **Trademark / naming.** "ruLake" vs alternatives before docs and
   crate names lock.
4. **Cost accounting.** When the planner pushes down to BQ, whose
   BQ credits are burned? Needs a customer-facing cost-attribution
   story, not just an engineering one.
5. **Remote-backend tax.** BENCHMARK.md's 1.00× tax on `LocalBackend`
   is the floor. M2 needs to measure the real tax on a Parquet-on-GCS
   prime and document the p50/p99 numbers the BQ UDF path can expect.
6. **Cache sidecar daemon protocol.** Bundle sidecar is shipped, FS
   persistence is atomic. The missing piece is the on-line protocol
   by which the warehouse-push path hands a new `table.rulake.json`
   to a running ruLake — file-watcher, signed notification, or a
   polling refresh on `Eventual` TTL?
