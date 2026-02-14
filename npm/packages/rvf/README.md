# @ruvector/rvf

Unified TypeScript/JavaScript SDK for the **RuVector Format (RVF)** — a cognitive container that stores vectors, carries models, boots compute kernels, and proves everything in a single `.rvf` file.

## Platform Support

| Platform | Runtime | Backend | Status |
|----------|---------|---------|--------|
| Linux x86_64 | Node.js 18+ | Native (N-API) | Stable |
| Linux aarch64 | Node.js 18+ | Native (N-API) | Stable |
| macOS x86_64 | Node.js 18+ | Native (N-API) | Stable |
| macOS arm64 (Apple Silicon) | Node.js 18+ | Native (N-API) | Stable |
| Windows x86_64 | Node.js 18+ | Native (N-API) | Stable |
| Any | Deno | WASM | Supported |
| Any | Browser (Chrome, Firefox, Safari) | WASM | Supported |
| Any | Cloudflare Workers / Edge | WASM | Supported |
| Any | Bun | Native (N-API) | Experimental |

**Deno**: The WASM build targets `wasm32-unknown-unknown`, which runs natively in Deno. Import via `npm:` specifier or load the `.wasm` bundle directly.

**Browser**: The `@ruvector/rvf-wasm` package provides a ~46 KB control-plane WASM module plus a ~5.5 KB tile-compute module. Works in any browser with WebAssembly support.

## Install

```bash
# Node.js (auto-detects native or WASM)
npm install @ruvector/rvf

# WASM only (browser, Deno, edge)
npm install @ruvector/rvf-wasm
```

## Quick Start

### Node.js

```typescript
import { RvfDatabase } from '@ruvector/rvf';

// Create a vector store
const db = RvfDatabase.create('vectors.rvf', { dimension: 384 });

// Insert vectors
db.ingestBatch(new Float32Array(384), [1]);

// Query nearest neighbors
const results = db.query(new Float32Array(384), 10);

// Lineage & inspection
console.log(db.fileId());       // unique file UUID
console.log(db.dimension());    // 384
console.log(db.segments());     // [{ type, id, size }]

// Derive child (COW branching)
const child = db.derive('child.rvf');

db.close();
```

### Browser (WASM)

```html
<script type="module">
import init, { RvfStore } from '@ruvector/rvf-wasm';

await init();

const store = RvfStore.create(384, 'cosine');
store.ingest(new Float32Array(384), 0);
const results = store.query(new Float32Array(384), 10);
console.log('Results:', results);
</script>
```

### Deno

```typescript
// Import via npm: specifier
import init, { RvfStore } from "npm:@ruvector/rvf-wasm";

await init();

const store = RvfStore.create(384, 'cosine');
store.ingest(new Float32Array(384), 0);
const results = store.query(new Float32Array(384), 10);
console.log('Results:', results);
```

## What is RVF?

RVF (RuVector Format) is a universal binary substrate that merges database, model, graph engine, kernel, and attestation into a single deployable file. A `.rvf` file is segmented — each segment carries a different payload type, and unknown segments are preserved by all tools.

### Segment Types

| ID | Segment | Description |
|----|---------|-------------|
| 0x00 | MANIFEST_SEG | Level0Root manifest with file metadata |
| 0x01 | VEC_SEG | Raw vector data (f32, f16, bf16, int8) |
| 0x02 | INDEX_SEG | HNSW graph for approximate nearest neighbor |
| 0x03 | META_SEG | Vector metadata (JSON, CBOR) |
| 0x04 | QUANT_SEG | Quantization codebooks |
| 0x05 | OVERLAY_SEG | LoRA/adapter weight overlays |
| 0x06 | GRAPH_SEG | Property graph adjacency data |
| 0x07 | TENSOR_SEG | Dense tensor data |
| 0x08 | WASM_SEG | Embedded WASM modules |
| 0x09 | MODEL_SEG | ML model weights |
| 0x0A | CRYPTO_SEG | Signatures and key material |
| 0x0B | WITNESS_SEG | Append-only witness/audit chain |
| 0x0C | CONFIG_SEG | Runtime configuration |
| 0x0D | CUSTOM_SEG | User-defined segment |
| 0x0E | KERNEL_SEG | Linux microkernel image |
| 0x0F | EBPF_SEG | eBPF programs |
| 0x20 | COW_MAP_SEG | Copy-on-write cluster map |
| 0x21 | REFCOUNT_SEG | Cluster reference counts |
| 0x22 | MEMBERSHIP_SEG | Branch membership filter |
| 0x23 | DELTA_SEG | Sparse delta patches (LoRA) |

## N-API Methods (Node.js)

19 methods on the `RvfDatabase` class:

| Method | Description |
|--------|-------------|
| `RvfDatabase.create(path, opts)` | Create new RVF file |
| `RvfDatabase.open(path)` | Open existing (read-write) |
| `RvfDatabase.openReadonly(path)` | Open existing (read-only) |
| `db.ingestBatch(vectors, ids)` | Insert vectors by batch |
| `db.query(vector, k)` | k-NN search |
| `db.delete(ids)` | Delete vectors by ID |
| `db.deleteByFilter(filter)` | Delete vectors matching filter |
| `db.compact()` | Compact and reclaim space |
| `db.status()` | File status (count, dimension, metric) |
| `db.close()` | Close file handle |
| `db.fileId()` | UUID of this file |
| `db.parentId()` | UUID of parent (if derived) |
| `db.lineageDepth()` | Derivation depth |
| `db.derive(path)` | COW-branch to new file |
| `db.embedKernel(bytes)` | Embed Linux kernel image |
| `db.extractKernel()` | Extract kernel image |
| `db.embedEbpf(bytes)` | Embed eBPF program |
| `db.extractEbpf()` | Extract eBPF program |
| `db.segments()` | List all segments |

## WASM Exports

29 exported functions for browser and edge runtimes:

**Control plane** (10): `rvf_create`, `rvf_open`, `rvf_close`, `rvf_ingest`, `rvf_query`, `rvf_delete`, `rvf_status`, `rvf_compact`, `rvf_derive`, `rvf_segments`

**Tile compute** (14): `tile_dot_f32`, `tile_cosine_f32`, `tile_l2_f32`, `tile_dot_f16`, `tile_cosine_f16`, `tile_l2_f16`, `tile_topk`, `tile_quantize_sq8`, `tile_dequantize_sq8`, `tile_scan_filtered`, `tile_merge_topk`, `tile_batch_distance`, `tile_prefetch`, `tile_accumulate`

**Segment parsing** (3): `parse_segment_header`, `parse_vec_header`, `parse_manifest`

**Memory** (2): `rvf_alloc`, `rvf_free`

## CLI (Rust)

18 subcommands available through the `rvf` binary:

```bash
# Core operations
rvf create vectors.rvf --dimension 384 --metric cosine
rvf ingest vectors.rvf --input data.json
rvf query vectors.rvf --vector "[0.1,0.2,...]" --k 10
rvf delete vectors.rvf --ids "[1,2,3]"
rvf status vectors.rvf
rvf inspect vectors.rvf
rvf compact vectors.rvf

# Branching & lineage
rvf derive vectors.rvf --output child.rvf
rvf filter vectors.rvf --include "[1,2,3]"
rvf freeze vectors.rvf
rvf rebuild-refcounts vectors.rvf

# Compute containers
rvf serve vectors.rvf --port 8080
rvf launch vectors.rvf
rvf embed-kernel vectors.rvf --image bzImage
rvf embed-ebpf vectors.rvf --program filter.o

# Verification
rvf verify-witness vectors.rvf
rvf verify-attestation vectors.rvf

# Export
rvf export vectors.rvf --output dump.json
```

Build the CLI:

```bash
cargo install --path crates/rvf/rvf-cli
```

## Example .rvf Files

45 pre-built example files are available for download (~11 MB total). These demonstrate every segment type and use case.

### Download

```bash
# Download a specific example
curl -LO https://raw.githubusercontent.com/ruvnet/ruvector/main/examples/rvf/output/basic_store.rvf

# Clone just the examples
git clone --depth 1 --filter=blob:none --sparse https://github.com/ruvnet/ruvector.git
cd ruvector && git sparse-checkout set examples/rvf/output
```

### Example Catalog

| File | Size | Description |
|------|------|-------------|
| `basic_store.rvf` | 152 KB | 1,000 vectors, dim 128, cosine metric |
| `semantic_search.rvf` | 755 KB | Semantic search with HNSW index |
| `rag_pipeline.rvf` | 303 KB | RAG pipeline with embeddings |
| `embedding_cache.rvf` | 755 KB | Cached embedding store |
| `quantization.rvf` | 1.5 MB | PQ-compressed vectors |
| `progressive_index.rvf` | 2.5 MB | Large-scale progressive HNSW index |
| `filtered_search.rvf` | 255 KB | Metadata-filtered vector search |
| `recommendation.rvf` | 102 KB | Recommendation engine vectors |
| `agent_memory.rvf` | 32 KB | AI agent episodic memory |
| `swarm_knowledge.rvf` | 86 KB | Multi-agent shared knowledge base |
| `experience_replay.rvf` | 27 KB | RL experience replay buffer |
| `tool_cache.rvf` | 26 KB | MCP tool call cache |
| `mcp_in_rvf.rvf` | 32 KB | MCP server embedded in RVF |
| `ruvbot.rvf` | 51 KB | Chatbot knowledge store |
| `claude_code_appliance.rvf` | 17 KB | Claude Code cognitive appliance |
| `lineage_parent.rvf` | 52 KB | COW parent file |
| `lineage_child.rvf` | 26 KB | COW child (derived) file |
| `reasoning_parent.rvf` | 5.6 KB | Reasoning chain parent |
| `reasoning_child.rvf` | 8.1 KB | Reasoning chain child |
| `reasoning_grandchild.rvf` | 162 B | Minimal derived file |
| `self_booting.rvf` | 31 KB | Self-booting with KERNEL_SEG |
| `linux_microkernel.rvf` | 15 KB | Embedded Linux microkernel |
| `ebpf_accelerator.rvf` | 153 KB | eBPF distance accelerator |
| `browser_wasm.rvf` | 14 KB | Browser WASM module embedded |
| `tee_attestation.rvf` | 102 KB | TEE attestation with witnesses |
| `zero_knowledge.rvf` | 52 KB | ZK-proof witness chain |
| `crypto_signed.rvf` | (see `sealed_engine.rvf`) | Signed + sealed |
| `sealed_engine.rvf` | 208 KB | Sealed inference engine |
| `access_control.rvf` | 77 KB | Permission-gated vectors |
| `financial_signals.rvf` | 202 KB | Financial signal vectors |
| `medical_imaging.rvf` | 302 KB | Medical imaging embeddings |
| `legal_discovery.rvf` | 903 KB | Legal document discovery |
| `multimodal_fusion.rvf` | 804 KB | Multi-modal embedding fusion |
| `hyperbolic_taxonomy.rvf` | 23 KB | Hyperbolic space taxonomy |
| `network_telemetry.rvf` | 16 KB | Network telemetry vectors |
| `postgres_bridge.rvf` | 152 KB | PostgreSQL bridge vectors |
| `ruvllm_inference.rvf` | 133 KB | RuvLLM inference cache |
| `serverless.rvf` | 509 KB | Serverless deployment bundle |
| `edge_iot.rvf` | 27 KB | Edge/IoT lightweight store |
| `dedup_detector.rvf` | 153 KB | Deduplication detector |
| `compacted.rvf` | 77 KB | Post-compaction example |
| `posix_fileops.rvf` | 52 KB | POSIX file operations test |
| `network_sync_a.rvf` | 52 KB | Network sync peer A |
| `network_sync_b.rvf` | 52 KB | Network sync peer B |
| `agent_handoff_a.rvf` | 31 KB | Agent handoff source |
| `agent_handoff_b.rvf` | 11 KB | Agent handoff target |

### Generate Examples Locally

```bash
cd crates/rvf
cargo run --example generate_all
ls output/  # 45 .rvf files
```

## Integration

### With `ruvector` (npx ruvector)

The `ruvector` npm package includes 8 RVF CLI commands:

```bash
npm install ruvector @ruvector/rvf

# Enable RVF backend
export RUVECTOR_BACKEND=rvf

# Or use --backend flag
npx ruvector --backend rvf create mydb.rvf -d 384

# RVF-specific commands
npx ruvector rvf create mydb.rvf -d 384
npx ruvector rvf ingest mydb.rvf --input data.json
npx ruvector rvf query mydb.rvf --vector "[0.1,...]" --k 10
npx ruvector rvf status mydb.rvf
npx ruvector rvf segments mydb.rvf
npx ruvector rvf derive mydb.rvf --output child.rvf
npx ruvector rvf compact mydb.rvf
npx ruvector rvf export mydb.rvf --output dump.json
```

### With `rvlite`

```bash
npm install rvlite @ruvector/rvf-wasm
```

When `@ruvector/rvf-wasm` is installed, rvlite can use RVF as a persistent storage backend:

```typescript
import { createRvLite } from 'rvlite';

// rvlite auto-detects @ruvector/rvf-wasm for persistence
const db = await createRvLite({ dimensions: 384 });
await db.insert([0.1, 0.2, ...], { text: "Hello world" });
const results = await db.search([0.1, 0.2, ...], 5);
```

## Packages

| Package | Description | Runtime |
|---------|-------------|---------|
| `@ruvector/rvf` | Unified SDK (this package) | Node.js |
| `@ruvector/rvf-node` | Native N-API bindings | Node.js |
| `@ruvector/rvf-wasm` | WASM build (~46 KB + ~5.5 KB tile) | Browser, Deno, Edge |
| `@ruvector/rvf-mcp-server` | MCP server for AI agents | Node.js |

## Crate Structure (Rust)

| Crate | Description |
|-------|-------------|
| `rvf-types` | Wire types, segment headers, `no_std` compatible |
| `rvf-wire` | Serialization/deserialization |
| `rvf-manifest` | Level0Root manifest parsing |
| `rvf-index` | HNSW index operations |
| `rvf-quant` | Quantization codebooks |
| `rvf-crypto` | Signing, verification, key management |
| `rvf-runtime` | Full runtime (store, ingest, query, derive) |
| `rvf-kernel` | Linux microkernel builder |
| `rvf-launch` | QEMU launcher for self-booting files |
| `rvf-ebpf` | eBPF compiler and loader |
| `rvf-server` | HTTP API server (axum) |
| `rvf-cli` | CLI binary |
| `rvf-import` | Import from external formats |

## License

MIT
