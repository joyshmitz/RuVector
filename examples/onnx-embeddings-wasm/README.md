# RuVector ONNX Embeddings - WASM Edition

> **Portable embedding generation that runs anywhere WebAssembly runs**

This is a WASM-compatible companion to `ruvector-onnx-embeddings`. It provides the same embedding capabilities but uses [Tract](https://github.com/sonos/tract) for inference, enabling deployment to browsers, edge workers, and any WASM runtime.

## Features

| Feature | Description |
|---------|-------------|
| **Browser Support** | Generate embeddings directly in web browsers |
| **Edge Computing** | Deploy to Cloudflare Workers, Vercel Edge, Deno |
| **Portable** | Single WASM binary, no platform dependencies |
| **Same API** | Compatible interface with native crate |
| **Small Size** | ~5-10MB WASM bundle (compressed) |

## Installation

### Rust (as library)

```toml
[dependencies]
ruvector-onnx-embeddings-wasm = "0.1"
```

### JavaScript/TypeScript

```bash
npm install ruvector-onnx-embeddings-wasm
```

### Build from source

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for web
wasm-pack build --target web

# Build for Node.js
wasm-pack build --target nodejs

# Build for bundlers (webpack, etc.)
wasm-pack build --target bundler
```

## Usage

### JavaScript (Browser)

```html
<script type="module">
import init, { WasmEmbedder, WasmEmbedderConfig } from './pkg/ruvector_onnx_embeddings_wasm.js';

async function main() {
    // Initialize WASM
    await init();

    // Load model and tokenizer
    const modelBytes = await fetch('/models/all-MiniLM-L6-v2.onnx')
        .then(r => r.arrayBuffer())
        .then(b => new Uint8Array(b));

    const tokenizerJson = await fetch('/models/tokenizer.json')
        .then(r => r.text());

    // Create embedder
    const embedder = new WasmEmbedder(modelBytes, tokenizerJson);

    // Generate embedding
    const embedding = embedder.embedOne("Hello, world!");
    console.log("Dimension:", embedding.length); // 384

    // Compute similarity
    const sim = embedder.similarity(
        "I love programming",
        "Coding is my passion"
    );
    console.log("Similarity:", sim); // ~0.85
}

main();
</script>
```

### JavaScript (Node.js)

```javascript
const { WasmEmbedder } = require('ruvector-onnx-embeddings-wasm');
const fs = require('fs');

// Load model and tokenizer
const modelBytes = fs.readFileSync('./model.onnx');
const tokenizerJson = fs.readFileSync('./tokenizer.json', 'utf8');

// Create embedder
const embedder = new WasmEmbedder(modelBytes, tokenizerJson);

// Generate embeddings
const embedding = embedder.embedOne("Hello from Node.js!");
console.log("Embedding dimension:", embedding.length);
```

### Cloudflare Workers

```javascript
import { WasmEmbedder } from 'ruvector-onnx-embeddings-wasm';

export default {
    async fetch(request, env) {
        // Load model from R2 or KV
        const modelBytes = await env.MODELS.get('model.onnx', 'arrayBuffer');
        const tokenizerJson = await env.MODELS.get('tokenizer.json', 'text');

        const embedder = new WasmEmbedder(
            new Uint8Array(modelBytes),
            tokenizerJson
        );

        const { text } = await request.json();
        const embedding = embedder.embedOne(text);

        return Response.json({ embedding: Array.from(embedding) });
    }
};
```

### Rust (WASM target)

```rust
use ruvector_onnx_embeddings_wasm::{WasmEmbedder, WasmEmbedderConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_bytes = include_bytes!("../model.onnx");
    let tokenizer_json = include_str!("../tokenizer.json");

    let embedder = WasmEmbedder::new(model_bytes, tokenizer_json)?;

    let embedding = embedder.embed_one("Hello from Rust WASM!")?;
    println!("Dimension: {}", embedding.len());

    Ok(())
}
```

## Configuration

```javascript
import { WasmEmbedder, WasmEmbedderConfig } from 'ruvector-onnx-embeddings-wasm';

// Create custom config
const config = new WasmEmbedderConfig()
    .setMaxLength(512)      // Max tokens
    .setNormalize(true)     // L2 normalize
    .setPooling(0);         // 0=Mean, 1=Cls, 2=Max

const embedder = WasmEmbedder.withConfig(modelBytes, tokenizerJson, config);
```

### Pooling Strategies

| Value | Strategy | Description |
|-------|----------|-------------|
| 0 | Mean | Average all tokens (default) |
| 1 | Cls | Use [CLS] token only |
| 2 | Max | Max pooling across tokens |
| 3 | MeanSqrtLen | Mean normalized by sqrt(length) |
| 4 | LastToken | Use last token (decoder models) |

## Supported Models

Any ONNX model with standard transformer inputs works:
- `input_ids`: Token IDs `[batch, seq_len]`
- `attention_mask`: Attention mask `[batch, seq_len]`
- `token_type_ids`: Token types `[batch, seq_len]`

### Recommended Models

| Model | Dimension | Size | Notes |
|-------|-----------|------|-------|
| all-MiniLM-L6-v2 | 384 | 23MB | Fast, good quality |
| all-MiniLM-L12-v2 | 384 | 33MB | Better quality |
| bge-small-en-v1.5 | 384 | 33MB | State-of-the-art small |

### Converting Models

```bash
# Install optimum
pip install optimum[onnxruntime]

# Export to ONNX
optimum-cli export onnx \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --task feature-extraction \
    ./model_output
```

## Performance

| Environment | Throughput | Latency (single) |
|-------------|------------|------------------|
| Chrome (M1 Mac) | ~50 texts/sec | ~20ms |
| Firefox (M1 Mac) | ~45 texts/sec | ~22ms |
| Node.js | ~80 texts/sec | ~12ms |
| Cloudflare Workers | ~30 texts/sec | ~33ms |
| Deno | ~75 texts/sec | ~13ms |

*Tested with all-MiniLM-L6-v2, 128 token inputs*

## Comparison with Native Crate

| Aspect | Native (`ort`) | WASM (`tract`) |
|--------|----------------|----------------|
| Speed | ⚡⚡⚡ | ⚡⚡ |
| Browser | ❌ | ✅ |
| Edge Workers | ❌ | ✅ |
| GPU | CUDA, TensorRT | ❌ |
| Bundle Size | ~50MB | ~5-10MB |
| Portability | Platform-specific | Universal |

**Use native** for: servers, high throughput, GPU acceleration
**Use WASM** for: browsers, edge computing, portability

## API Reference

### WasmEmbedder

```typescript
class WasmEmbedder {
    constructor(modelBytes: Uint8Array, tokenizerJson: string);
    static withConfig(modelBytes: Uint8Array, tokenizerJson: string, config: WasmEmbedderConfig): WasmEmbedder;

    embedOne(text: string): Float32Array;
    embedBatch(texts: string[]): Float32Array;
    similarity(text1: string, text2: string): number;

    dimension(): number;
    maxLength(): number;
}
```

### Utility Functions

```typescript
function cosineSimilarity(a: Float32Array, b: Float32Array): number;
function normalizeL2(embedding: Float32Array): Float32Array;
function version(): string;
function simdAvailable(): boolean;
```

## License

MIT License - See [LICENSE](../../LICENSE) for details.

---

**Part of the RuVector ecosystem** - High-performance vector operations in Rust
