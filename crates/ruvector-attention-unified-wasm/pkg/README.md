# @ruvector/attention-unified-wasm

Unified WebAssembly bindings for 18+ attention mechanisms: Neural, DAG, Graph, and Mamba SSM.

## Features

- **18+ Attention Mechanisms**: Comprehensive collection of attention variants
- **Neural Attention**: Standard transformer-style attention
- **DAG Attention**: Directed Acyclic Graph structured attention
- **Graph Attention**: For graph neural networks (GAT, GATv2)
- **Mamba SSM**: State Space Model attention alternative

## Installation

```bash
npm install @ruvector/attention-unified-wasm
```

## Usage

```javascript
import init, {
  MultiHeadAttention,
  DagAttention,
  GraphAttention,
  MambaSSM
} from '@ruvector/attention-unified-wasm';

await init();

// Standard Multi-Head Attention
const mha = new MultiHeadAttention(dim, heads);
const output = mha.forward(query, key, value);

// DAG Attention
const dag = new DagAttention(config);
dag.processGraph(nodes, edges);

// Graph Attention (GAT)
const gat = new GraphAttention(inFeatures, outFeatures, heads);
gat.forward(nodeFeatures, adjacency);

// Mamba SSM
const mamba = new MambaSSM(dim, stateSize);
mamba.forward(sequence);
```

## Supported Mechanisms

### Neural Attention
- Scaled Dot-Product Attention
- Multi-Head Attention
- Linear Attention
- Sparse Attention

### DAG Attention
- Topological Attention
- Hierarchical DAG Attention
- Causal DAG Attention

### Graph Attention
- GAT (Graph Attention Network)
- GATv2
- Graph Transformer

### State Space Models
- Mamba (S4-inspired)
- H3 Attention
- Hyena

## License

MIT OR Apache-2.0

## Links

- [GitHub Repository](https://github.com/ruvnet/ruvector)
- [Documentation](https://ruv.io)
