# RuVector Nervous System

Biological neural network models for vector databases, implementing neuroscience-inspired learning algorithms.

## Features

### BTSP: Behavioral Timescale Synaptic Plasticity

One-shot learning mechanism based on Bittner et al. 2017 hippocampal research:

- **One-shot learning**: Learn associations in seconds, not iterations
- **Bidirectional plasticity**: Weak synapses potentiate, strong synapses depress
- **Eligibility traces**: 1-3 second time windows for credit assignment
- **Plateau potentials**: Dendritic events gate plasticity

### Performance

- Single synapse update: <100ns
- Layer update (10K synapses): <100μs
- One-shot learning: Immediate, no iteration needed

## Usage

```rust
use ruvector_nervous_system::plasticity::btsp::{BTSPLayer, BTSPAssociativeMemory};

// Create a layer with 100 inputs
let mut layer = BTSPLayer::new(100, 2000.0); // 2 second time constant

// One-shot association: pattern -> target
let pattern = vec![0.1; 100];
layer.one_shot_associate(&pattern, 1.0);

// Immediate recall
let output = layer.forward(&pattern);
assert!((output - 1.0).abs() < 0.1);

// Associative memory for key-value storage
let mut memory = BTSPAssociativeMemory::new(128, 64);
let key = vec![0.5; 128];
let value = vec![0.1; 64];

memory.store_one_shot(&key, &value).unwrap();
let retrieved = memory.retrieve(&key).unwrap();
```

## Architecture

```
ruvector-nervous-system/
├── src/
│   ├── lib.rs              # Main library exports
│   ├── plasticity/
│   │   ├── mod.rs
│   │   └── btsp.rs         # BTSP implementation
│   └── routing/
│       └── mod.rs          # Neural routing (future)
├── benches/
│   └── btsp_bench.rs       # Performance benchmarks
└── tests/
    └── btsp_integration.rs # Integration tests
```

## Biological Basis

BTSP is based on:

1. **Dendritic plateau potentials**: Ca²⁺ spikes in dendrites
2. **Eligibility traces**: Short-term memory of recent activity
3. **Bidirectional plasticity**: Homeostatic weight regulation
4. **Hippocampal place fields**: One-shot spatial learning

## Applications for Vector Databases

- **Immediate indexing**: Add vectors without retraining
- **Adaptive routing**: Learn query patterns on-the-fly
- **Error correction**: Self-healing index structures
- **Context learning**: Remember user preferences instantly

## References

Bittner, K. C., Milstein, A. D., Grienberger, C., Romani, S., & Magee, J. C. (2017).
"Behavioral time scale synaptic plasticity underlies CA1 place fields."
*Science*, 357(6355), 1033-1036.

## License

MIT
