# @ruvector/nervous-system-wasm

WASM bindings for bio-inspired AI components - Hyperdimensional Computing (HDC), Behavioral Time-Scale Plasticity (BTSP), and neuromorphic computing.

## Features

- **Hyperdimensional Computing (HDC)**: High-dimensional vector operations for cognitive computing
- **Behavioral Time-Scale Plasticity (BTSP)**: Biologically-inspired learning mechanisms
- **Neuromorphic Computing**: Brain-inspired computing primitives
- **Spiking Neural Networks**: Event-driven neural computation

## Installation

```bash
npm install @ruvector/nervous-system-wasm
```

## Usage

```javascript
import init, {
  HyperdimensionalMemory,
  BTSPNeuron,
  SpikingNetwork
} from '@ruvector/nervous-system-wasm';

await init();

// Create HDC memory
const hdc = new HyperdimensionalMemory(dimension);
hdc.encode("concept", data);
const similar = hdc.query(vector);

// Create BTSP neuron
const neuron = new BTSPNeuron(config);
neuron.process(input);

// Create spiking network
const snn = new SpikingNetwork(topology);
snn.simulate(spikes, duration);
```

## Components

### Hyperdimensional Computing
- Binary/bipolar hypervectors
- Bundling and binding operations
- Similarity search

### BTSP Learning
- Plateau potentials
- Dendritic computation
- Temporal credit assignment

### Neuromorphic Primitives
- Leaky integrate-and-fire neurons
- STDP learning
- Spike encoding/decoding

## License

MIT

## Links

- [GitHub Repository](https://github.com/ruvnet/ruvector)
- [Documentation](https://ruv.io)
