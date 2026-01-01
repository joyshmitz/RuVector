# @ruvector/economy-wasm

CRDT-based autonomous credit economy for distributed compute networks - WASM optimized.

## Features

- **CRDT-Based**: Conflict-free Replicated Data Types for eventual consistency
- **Autonomous Credits**: Self-managing credit system for compute resources
- **P2P Reputation**: Decentralized reputation tracking
- **Byzantine Fault Tolerant**: Handles malicious actors gracefully

## Installation

```bash
npm install @ruvector/economy-wasm
```

## Usage

```javascript
import init, { CreditEconomy, ReputationSystem } from '@ruvector/economy-wasm';

await init();

// Create an economy instance
const economy = new CreditEconomy();

// Create agents and allocate credits
economy.createAgent("agent-1", 1000);

// Transfer credits between agents
economy.transfer("agent-1", "agent-2", 100);

// Track reputation
const reputation = new ReputationSystem();
reputation.recordSuccess("agent-1", "task-123");
```

## Architecture

- **GCounter/PNCounter**: For credit tracking
- **ORSet**: For agent membership
- **LWWRegister**: For reputation scores

## License

MIT

## Links

- [GitHub Repository](https://github.com/ruvnet/ruvector)
- [Documentation](https://ruv.io)
