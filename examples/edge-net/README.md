# @ruvector/edge-net

**Collective AI Computing Network - Share, Contribute, Compute Together**

A distributed computing platform that enables collective resource sharing for AI workloads. Contributors share idle compute resources, earning participation units (rUv) that can be used to access the network's collective AI computing power.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              EDGE-NET: COLLECTIVE AI COMPUTING NETWORK                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐              │
│   │  Your       │       │  Collective │       │  AI Tasks   │              │
│   │  Browser    │◄─────►│  Network    │◄─────►│  Completed  │              │
│   │  (Idle CPU) │  P2P  │  (1000s)    │       │  for You    │              │
│   └─────────────┘       └─────────────┘       └─────────────┘              │
│         │                     │                     │                       │
│         ▼                     ▼                     ▼                       │
│   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐              │
│   │  Contribute │       │  Earn rUv   │       │  Use rUv    │              │
│   │  Compute    │  ───► │  Units      │  ───► │  for AI     │              │
│   │  When Idle  │       │  (Credits)  │       │  Workloads  │              │
│   └─────────────┘       └─────────────┘       └─────────────┘              │
│                                                                             │
│   Vector Search │ Embeddings │ Semantic Match │ Encryption │ Compression   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Table of Contents

- [What is Edge-Net?](#what-is-edge-net)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [AI Computing Tasks](#ai-computing-tasks)
- [Pi-Key Identity System](#pi-key-identity-system)
- [Self-Optimization](#self-optimization)
- [Tutorials](#tutorials)
- [API Reference](#api-reference)
- [Development](#development)

---

## What is Edge-Net?

Edge-net creates a **collective computing network** where participants share idle browser resources to power distributed AI workloads. Think of it as a cooperative where:

1. **You Contribute** - Share unused CPU cycles when browsing
2. **You Earn** - Accumulate rUv (Resource Utility Vouchers) based on contribution
3. **You Use** - Spend rUv to run AI tasks across the collective network
4. **Network Grows** - More participants = more collective computing power

### Why Collective AI Computing?

| Traditional AI Computing | Collective Edge-Net |
|-------------------------|---------------------|
| Expensive GPU servers | Free idle browser CPUs |
| Centralized data centers | Distributed global network |
| Pay-per-use pricing | Contribution-based access |
| Single point of failure | Resilient P2P mesh |
| Limited by your hardware | Scale with the collective |

### Core Principles

| Principle | Description |
|-----------|-------------|
| **Collectibility** | Resources are pooled and shared fairly |
| **Contribution** | Earn by giving, spend by using |
| **Self-Sustaining** | Network operates without central control |
| **Privacy-First** | Pi-Key cryptographic identity system |
| **Adaptive** | Q-learning security protects the collective |

---

## Key Features

### Collective Resource Sharing

| Feature | Benefit |
|---------|---------|
| **Idle CPU Utilization** | Use resources that would otherwise be wasted |
| **Browser-Based** | No installation, runs in any modern browser |
| **Adjustable Contribution** | Control how much you share (10-50% CPU) |
| **Battery Aware** | Automatically reduces on battery power |
| **Fair Distribution** | Work routed based on capability matching |

### AI Computing Capabilities

| Task Type | Use Case | How It Works |
|-----------|----------|--------------|
| **Vector Search** | Find similar items | k-NN across distributed index |
| **Embeddings** | Text understanding | Generate semantic vectors |
| **Semantic Match** | Intent detection | Classify meaning |
| **Encryption** | Data privacy | Secure distributed storage |
| **Compression** | Efficiency | Optimize data transfer |

### Pi-Key Identity System

Ultra-compact cryptographic identity using mathematical constants:

| Key Type | Size | Purpose |
|----------|------|---------|
| **π (Pi-Key)** | 40 bytes | Your permanent identity |
| **e (Session)** | 34 bytes | Temporary encrypted sessions |
| **φ (Genesis)** | 21 bytes | Network origin markers |

### Self-Optimizing Network

- **Automatic Task Routing** - Work goes to best-suited nodes
- **Topology Optimization** - Network self-organizes for efficiency
- **Q-Learning Security** - Learns to defend against threats
- **Economic Balance** - Self-sustaining resource economy

---

## Quick Start

### 1. Add to Your Website

```html
<script type="module">
  import init, { EdgeNetNode, EdgeNetConfig } from '@ruvector/edge-net';

  async function joinCollective() {
    await init();

    // Join the collective with your site ID
    const node = new EdgeNetConfig('my-website')
      .cpuLimit(0.3)          // Contribute 30% CPU when idle
      .memoryLimit(256 * 1024 * 1024)  // 256MB max
      .respectBattery(true)   // Reduce on battery
      .build();

    // Start contributing to the collective
    node.start();

    // Monitor your participation
    setInterval(() => {
      console.log(`Contributed: ${node.ruvBalance()} rUv`);
      console.log(`Tasks completed: ${node.getStats().tasks_completed}`);
    }, 10000);
  }

  joinCollective();
</script>
```

### 2. Use the Collective's AI Power

```javascript
// Submit an AI task to the collective
const result = await node.submitTask('vector_search', {
  query: embeddings,
  k: 10,
  index: 'shared-knowledge-base'
}, 5);  // Spend up to 5 rUv

console.log('Similar items:', result);
```

### 3. Monitor Your Contribution

```javascript
// Check your standing in the collective
const stats = node.getStats();
console.log(`
  rUv Earned: ${stats.ruv_earned}
  rUv Spent: ${stats.ruv_spent}
  Net Balance: ${stats.ruv_earned - stats.ruv_spent}
  Tasks Completed: ${stats.tasks_completed}
  Reputation: ${(stats.reputation * 100).toFixed(1)}%
`);
```

---

## How It Works

### The Contribution Cycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONTRIBUTION CYCLE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. CONTRIBUTE          2. EARN              3. USE                        │
│   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐              │
│   │   Browser   │       │    rUv      │       │  AI Tasks   │              │
│   │   detects   │  ───► │   credited  │  ───► │  submitted  │              │
│   │   idle time │       │   to you    │       │  to network │              │
│   └─────────────┘       └─────────────┘       └─────────────┘              │
│         │                     │                     │                       │
│         ▼                     ▼                     ▼                       │
│   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐              │
│   │  Process    │       │  10x boost  │       │  Results    │              │
│   │  incoming   │       │  for early  │       │  returned   │              │
│   │  tasks      │       │  adopters   │       │  to you     │              │
│   └─────────────┘       └─────────────┘       └─────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Network Growth Phases

The collective grows through natural phases:

| Phase | Size | Your Benefit |
|-------|------|--------------|
| **Genesis** | 0-10K nodes | 10x rUv multiplier (early adopter bonus) |
| **Growth** | 10K-50K | Multiplier decreases, network strengthens |
| **Maturation** | 50K-100K | Stable economy, high reliability |
| **Independence** | 100K+ | Self-sustaining, maximum collective power |

### Fair Resource Allocation

```javascript
// The network automatically optimizes task distribution
const health = JSON.parse(node.getEconomicHealth());

console.log(`
  Resource Velocity: ${health.velocity}      // How fast resources circulate
  Utilization: ${health.utilization}         // Network capacity used
  Growth Rate: ${health.growth}              // Network expansion
  Stability: ${health.stability}             // Economic equilibrium
`);
```

---

## AI Computing Tasks

### Vector Search (Distributed Similarity)

Find similar items across the collective's distributed index:

```javascript
// Search for similar documents
const similar = await node.submitTask('vector_search', {
  query: [0.1, 0.2, 0.3, ...],  // Your query vector
  k: 10,                         // Top 10 results
  index: 'shared-docs'           // Distributed index name
}, 3);  // Max 3 rUv

// Results from across the network
similar.forEach(item => {
  console.log(`Score: ${item.score}, ID: ${item.id}`);
});
```

### Embedding Generation

Generate semantic embeddings using collective compute:

```javascript
// Generate embeddings for text
const embeddings = await node.submitTask('embedding', {
  text: 'Your text to embed',
  model: 'sentence-transformer'
}, 2);

console.log('Embedding vector:', embeddings);
```

### Semantic Matching

Classify intent or meaning:

```javascript
// Classify text intent
const intent = await node.submitTask('semantic_match', {
  text: 'I want to cancel my subscription',
  categories: ['billing', 'support', 'sales', 'general']
}, 1);

console.log('Detected intent:', intent.category);
```

### Secure Operations

Encrypt data across the network:

```javascript
// Distributed encryption
const encrypted = await node.submitTask('encryption', {
  data: sensitiveData,
  operation: 'encrypt',
  key_id: 'my-shared-key'
}, 2);
```

---

## Pi-Key Identity System

Your identity in the collective uses mathematical constants for key sizes:

### Key Types

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PI-KEY IDENTITY SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   π Pi-Key (Identity)     e Euler-Key (Session)    φ Phi-Key (Genesis)     │
│   ┌─────────────────┐     ┌───────────────┐        ┌───────────────┐       │
│   │   314 bits      │     │   271 bits    │        │   161 bits    │       │
│   │   = 40 bytes    │     │   = 34 bytes  │        │   = 21 bytes  │       │
│   │                 │     │               │        │               │       │
│   │   Your unique   │     │   Temporary   │        │   Origin      │       │
│   │   identity      │     │   sessions    │        │   markers     │       │
│   │   (permanent)   │     │   (encrypted) │        │   (network)   │       │
│   └─────────────────┘     └───────────────┘        └───────────────┘       │
│                                                                             │
│   Ed25519 Signing         AES-256-GCM              SHA-256 Derived         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Using Pi-Keys

```javascript
import { PiKey, SessionKey, GenesisKey } from '@ruvector/edge-net';

// Create your permanent identity
const identity = new PiKey();
console.log(`Your ID: ${identity.getShortId()}`);  // π:a1b2c3d4...

// Sign data
const signature = identity.sign(data);
const valid = identity.verify(data, signature, identity.getPublicKey());

// Create encrypted backup
const backup = identity.createEncryptedBackup('my-password');

// Create temporary session
const session = SessionKey.create(identity, 3600);  // 1 hour
const encrypted = session.encrypt(sensitiveData);
const decrypted = session.decrypt(encrypted);
```

---

## Security Architecture

Edge-net implements production-grade cryptographic security:

### Cryptographic Primitives

| Component | Algorithm | Purpose |
|-----------|-----------|---------|
| **Key Derivation** | Argon2id (64MB, 3 iterations) | Memory-hard password hashing |
| **Signing** | Ed25519 | Digital signatures (128-bit security) |
| **Encryption** | AES-256-GCM | Authenticated encryption |
| **Hashing** | SHA-256 | Content hashing and verification |

### Identity Protection

```rust
// Password-protected key export with Argon2id + AES-256-GCM
let encrypted = identity.export_secret_key("strong_password")?;

// Secure memory cleanup (zeroize)
// All sensitive key material is automatically zeroed after use
```

### Authority Verification

All resolution events require cryptographic proof:

```rust
// Ed25519 signature verification for authority decisions
let signature = ScopedAuthority::sign_resolution(&resolution, &context, &signing_key);
// Signature verified against registered authority public keys
```

### Attack Resistance

The RAC (RuVector Adversarial Coherence) protocol defends against:

| Attack | Defense |
|--------|---------|
| **Sybil** | Stake-weighted voting, witness path diversity |
| **Eclipse** | Context isolation, Merkle divergence detection |
| **Byzantine** | 1/3 threshold, escalation tracking |
| **Replay** | Timestamp validation, duplicate detection |
| **Double-spend** | Conflict detection, quarantine system |

---

## Self-Optimization

The network continuously improves itself:

### Automatic Task Routing

```javascript
// Get optimal peers for your tasks
const peers = node.getOptimalPeers(5);

// Network learns from every interaction
node.recordTaskRouting('vector_search', 'peer-123', 45, true);
```

### Fitness-Based Evolution

```javascript
// High-performing nodes can replicate their config
if (node.shouldReplicate()) {
  const optimalConfig = node.getRecommendedConfig();
  // New nodes inherit successful configurations
}

// Track your contribution
const fitness = node.getNetworkFitness();  // 0.0 - 1.0
```

### Q-Learning Security

The collective learns to defend itself:

```javascript
// Run security audit
const audit = JSON.parse(node.runSecurityAudit());
console.log(`Security Score: ${audit.security_score}/10`);

// Defends against:
// - DDoS attacks
// - Sybil attacks
// - Byzantine behavior
// - Eclipse attacks
// - Replay attacks
```

---

## Tutorials

### Tutorial 1: Join the Collective

```javascript
import init, { EdgeNetConfig } from '@ruvector/edge-net';

async function joinCollective() {
  await init();

  // Configure your contribution
  const node = new EdgeNetConfig('my-site')
    .cpuLimit(0.25)           // 25% CPU when idle
    .memoryLimit(128 * 1024 * 1024)  // 128MB
    .minIdleTime(5000)        // Wait 5s of idle
    .respectBattery(true)     // Reduce on battery
    .build();

  // Join the network
  node.start();

  // Check your status
  console.log('Joined collective!');
  console.log(`Node ID: ${node.nodeId()}`);
  console.log(`Multiplier: ${node.getMultiplier()}x`);

  return node;
}
```

### Tutorial 2: Contribute and Earn

```javascript
async function contributeAndEarn(node) {
  // Process tasks from the collective
  let tasksCompleted = 0;

  while (true) {
    // Check if we should work
    if (node.isIdle()) {
      // Process a task from the network
      const processed = await node.processNextTask();

      if (processed) {
        tasksCompleted++;
        const stats = node.getStats();
        console.log(`Completed ${tasksCompleted} tasks, earned ${stats.ruv_earned} rUv`);
      }
    }

    await new Promise(r => setTimeout(r, 1000));
  }
}
```

### Tutorial 3: Use Collective AI Power

```javascript
async function useCollectiveAI(node) {
  // Check your balance
  const balance = node.ruvBalance();
  console.log(`Available: ${balance} rUv`);

  // Submit AI tasks
  const tasks = [
    { type: 'vector_search', cost: 3 },
    { type: 'embedding', cost: 2 },
    { type: 'semantic_match', cost: 1 }
  ];

  for (const task of tasks) {
    if (balance >= task.cost) {
      console.log(`Running ${task.type}...`);
      const result = await node.submitTask(
        task.type,
        { data: 'sample' },
        task.cost
      );
      console.log(`Result: ${JSON.stringify(result)}`);
    }
  }
}
```

### Tutorial 4: Monitor Network Health

```javascript
async function monitorHealth(node) {
  setInterval(() => {
    // Your contribution
    const stats = node.getStats();
    console.log(`
      === Your Contribution ===
      Earned: ${stats.ruv_earned} rUv
      Spent: ${stats.ruv_spent} rUv
      Tasks: ${stats.tasks_completed}
      Reputation: ${(stats.reputation * 100).toFixed(1)}%
    `);

    // Network health
    const health = JSON.parse(node.getEconomicHealth());
    console.log(`
      === Network Health ===
      Velocity: ${health.velocity.toFixed(2)}
      Utilization: ${(health.utilization * 100).toFixed(1)}%
      Stability: ${health.stability.toFixed(2)}
    `);

    // Check sustainability
    const sustainable = node.isSelfSustaining(10000, 50000);
    console.log(`Self-sustaining: ${sustainable}`);

  }, 30000);
}
```

---

## API Reference

### Core Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `new EdgeNetNode(siteId)` | Join the collective | `EdgeNetNode` |
| `start()` | Begin contributing | `void` |
| `pause()` / `resume()` | Control contribution | `void` |
| `ruvBalance()` | Check your credits | `u64` |
| `submitTask(type, payload, maxCost)` | Use collective compute | `Promise<Result>` |
| `processNextTask()` | Process work for others | `Promise<bool>` |

### Identity Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `new PiKey()` | Generate identity | `PiKey` |
| `getIdentity()` | Get 40-byte identity | `Vec<u8>` |
| `sign(data)` | Sign data | `Vec<u8>` |
| `verify(data, sig, pubkey)` | Verify signature | `bool` |
| `createEncryptedBackup(password)` | Backup identity | `Vec<u8>` |

### Network Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `getNetworkFitness()` | Your contribution score | `f32` |
| `getOptimalPeers(count)` | Best nodes for tasks | `Vec<String>` |
| `getEconomicHealth()` | Network health metrics | `String (JSON)` |
| `isSelfSustaining(nodes, tasks)` | Check sustainability | `bool` |

---

## Development

### Build

```bash
cd examples/edge-net
wasm-pack build --target web --out-dir pkg
```

### Test

```bash
cargo test
```

### Run Simulation

```bash
cd sim
npm install
npm run simulate
```

---

## Research Foundation

Edge-net is built on research in:

- **Distributed Computing** - P2P resource sharing
- **Collective Intelligence** - Emergent optimization
- **Game Theory** - Incentive-compatible mechanisms
- **Adaptive Security** - Q-learning threat response

---

## Disclaimer

Edge-net is a **research platform** for collective computing. The rUv units are:

- Resource participation metrics, not currency
- Used for balancing contribution and consumption
- Not redeemable for money or goods outside the network

---

## Links

- [Design Document](./DESIGN.md)
- [Technical Report](./docs/FINAL_REPORT.md)
- [Simulation Guide](./sim/README.md)
- [RuVector GitHub](https://github.com/ruvnet/ruvector)

## License

MIT License
