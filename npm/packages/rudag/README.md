# @ruvector/rudag

**Self-learning DAG query optimization with WASM acceleration**

Build, analyze, and optimize Directed Acyclic Graphs (DAGs) in JavaScript/TypeScript with blazing-fast WebAssembly performance and automatic browser persistence.

## What is a DAG?

A DAG (Directed Acyclic Graph) is a graph where edges flow in one direction and no cycles exist. They're everywhere in software:

```
     [Scan Table]
          |
          v
      [Filter]
       /    \
      v      v
 [Project] [Aggregate]
      \      /
       v    v
      [Output]
```

**Common uses:**
- Query execution plans (SQL optimizers)
- Task schedulers (build systems, CI/CD)
- Data pipelines (ETL, streaming)
- Dependency resolution (package managers)
- Workflow engines (approval flows, state machines)

## Why rudag?

| Problem | rudag Solution |
|---------|----------------|
| JavaScript is slow for graph algorithms | **WASM core** - Rust-powered, 10-100x faster |
| DAGs disappear on page refresh | **Auto-persistence** - IndexedDB in browser, files in Node.js |
| Finding bottlenecks is hard | **Critical path analysis** - Identify slowest execution path |
| No insight into node importance | **Attention mechanisms** - ML-inspired node scoring |
| Complex setup | **Zero config** - Works out of the box |

## Installation

```bash
npm install @ruvector/rudag
```

## Quick Start

```typescript
import { RuDag, DagOperator } from '@ruvector/rudag';

// Create a DAG (auto-persists to IndexedDB in browser)
const dag = new RuDag({ name: 'my-query' });
await dag.init();

// Add nodes with operators and costs
const scan = dag.addNode(DagOperator.SCAN, 100);      // Read table: 100ms
const filter = dag.addNode(DagOperator.FILTER, 10);   // Filter rows: 10ms
const project = dag.addNode(DagOperator.PROJECT, 5);  // Select columns: 5ms

// Connect nodes (creates edges)
dag.addEdge(scan, filter);
dag.addEdge(filter, project);

// Analyze the DAG
const topo = dag.topoSort();           // [0, 1, 2] - execution order
const { path, cost } = dag.criticalPath();  // Slowest path: 115ms

console.log(`Critical path: ${path.join(' → ')} (${cost}ms)`);
// Output: Critical path: 0 → 1 → 2 (115ms)

// Cleanup when done
dag.dispose();
```

## Features

### Core Operations

| Feature | Description |
|---------|-------------|
| `addNode(operator, cost)` | Add a node with operator type and execution cost |
| `addEdge(from, to)` | Connect nodes (rejects cycles automatically) |
| `topoSort()` | Get nodes in topological order |
| `criticalPath()` | Find the longest/most expensive path |
| `attention(mechanism)` | Score nodes by importance |

### Operators

```typescript
import { DagOperator } from '@ruvector/rudag';

DagOperator.SCAN       // 0 - Table scan
DagOperator.FILTER     // 1 - WHERE clause
DagOperator.PROJECT    // 2 - SELECT columns
DagOperator.JOIN       // 3 - Table join
DagOperator.AGGREGATE  // 4 - GROUP BY
DagOperator.SORT       // 5 - ORDER BY
DagOperator.LIMIT      // 6 - LIMIT/TOP
DagOperator.UNION      // 7 - UNION
DagOperator.CUSTOM     // 255 - Custom operator
```

### Attention Mechanisms

Score nodes by their importance using ML-inspired attention:

```typescript
import { AttentionMechanism } from '@ruvector/rudag';

// Score by position in execution order
const topoScores = dag.attention(AttentionMechanism.TOPOLOGICAL);

// Score by distance from critical path (most useful)
const criticalScores = dag.attention(AttentionMechanism.CRITICAL_PATH);

// Equal scores for all nodes
const uniformScores = dag.attention(AttentionMechanism.UNIFORM);
```

### Persistence

**Browser (IndexedDB) - Automatic:**
```typescript
const dag = new RuDag({ name: 'my-dag' }); // Auto-saves to IndexedDB
await dag.init();

// Later: reload from storage
const loaded = await RuDag.load('dag-123456-abc');

// List all stored DAGs
const allDags = await RuDag.listStored();

// Delete a DAG
await RuDag.deleteStored('dag-123456-abc');
```

**Node.js (File System):**
```typescript
import { NodeDagManager } from '@ruvector/rudag/node';

const manager = new NodeDagManager('./.rudag');
await manager.init();

const dag = await manager.createDag('pipeline');
// ... build the DAG ...
await manager.saveDag(dag);

// Later: reload
const loaded = await manager.loadDag('pipeline-id');
```

**Disable Persistence:**
```typescript
const dag = new RuDag({ storage: null, autoSave: false });
```

### Serialization

```typescript
// Binary (compact, fast)
const bytes = dag.toBytes();
const restored = await RuDag.fromBytes(bytes);

// JSON (human-readable)
const json = dag.toJSON();
const restored = await RuDag.fromJSON(json);
```

## CLI Tool

```bash
# Create a sample DAG
npx rudag create my-query > my-query.dag

# Show DAG information
npx rudag info my-query.dag

# Print topological sort
npx rudag topo my-query.dag

# Find critical path
npx rudag critical my-query.dag

# Compute attention scores
npx rudag attention my-query.dag critical

# Convert between formats
npx rudag convert my-query.dag my-query.json
npx rudag convert my-query.json my-query.dag

# JSON output
npx rudag info my-query.dag --json
```

## Use Cases

### 1. SQL Query Optimizer

Build a query plan DAG and find the critical path:

```typescript
import { RuDag, DagOperator } from '@ruvector/rudag';

async function analyzeQuery(sql: string) {
  const dag = new RuDag({ name: sql.slice(0, 50) });
  await dag.init();

  // Parse SQL and build DAG (simplified example)
  const scan1 = dag.addNode(DagOperator.SCAN, estimateScanCost('users'));
  const scan2 = dag.addNode(DagOperator.SCAN, estimateScanCost('orders'));
  const join = dag.addNode(DagOperator.JOIN, estimateJoinCost(1000, 5000));
  const filter = dag.addNode(DagOperator.FILTER, 10);
  const project = dag.addNode(DagOperator.PROJECT, 5);

  dag.addEdge(scan1, join);
  dag.addEdge(scan2, join);
  dag.addEdge(join, filter);
  dag.addEdge(filter, project);

  const { path, cost } = dag.criticalPath();
  console.log(`Estimated query time: ${cost}ms`);
  console.log(`Bottleneck: node ${path[0]}`); // Usually the scan or join

  return dag;
}
```

### 2. Task Scheduler

Schedule tasks respecting dependencies:

```typescript
import { RuDag, DagOperator } from '@ruvector/rudag';

interface Task {
  id: string;
  duration: number;
  dependencies: string[];
}

async function scheduleTasks(tasks: Task[]) {
  const dag = new RuDag({ name: 'task-schedule', storage: null });
  await dag.init();

  const taskToNode = new Map<string, number>();

  // Add all tasks as nodes
  for (const task of tasks) {
    const nodeId = dag.addNode(DagOperator.CUSTOM, task.duration);
    taskToNode.set(task.id, nodeId);
  }

  // Add dependencies as edges
  for (const task of tasks) {
    const toNode = taskToNode.get(task.id)!;
    for (const dep of task.dependencies) {
      const fromNode = taskToNode.get(dep)!;
      dag.addEdge(fromNode, toNode);
    }
  }

  // Get execution order
  const order = dag.topoSort();
  const schedule = order.map(nodeId => {
    const task = tasks.find(t => taskToNode.get(t.id) === nodeId)!;
    return task.id;
  });

  // Total time (critical path)
  const { cost } = dag.criticalPath();
  console.log(`Total time with parallelization: ${cost}ms`);

  dag.dispose();
  return schedule;
}
```

### 3. Build System

```typescript
import { RuDag, DagOperator } from '@ruvector/rudag';

const dag = new RuDag({ name: 'build' });
await dag.init();

// Define build steps
const compile = dag.addNode(DagOperator.CUSTOM, 5000);   // 5s
const test = dag.addNode(DagOperator.CUSTOM, 10000);     // 10s
const lint = dag.addNode(DagOperator.CUSTOM, 2000);      // 2s
const bundle = dag.addNode(DagOperator.CUSTOM, 3000);    // 3s
const deploy = dag.addNode(DagOperator.CUSTOM, 1000);    // 1s

dag.addEdge(compile, test);
dag.addEdge(compile, lint);
dag.addEdge(test, bundle);
dag.addEdge(lint, bundle);
dag.addEdge(bundle, deploy);

// Parallel execution order
const order = dag.topoSort(); // [compile, test|lint (parallel), bundle, deploy]

// Critical path: compile → test → bundle → deploy = 19s
const { cost } = dag.criticalPath();
console.log(`Minimum build time: ${cost}ms`);
```

### 4. Data Pipeline (ETL)

```typescript
import { RuDag, DagOperator, AttentionMechanism } from '@ruvector/rudag';

const pipeline = new RuDag({ name: 'etl-pipeline' });
await pipeline.init();

// Extract
const extractUsers = pipeline.addNode(DagOperator.SCAN, 1000);
const extractOrders = pipeline.addNode(DagOperator.SCAN, 2000);
const extractProducts = pipeline.addNode(DagOperator.SCAN, 500);

// Transform
const cleanUsers = pipeline.addNode(DagOperator.FILTER, 100);
const joinData = pipeline.addNode(DagOperator.JOIN, 3000);
const aggregate = pipeline.addNode(DagOperator.AGGREGATE, 500);

// Load
const loadWarehouse = pipeline.addNode(DagOperator.CUSTOM, 1000);

// Wire it up
pipeline.addEdge(extractUsers, cleanUsers);
pipeline.addEdge(cleanUsers, joinData);
pipeline.addEdge(extractOrders, joinData);
pipeline.addEdge(extractProducts, joinData);
pipeline.addEdge(joinData, aggregate);
pipeline.addEdge(aggregate, loadWarehouse);

// Find bottlenecks using attention scores
const scores = pipeline.attention(AttentionMechanism.CRITICAL_PATH);
console.log('Node importance:', scores);
// Nodes on critical path have higher scores
```

## Integration with Other Packages

### With Express.js (REST API)

```typescript
import express from 'express';
import { RuDag, DagOperator } from '@ruvector/rudag';
import { NodeDagManager } from '@ruvector/rudag/node';

const app = express();
const manager = new NodeDagManager('./data/dags');

app.use(express.json());

app.post('/dags', async (req, res) => {
  const dag = await manager.createDag(req.body.name);
  // ... add nodes from request ...
  await manager.saveDag(dag);
  res.json({ id: dag.getId() });
});

app.get('/dags/:id/critical-path', async (req, res) => {
  const dag = await manager.loadDag(req.params.id);
  if (!dag) return res.status(404).json({ error: 'Not found' });

  const result = dag.criticalPath();
  dag.dispose();
  res.json(result);
});

app.listen(3000);
```

### With React (State Management)

```typescript
import { useState, useEffect } from 'react';
import { RuDag, DagOperator } from '@ruvector/rudag';

function useDag(name: string) {
  const [dag, setDag] = useState<RuDag | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const init = async () => {
      const d = new RuDag({ name });
      await d.init();
      setDag(d);
      setLoading(false);
    };
    init();

    return () => dag?.dispose();
  }, [name]);

  return { dag, loading };
}

function DagViewer({ name }: { name: string }) {
  const { dag, loading } = useDag(name);
  const [criticalPath, setCriticalPath] = useState<number[]>([]);

  useEffect(() => {
    if (dag && dag.nodeCount > 0) {
      setCriticalPath(dag.criticalPath().path);
    }
  }, [dag]);

  if (loading) return <div>Loading...</div>;

  return (
    <div>
      <p>Nodes: {dag?.nodeCount}</p>
      <p>Critical Path: {criticalPath.join(' → ')}</p>
    </div>
  );
}
```

### With D3.js (Visualization)

```typescript
import * as d3 from 'd3';
import { RuDag, DagOperator } from '@ruvector/rudag';

async function visualizeDag(dag: RuDag, container: HTMLElement) {
  const nodes = dag.getNodes().map(n => ({
    id: n.id,
    label: DagOperator[n.operator],
    cost: n.cost,
  }));

  const topo = dag.topoSort();
  const { path: criticalPath } = dag.criticalPath();
  const criticalSet = new Set(criticalPath);

  // Create D3 visualization
  const svg = d3.select(container).append('svg');

  svg.selectAll('circle')
    .data(nodes)
    .enter()
    .append('circle')
    .attr('r', d => Math.sqrt(d.cost) * 2)
    .attr('fill', d => criticalSet.has(d.id) ? '#ff6b6b' : '#4dabf7')
    .attr('cx', (d, i) => 100 + topo.indexOf(d.id) * 150)
    .attr('cy', 100);
}
```

### With Bull (Job Queue)

```typescript
import Queue from 'bull';
import { RuDag, DagOperator } from '@ruvector/rudag';

const jobQueue = new Queue('dag-jobs');

async function queueDagExecution(dag: RuDag) {
  const order = dag.topoSort();
  const nodes = dag.getNodes();

  // Queue jobs in topological order with dependencies
  const jobIds: Record<number, string> = {};

  for (const nodeId of order) {
    const node = nodes.find(n => n.id === nodeId)!;

    const job = await jobQueue.add({
      nodeId,
      operator: node.operator,
      cost: node.cost,
    }, {
      // Jobs wait for their dependencies
      delay: 0,
    });

    jobIds[nodeId] = job.id as string;
  }

  return jobIds;
}
```

### With GraphQL

```typescript
import { ApolloServer, gql } from 'apollo-server';
import { RuDag, DagOperator } from '@ruvector/rudag';
import { NodeDagManager } from '@ruvector/rudag/node';

const manager = new NodeDagManager('./dags');

const typeDefs = gql`
  type Dag {
    id: String!
    name: String
    nodeCount: Int!
    edgeCount: Int!
    criticalPath: CriticalPath!
  }

  type CriticalPath {
    path: [Int!]!
    cost: Float!
  }

  type Query {
    dag(id: String!): Dag
    dags: [Dag!]!
  }
`;

const resolvers = {
  Query: {
    dag: async (_: any, { id }: { id: string }) => {
      const dag = await manager.loadDag(id);
      if (!dag) return null;

      const result = {
        id: dag.getId(),
        name: dag.getName(),
        nodeCount: dag.nodeCount,
        edgeCount: dag.edgeCount,
        criticalPath: dag.criticalPath(),
      };

      dag.dispose();
      return result;
    },
  },
};
```

### With RxJS (Reactive Streams)

```typescript
import { Subject, from } from 'rxjs';
import { mergeMap, toArray } from 'rxjs/operators';
import { RuDag, DagOperator } from '@ruvector/rudag';

async function executeWithRxJS(dag: RuDag) {
  const order = dag.topoSort();
  const nodes = dag.getNodes();

  const results$ = from(order).pipe(
    mergeMap(async (nodeId) => {
      const node = nodes.find(n => n.id === nodeId)!;

      // Simulate execution
      await new Promise(r => setTimeout(r, node.cost));

      return { nodeId, completed: true };
    }, 3), // Max 3 concurrent executions
    toArray()
  );

  return results$.toPromise();
}
```

## Performance

| Operation | rudag (WASM) | Pure JS |
|-----------|--------------|---------|
| Add 10k nodes | ~15ms | ~150ms |
| Topological sort (10k) | ~2ms | ~50ms |
| Critical path (10k) | ~3ms | ~80ms |
| Serialization (10k) | ~5ms | ~100ms |

## Browser Support

- Chrome 57+
- Firefox 52+
- Safari 11+
- Edge 79+

Requires WebAssembly support.

## API Reference

### RuDag

```typescript
class RuDag {
  constructor(options?: RuDagOptions);
  init(): Promise<this>;

  // Graph operations
  addNode(operator: DagOperator, cost: number, metadata?: object): number;
  addEdge(from: number, to: number): boolean;

  // Properties
  nodeCount: number;
  edgeCount: number;

  // Analysis
  topoSort(): number[];
  criticalPath(): { path: number[]; cost: number };
  attention(mechanism?: AttentionMechanism): number[];

  // Node access
  getNode(id: number): DagNode | undefined;
  getNodes(): DagNode[];

  // Serialization
  toBytes(): Uint8Array;
  toJSON(): string;

  // Persistence
  save(): Promise<StoredDag | null>;
  static load(id: string, storage?): Promise<RuDag | null>;
  static fromBytes(data: Uint8Array, options?): Promise<RuDag>;
  static fromJSON(json: string, options?): Promise<RuDag>;
  static listStored(storage?): Promise<StoredDag[]>;
  static deleteStored(id: string, storage?): Promise<boolean>;

  // Lifecycle
  getId(): string;
  getName(): string | undefined;
  setName(name: string): void;
  dispose(): void;
}
```

### Options

```typescript
interface RuDagOptions {
  id?: string;              // Custom ID (auto-generated if not provided)
  name?: string;            // Human-readable name
  storage?: Storage | null; // Persistence backend (null = disabled)
  autoSave?: boolean;       // Auto-save on changes (default: true)
  onSaveError?: (error) => void;  // Handle background save errors
}
```

## License

MIT OR Apache-2.0
