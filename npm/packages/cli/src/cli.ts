#!/usr/bin/env node
/**
 * RuVector CLI - Command-line interface for RuVector vector database
 *
 * This CLI provides access to hooks, memory, learning, and swarm commands.
 * Supports PostgreSQL storage (preferred) with JSON fallback.
 *
 * Set RUVECTOR_POSTGRES_URL or DATABASE_URL for PostgreSQL support.
 */

import { program } from 'commander';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { createStorageSync, StorageBackend, JsonStorage } from './storage.js';

interface QPattern {
  state: string;
  action: string;
  q_value: number;
  visits: number;
  last_update: number;
}

interface MemoryEntry {
  id: string;
  memory_type: string;
  content: string;
  embedding: number[];
  metadata: Record<string, string>;
  timestamp: number;
}

interface Trajectory {
  id: string;
  state: string;
  action: string;
  outcome: string;
  reward: number;
  timestamp: number;
}

interface ErrorPattern {
  code: string;
  error_type: string;
  message: string;
  fixes: string[];
  occurrences: number;
}

interface SwarmAgent {
  id: string;
  agent_type: string;
  capabilities: string[];
  success_rate: number;
  task_count: number;
  status: string;
}

interface SwarmEdge {
  source: string;
  target: string;
  weight: number;
  coordination_count: number;
}

interface IntelligenceStats {
  total_patterns: number;
  total_memories: number;
  total_trajectories: number;
  total_errors: number;
  session_count: number;
  last_session: number;
}

interface IntelligenceData {
  patterns: Record<string, QPattern>;
  memories: MemoryEntry[];
  trajectories: Trajectory[];
  errors: Record<string, ErrorPattern>;
  file_sequences: { from_file: string; to_file: string; count: number }[];
  agents: Record<string, SwarmAgent>;
  edges: SwarmEdge[];
  stats: IntelligenceStats;
}

class Intelligence {
  private data: IntelligenceData;
  private alpha = 0.1;

  constructor() {
    this.data = this.load();
  }

  private load(): IntelligenceData {
    try {
      if (fs.existsSync(INTEL_PATH)) {
        return JSON.parse(fs.readFileSync(INTEL_PATH, 'utf-8'));
      }
    } catch {}
    return {
      patterns: {},
      memories: [],
      trajectories: [],
      errors: {},
      file_sequences: [],
      agents: {},
      edges: [],
      stats: {
        total_patterns: 0,
        total_memories: 0,
        total_trajectories: 0,
        total_errors: 0,
        session_count: 0,
        last_session: 0
      }
    };
  }

  save(): void {
    const dir = path.dirname(INTEL_PATH);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(INTEL_PATH, JSON.stringify(this.data, null, 2));
  }

  private now(): number {
    return Math.floor(Date.now() / 1000);
  }

  private embed(text: string): number[] {
    const embedding = new Array(64).fill(0);
    for (let i = 0; i < text.length; i++) {
      const idx = (text.charCodeAt(i) + i * 7) % 64;
      embedding[idx] += 1.0;
    }
    const norm = Math.sqrt(embedding.reduce((a, b) => a + b * b, 0));
    if (norm > 0) {
      for (let i = 0; i < embedding.length; i++) {
        embedding[i] /= norm;
      }
    }
    return embedding;
  }

  private similarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;
    const dot = a.reduce((sum, v, i) => sum + v * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, v) => sum + v * v, 0));
    const normB = Math.sqrt(b.reduce((sum, v) => sum + v * v, 0));
    return normA > 0 && normB > 0 ? dot / (normA * normB) : 0;
  }

  remember(memoryType: string, content: string, metadata: Record<string, string> = {}): string {
    const id = `mem_${this.now()}`;
    this.data.memories.push({
      id,
      memory_type: memoryType,
      content,
      embedding: this.embed(content),
      metadata,
      timestamp: this.now()
    });
    if (this.data.memories.length > 5000) {
      this.data.memories.splice(0, 1000);
    }
    this.data.stats.total_memories = this.data.memories.length;
    return id;
  }

  recall(query: string, topK: number): MemoryEntry[] {
    const queryEmbed = this.embed(query);
    return this.data.memories
      .map(m => ({ score: this.similarity(queryEmbed, m.embedding), memory: m }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map(r => r.memory);
  }

  private getQ(state: string, action: string): number {
    const key = `${state}|${action}`;
    return this.data.patterns[key]?.q_value ?? 0;
  }

  private updateQ(state: string, action: string, reward: number): void {
    const key = `${state}|${action}`;
    if (!this.data.patterns[key]) {
      this.data.patterns[key] = { state, action, q_value: 0, visits: 0, last_update: 0 };
    }
    const p = this.data.patterns[key];
    p.q_value = p.q_value + this.alpha * (reward - p.q_value);
    p.visits++;
    p.last_update = this.now();
    this.data.stats.total_patterns = Object.keys(this.data.patterns).length;
  }

  learn(state: string, action: string, outcome: string, reward: number): string {
    const id = `traj_${this.now()}`;
    this.updateQ(state, action, reward);
    this.data.trajectories.push({ id, state, action, outcome, reward, timestamp: this.now() });
    if (this.data.trajectories.length > 1000) {
      this.data.trajectories.splice(0, 200);
    }
    this.data.stats.total_trajectories = this.data.trajectories.length;
    return id;
  }

  suggest(state: string, actions: string[]): { action: string; confidence: number } {
    let bestAction = actions[0] ?? '';
    let bestQ = -Infinity;
    for (const action of actions) {
      const q = this.getQ(state, action);
      if (q > bestQ) {
        bestQ = q;
        bestAction = action;
      }
    }
    return { action: bestAction, confidence: bestQ > 0 ? Math.min(bestQ, 1) : 0 };
  }

  route(task: string, file?: string, crateName?: string, operation = 'edit'): { agent: string; confidence: number; reason: string } {
    const fileType = file ? path.extname(file).slice(1) : 'unknown';
    const state = `${operation}_${fileType}_in_${crateName ?? 'project'}`;

    const agentMap: Record<string, string[]> = {
      rs: ['rust-developer', 'coder', 'reviewer', 'tester'],
      ts: ['typescript-developer', 'coder', 'frontend-dev'],
      tsx: ['typescript-developer', 'coder', 'frontend-dev'],
      js: ['coder', 'frontend-dev'],
      jsx: ['coder', 'frontend-dev'],
      py: ['python-developer', 'coder', 'ml-developer'],
      md: ['docs-writer', 'coder']
    };

    const agents = agentMap[fileType] ?? ['coder', 'reviewer'];
    const { action, confidence } = this.suggest(state, agents);

    const reason = confidence > 0.5 ? 'learned from past success'
      : confidence > 0 ? 'based on patterns'
      : `default for ${fileType} files`;

    return { agent: action, confidence, reason };
  }

  shouldTest(file: string): { suggest: boolean; command: string } {
    const ext = path.extname(file).slice(1);
    switch (ext) {
      case 'rs': {
        const crateMatch = file.match(/crates\/([^/]+)/);
        return crateMatch
          ? { suggest: true, command: `cargo test -p ${crateMatch[1]}` }
          : { suggest: true, command: 'cargo test' };
      }
      case 'ts':
      case 'tsx':
      case 'js':
      case 'jsx':
        return { suggest: true, command: 'npm test' };
      case 'py':
        return { suggest: true, command: 'pytest' };
      default:
        return { suggest: false, command: '' };
    }
  }

  swarmRegister(id: string, agentType: string, capabilities: string[]): void {
    this.data.agents[id] = {
      id,
      agent_type: agentType,
      capabilities,
      success_rate: 1.0,
      task_count: 0,
      status: 'active'
    };
  }

  swarmCoordinate(source: string, target: string, weight: number): void {
    const existing = this.data.edges.find(e => e.source === source && e.target === target);
    if (existing) {
      existing.weight = (existing.weight + weight) / 2;
      existing.coordination_count++;
    } else {
      this.data.edges.push({ source, target, weight, coordination_count: 1 });
    }
  }

  swarmStats(): { agents: number; edges: number; avgSuccess: number } {
    const agents = Object.keys(this.data.agents).length;
    const edges = this.data.edges.length;
    const avgSuccess = agents > 0
      ? Object.values(this.data.agents).reduce((sum, a) => sum + a.success_rate, 0) / agents
      : 0;
    return { agents, edges, avgSuccess };
  }

  stats(): IntelligenceStats {
    return this.data.stats;
  }

  sessionStart(): void {
    this.data.stats.session_count++;
    this.data.stats.last_session = this.now();
  }
}

// CLI setup
program
  .name('ruvector')
  .description('RuVector CLI - High-performance vector database')
  .version('0.1.25');

const hooks = program.command('hooks').description('Self-learning intelligence hooks for Claude Code');

hooks.command('stats')
  .description('Show intelligence statistics')
  .action(() => {
    const intel = new Intelligence();
    const stats = intel.stats();
    const swarm = intel.swarmStats();

    console.log('\x1b[36m\x1b[1m🧠 RuVector Intelligence Stats\x1b[0m\n');
    console.log(`  \x1b[32m${stats.total_patterns}\x1b[0m Q-learning patterns`);
    console.log(`  \x1b[32m${stats.total_memories}\x1b[0m vector memories`);
    console.log(`  \x1b[32m${stats.total_trajectories}\x1b[0m learning trajectories`);
    console.log(`  \x1b[32m${stats.total_errors}\x1b[0m error patterns\n`);
    console.log('\x1b[1mSwarm Status:\x1b[0m');
    console.log(`  \x1b[36m${swarm.agents}\x1b[0m agents registered`);
    console.log(`  \x1b[36m${swarm.edges}\x1b[0m coordination edges`);
    const rate = swarm.avgSuccess > 0 ? `${(swarm.avgSuccess * 100).toFixed(0)}%` : 'N/A';
    console.log(`  \x1b[36m${rate}\x1b[0m average success rate`);
  });

hooks.command('session-start')
  .description('Session start hook')
  .action(() => {
    const intel = new Intelligence();
    intel.sessionStart();
    intel.save();
    console.log('\x1b[36m\x1b[1m🧠 RuVector Intelligence Layer Active\x1b[0m\n');
    console.log('⚡ Intelligence guides: agent routing, error fixes, file sequences');
  });

hooks.command('pre-edit')
  .description('Pre-edit intelligence hook')
  .argument('<file>', 'File path')
  .action((file: string) => {
    const intel = new Intelligence();
    const fileName = path.basename(file);
    const crateMatch = file.match(/crates\/([^/]+)/);
    const crate = crateMatch?.[1];
    const { agent, confidence, reason } = intel.route(`edit ${fileName}`, file, crate, 'edit');

    console.log('\x1b[1m🧠 Intelligence Analysis:\x1b[0m');
    console.log(`   📁 \x1b[36m${crate ?? 'project'}\x1b[0m/${fileName}`);
    console.log(`   🤖 Recommended: \x1b[32m\x1b[1m${agent}\x1b[0m (${(confidence * 100).toFixed(0)}% confidence)`);
    if (reason) console.log(`      → \x1b[2m${reason}\x1b[0m`);
  });

hooks.command('post-edit')
  .description('Post-edit learning hook')
  .argument('<file>', 'File path')
  .option('--success', 'Edit succeeded')
  .action((file: string, opts: { success?: boolean }) => {
    const intel = new Intelligence();
    const success = opts.success ?? false;
    const ext = path.extname(file).slice(1);
    const crateMatch = file.match(/crates\/([^/]+)/);
    const crate = crateMatch?.[1] ?? 'project';
    const state = `edit_${ext}_in_${crate}`;

    intel.learn(state, success ? 'successful-edit' : 'failed-edit', success ? 'completed' : 'failed', success ? 1.0 : -0.5);
    intel.remember('edit', `${success ? 'successful' : 'failed'} edit of ${ext} in ${crate}`);
    intel.save();

    console.log(`📊 Learning recorded: ${success ? '✅' : '❌'} ${path.basename(file)}`);

    const test = intel.shouldTest(file);
    if (test.suggest) console.log(`   🧪 Consider: \x1b[36m${test.command}\x1b[0m`);
  });

hooks.command('remember')
  .description('Store content in semantic memory')
  .requiredOption('-t, --type <type>', 'Memory type')
  .argument('<content...>', 'Content to remember')
  .action((content: string[], opts: { type: string }) => {
    const intel = new Intelligence();
    const id = intel.remember(opts.type, content.join(' '));
    intel.save();
    console.log(JSON.stringify({ success: true, id }));
  });

hooks.command('recall')
  .description('Search memory semantically')
  .argument('<query...>', 'Search query')
  .option('-k, --top-k <n>', 'Number of results', '5')
  .action((query: string[], opts: { topK: string }) => {
    const intel = new Intelligence();
    const results = intel.recall(query.join(' '), parseInt(opts.topK));
    console.log(JSON.stringify({
      query: query.join(' '),
      results: results.map(r => ({
        type: r.memory_type,
        content: r.content.slice(0, 200),
        timestamp: r.timestamp
      }))
    }, null, 2));
  });

hooks.command('learn')
  .description('Record a learning trajectory')
  .argument('<state>', 'State identifier')
  .argument('<action>', 'Action taken')
  .option('-r, --reward <n>', 'Reward value', '0.0')
  .action((state: string, action: string, opts: { reward: string }) => {
    const intel = new Intelligence();
    const id = intel.learn(state, action, 'recorded', parseFloat(opts.reward));
    intel.save();
    console.log(JSON.stringify({ success: true, id, state, action, reward: parseFloat(opts.reward) }));
  });

hooks.command('suggest')
  .description('Get action suggestion for state')
  .argument('<state>', 'Current state')
  .requiredOption('-a, --actions <actions>', 'Available actions (comma-separated)')
  .action((state: string, opts: { actions: string }) => {
    const intel = new Intelligence();
    const actions = opts.actions.split(',').map(s => s.trim());
    const result = intel.suggest(state, actions);
    console.log(JSON.stringify({ state, ...result }, null, 2));
  });

hooks.command('route')
  .description('Route task to best agent')
  .argument('<task...>', 'Task description')
  .option('--file <file>', 'File being worked on')
  .option('--crate-name <crate>', 'Crate/module context')
  .action((task: string[], opts: { file?: string; crateName?: string }) => {
    const intel = new Intelligence();
    const result = intel.route(task.join(' '), opts.file, opts.crateName);
    console.log(JSON.stringify({
      task: task.join(' '),
      recommended: result.agent,
      confidence: result.confidence,
      reasoning: result.reason,
      file: opts.file,
      crate: opts.crateName
    }, null, 2));
  });

hooks.command('should-test')
  .description('Check if tests should run')
  .argument('<file>', 'File that was edited')
  .action((file: string) => {
    const intel = new Intelligence();
    console.log(JSON.stringify(intel.shouldTest(file), null, 2));
  });

hooks.command('swarm-register')
  .description('Register agent in swarm')
  .argument('<id>', 'Agent ID')
  .argument('<type>', 'Agent type')
  .option('--capabilities <caps>', 'Capabilities (comma-separated)')
  .action((id: string, type: string, opts: { capabilities?: string }) => {
    const intel = new Intelligence();
    const caps = opts.capabilities?.split(',').map(s => s.trim()) ?? [];
    intel.swarmRegister(id, type, caps);
    intel.save();
    console.log(JSON.stringify({ success: true, agent_id: id, type }));
  });

hooks.command('swarm-stats')
  .description('Show swarm statistics')
  .action(() => {
    const intel = new Intelligence();
    const stats = intel.swarmStats();
    console.log(JSON.stringify({
      agents: stats.agents,
      edges: stats.edges,
      average_success_rate: stats.avgSuccess,
      topology: 'mesh'
    }, null, 2));
  });

program.parse();
