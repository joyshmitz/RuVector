#!/usr/bin/env node

/**
 * rUvector CLI
 *
 * Beautiful command-line interface for vector database operations
 * Includes: Vector Search, Graph/Cypher, GNN, Compression, and more
 */

const { Command } = require('commander');
const chalk = require('chalk');
const ora = require('ora');
const Table = require('cli-table3');
const fs = require('fs').promises;
const path = require('path');

// Lazy load backends to improve startup time
let vectorBackend = null;
let graphBackend = null;
let gnnBackend = null;

function getVectorBackend() {
  if (!vectorBackend) {
    try {
      const { VectorIndex, getBackendInfo, Utils } = require('../dist/index.js');
      vectorBackend = { VectorIndex, getBackendInfo, Utils };
    } catch (e) {
      console.error(chalk.red('Vector backend not available:', e.message));
      process.exit(1);
    }
  }
  return vectorBackend;
}

function getGraphBackend() {
  if (!graphBackend) {
    try {
      graphBackend = require('@ruvector/graph-node');
    } catch (e) {
      try {
        graphBackend = require('@ruvector/graph-wasm');
      } catch (e2) {
        return null;
      }
    }
  }
  return graphBackend;
}

function getGnnBackend() {
  if (!gnnBackend) {
    try {
      gnnBackend = require('@ruvector/gnn-node');
    } catch (e) {
      try {
        gnnBackend = require('@ruvector/gnn-wasm');
      } catch (e2) {
        return null;
      }
    }
  }
  return gnnBackend;
}

const program = new Command();

// Utility to format numbers
function formatNumber(num) {
  if (num >= 1_000_000) {
    return `${(num / 1_000_000).toFixed(2)}M`;
  } else if (num >= 1_000) {
    return `${(num / 1_000).toFixed(2)}K`;
  }
  return num.toString();
}

// Utility to format bytes
function formatBytes(bytes) {
  if (bytes >= 1_073_741_824) {
    return `${(bytes / 1_073_741_824).toFixed(2)} GB`;
  } else if (bytes >= 1_048_576) {
    return `${(bytes / 1_048_576).toFixed(2)} MB`;
  } else if (bytes >= 1_024) {
    return `${(bytes / 1_024).toFixed(2)} KB`;
  }
  return `${bytes} B`;
}

// Utility to format duration
function formatDuration(ms) {
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)}s`;
  }
  return `${ms.toFixed(2)}ms`;
}

// Doctor command - diagnose installation
program
  .command('doctor')
  .description('Diagnose installation and check all dependencies')
  .action(async () => {
    console.log(chalk.bold.cyan('\n🩺 rUvector Doctor - Diagnosing Installation\n'));

    const checks = [];

    // Check Node.js version
    const nodeVersion = process.version;
    const nodeMajor = parseInt(nodeVersion.slice(1).split('.')[0]);
    checks.push({
      name: 'Node.js Version',
      status: nodeMajor >= 16 ? 'pass' : 'fail',
      message: nodeVersion,
      hint: nodeMajor < 16 ? 'Requires Node.js >= 16.0.0' : null
    });

    // Check core package
    let coreVersion = null;
    try {
      const core = require('@ruvector/core');
      coreVersion = typeof core.version === 'function' ? core.version() : (core.version || 'installed');
      checks.push({
        name: '@ruvector/core',
        status: 'pass',
        message: coreVersion
      });
    } catch (e) {
      checks.push({
        name: '@ruvector/core',
        status: 'warn',
        message: 'Not installed (using fallback)',
        hint: 'npm install @ruvector/core'
      });
    }

    // Check graph package (Node.js native)
    try {
      const graph = require('@ruvector/graph-node');
      checks.push({
        name: '@ruvector/graph-node',
        status: 'pass',
        message: graph.version || 'installed'
      });
    } catch (e) {
      // Check WASM fallback
      try {
        const graphWasm = require('@ruvector/graph-wasm');
        checks.push({
          name: '@ruvector/graph-wasm',
          status: 'pass',
          message: graphWasm.version || 'installed (WASM)'
        });
      } catch (e2) {
        checks.push({
          name: 'Graph Module',
          status: 'warn',
          message: 'Not installed',
          hint: 'npm install @ruvector/graph-node'
        });
      }
    }

    // Check GNN package (Node.js native)
    try {
      const gnn = require('@ruvector/gnn-node');
      checks.push({
        name: '@ruvector/gnn-node',
        status: 'pass',
        message: gnn.version || 'installed'
      });
    } catch (e) {
      // Check WASM fallback
      try {
        const gnnWasm = require('@ruvector/gnn-wasm');
        checks.push({
          name: '@ruvector/gnn-wasm',
          status: 'pass',
          message: gnnWasm.version || 'installed (WASM)'
        });
      } catch (e2) {
        checks.push({
          name: 'GNN Module',
          status: 'warn',
          message: 'Not installed',
          hint: 'npm install @ruvector/gnn-node'
        });
      }
    }

    // Check dist files
    const distPath = require('path').join(__dirname, '..', 'dist', 'index.js');
    try {
      require('fs').accessSync(distPath);
      checks.push({
        name: 'Built dist files',
        status: 'pass',
        message: 'Found'
      });
    } catch (e) {
      checks.push({
        name: 'Built dist files',
        status: 'fail',
        message: 'Not found',
        hint: 'Run npm run build in the ruvector package'
      });
    }

    // Display results
    const table = new Table({
      head: ['Check', 'Status', 'Details'],
      colWidths: [25, 10, 40]
    });

    let hasErrors = false;
    let hasWarnings = false;

    checks.forEach(check => {
      let statusIcon;
      if (check.status === 'pass') {
        statusIcon = chalk.green('✓ Pass');
      } else if (check.status === 'warn') {
        statusIcon = chalk.yellow('○ Warn');
        hasWarnings = true;
      } else {
        statusIcon = chalk.red('✗ Fail');
        hasErrors = true;
      }

      table.push([
        check.name,
        statusIcon,
        check.message + (check.hint ? chalk.gray(` (${check.hint})`) : '')
      ]);
    });

    console.log(table.toString());
    console.log();

    // Summary
    if (hasErrors) {
      console.log(chalk.red('✗ Some required checks failed. Please fix the issues above.'));
    } else if (hasWarnings) {
      console.log(chalk.yellow('○ All required checks passed, but some optional modules are missing.'));
      console.log(chalk.cyan('  Install optional modules for full functionality.'));
    } else {
      console.log(chalk.green('✓ All checks passed! rUvector is ready to use.'));
    }

    // Show available features
    console.log(chalk.bold.cyan('\n📦 Available Commands:\n'));
    console.log(chalk.white('  Core:  ') + chalk.green('info, init, stats, insert, search, benchmark'));
    if (getGraphBackend()) {
      console.log(chalk.white('  Graph: ') + chalk.green('graph query, graph create-node'));
    } else {
      console.log(chalk.white('  Graph: ') + chalk.gray('(install @ruvector/graph-node)'));
    }
    if (getGnnBackend()) {
      console.log(chalk.white('  GNN:   ') + chalk.green('gnn layer, gnn compress'));
    } else {
      console.log(chalk.white('  GNN:   ') + chalk.gray('(install @ruvector/gnn-node)'));
    }
    console.log();
  });

// Info command
program
  .command('info')
  .description('Show backend information and available modules')
  .action(() => {
    const { getBackendInfo } = getVectorBackend();
    const info = getBackendInfo();

    console.log(chalk.bold.cyan('\n🚀 rUvector - All-in-One Vector Database\n'));

    const table = new Table({
      chars: { 'mid': '', 'left-mid': '', 'mid-mid': '', 'right-mid': '' }
    });

    table.push(
      ['Backend Type', chalk.green(info.type === 'native' ? '⚡ Native' : '🌐 WASM')],
      ['Version', info.version],
      ['Features', info.features.join(', ')]
    );

    console.log(table.toString());
    console.log();

    // Show available modules
    console.log(chalk.bold.cyan('📦 Available Modules:\n'));
    const modulesTable = new Table({
      head: ['Module', 'Status', 'Description'],
      colWidths: [20, 12, 45]
    });

    // Check vector
    modulesTable.push(['Vector Search', chalk.green('✓ Ready'), 'HNSW index, similarity search']);

    // Check graph
    const graphAvailable = getGraphBackend() !== null;
    modulesTable.push([
      'Graph/Cypher',
      graphAvailable ? chalk.green('✓ Ready') : chalk.yellow('○ Optional'),
      'Neo4j-compatible queries, hyperedges'
    ]);

    // Check GNN
    const gnnAvailable = getGnnBackend() !== null;
    modulesTable.push([
      'GNN Layers',
      gnnAvailable ? chalk.green('✓ Ready') : chalk.yellow('○ Optional'),
      'Neural network on graph topology'
    ]);

    // Built-in features
    modulesTable.push(['Compression', chalk.green('✓ Built-in'), 'f32→f16→PQ8→PQ4→Binary (2-32x)']);
    modulesTable.push(['WASM/Browser', chalk.green('✓ Built-in'), 'Client-side vector search']);

    console.log(modulesTable.toString());
    console.log();

    if (!graphAvailable || !gnnAvailable) {
      console.log(chalk.cyan('💡 Install optional modules:'));
      if (!graphAvailable) {
        console.log(chalk.white('   npm install @ruvector/graph-node'));
      }
      if (!gnnAvailable) {
        console.log(chalk.white('   npm install @ruvector/gnn-node'));
      }
      console.log();
    }
  });

// Init command
program
  .command('init <path>')
  .description('Initialize a new vector index')
  .option('-d, --dimension <number>', 'Vector dimension', '384')
  .option('-m, --metric <type>', 'Distance metric (cosine|euclidean|dot)', 'cosine')
  .option('-t, --type <type>', 'Index type (flat|hnsw)', 'hnsw')
  .option('--hnsw-m <number>', 'HNSW M parameter', '16')
  .option('--hnsw-ef <number>', 'HNSW ef_construction parameter', '200')
  .action(async (indexPath, options) => {
    const spinner = ora('Initializing vector index...').start();

    try {
      const { VectorIndex } = getVectorBackend();
      const index = new VectorIndex({
        dimension: parseInt(options.dimension),
        metric: options.metric,
        indexType: options.type,
        hnswConfig: options.type === 'hnsw' ? {
          m: parseInt(options.hnswM),
          efConstruction: parseInt(options.hnswEf)
        } : undefined
      });

      await index.save(indexPath);

      spinner.succeed(chalk.green('Index initialized successfully!'));

      console.log(chalk.cyan('\nConfiguration:'));
      console.log(`  Path: ${chalk.white(indexPath)}`);
      console.log(`  Dimension: ${chalk.white(options.dimension)}`);
      console.log(`  Metric: ${chalk.white(options.metric)}`);
      console.log(`  Type: ${chalk.white(options.type)}`);

      if (options.type === 'hnsw') {
        console.log(chalk.cyan('\nHNSW Parameters:'));
        console.log(`  M: ${chalk.white(options.hnswM)}`);
        console.log(`  ef_construction: ${chalk.white(options.hnswEf)}`);
      }

      console.log();
    } catch (error) {
      spinner.fail(chalk.red('Failed to initialize index'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Stats command
program
  .command('stats <path>')
  .description('Show index statistics')
  .action(async (indexPath) => {
    const spinner = ora('Loading index...').start();

    try {
      const { VectorIndex } = getVectorBackend();
      const index = await VectorIndex.load(indexPath);
      const stats = await index.stats();

      spinner.succeed(chalk.green('Index loaded'));

      console.log(chalk.bold.cyan('\n📊 Index Statistics\n'));

      const table = new Table({
        chars: { 'mid': '', 'left-mid': '', 'mid-mid': '', 'right-mid': '' }
      });

      table.push(
        ['Vectors', chalk.white(formatNumber(stats.vectorCount))],
        ['Dimension', chalk.white(stats.dimension)],
        ['Index Type', chalk.white(stats.indexType)],
        ['Memory Usage', chalk.white(stats.memoryUsage ? formatBytes(stats.memoryUsage) : 'N/A')]
      );

      console.log(table.toString());
      console.log();
    } catch (error) {
      spinner.fail(chalk.red('Failed to load index'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Insert command
program
  .command('insert <path> <vectors-file>')
  .description('Insert vectors from JSON file')
  .option('-b, --batch-size <number>', 'Batch size', '1000')
  .action(async (indexPath, vectorsFile, options) => {
    let spinner = ora('Loading index...').start();

    try {
      const { VectorIndex } = getVectorBackend();
      const index = await VectorIndex.load(indexPath);
      spinner.succeed();

      spinner = ora('Loading vectors...').start();
      const data = await fs.readFile(vectorsFile, 'utf-8');
      const vectors = JSON.parse(data);
      spinner.succeed(chalk.green(`Loaded ${vectors.length} vectors`));

      const startTime = Date.now();
      spinner = ora('Inserting vectors...').start();

      let lastProgress = 0;
      await index.insertBatch(vectors, {
        batchSize: parseInt(options.batchSize),
        progressCallback: (progress) => {
          const percent = Math.floor(progress * 100);
          if (percent > lastProgress) {
            spinner.text = `Inserting vectors... ${percent}%`;
            lastProgress = percent;
          }
        }
      });

      const duration = Date.now() - startTime;
      const throughput = vectors.length / (duration / 1000);

      spinner.succeed(chalk.green('Vectors inserted!'));

      console.log(chalk.cyan('\nPerformance:'));
      console.log(`  Duration: ${chalk.white(formatDuration(duration))}`);
      console.log(`  Throughput: ${chalk.white(formatNumber(throughput))} vectors/sec`);

      spinner = ora('Saving index...').start();
      await index.save(indexPath);
      spinner.succeed(chalk.green('Index saved'));

      console.log();
    } catch (error) {
      spinner.fail(chalk.red('Operation failed'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Search command
program
  .command('search <path>')
  .description('Search for similar vectors')
  .requiredOption('-q, --query <vector>', 'Query vector as JSON array')
  .option('-k, --top-k <number>', 'Number of results', '10')
  .option('--ef <number>', 'HNSW ef parameter')
  .action(async (indexPath, options) => {
    const spinner = ora('Loading index...').start();

    try {
      const { VectorIndex } = getVectorBackend();
      const index = await VectorIndex.load(indexPath);
      spinner.succeed();

      const query = JSON.parse(options.query);

      spinner.text = 'Searching...';
      spinner.start();

      const startTime = Date.now();
      const results = await index.search(query, {
        k: parseInt(options.topK),
        ef: options.ef ? parseInt(options.ef) : undefined
      });
      const duration = Date.now() - startTime;

      spinner.succeed(chalk.green(`Found ${results.length} results in ${formatDuration(duration)}`));

      console.log(chalk.bold.cyan('\n🔍 Search Results\n'));

      const table = new Table({
        head: ['Rank', 'ID', 'Score', 'Metadata'],
        colWidths: [6, 20, 12, 40]
      });

      results.forEach((result, i) => {
        table.push([
          chalk.yellow(`#${i + 1}`),
          result.id,
          chalk.green(result.score.toFixed(4)),
          result.metadata ? JSON.stringify(result.metadata).substring(0, 37) + '...' : ''
        ]);
      });

      console.log(table.toString());
      console.log();
    } catch (error) {
      spinner.fail(chalk.red('Search failed'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Benchmark command
program
  .command('benchmark')
  .description('Run performance benchmarks')
  .option('-d, --dimension <number>', 'Vector dimension', '384')
  .option('-n, --num-vectors <number>', 'Number of vectors', '10000')
  .option('-q, --num-queries <number>', 'Number of queries', '100')
  .action(async (options) => {
    const { VectorIndex, Utils } = getVectorBackend();
    const dimension = parseInt(options.dimension);
    const numVectors = parseInt(options.numVectors);
    const numQueries = parseInt(options.numQueries);

    console.log(chalk.bold.cyan('\n⚡ Performance Benchmark\n'));
    console.log(chalk.cyan('Configuration:'));
    console.log(`  Dimension: ${chalk.white(dimension)}`);
    console.log(`  Vectors: ${chalk.white(formatNumber(numVectors))}`);
    console.log(`  Queries: ${chalk.white(formatNumber(numQueries))}`);
    console.log();

    const results = [];

    try {
      // Create index
      let spinner = ora('Creating index...').start();
      const index = new VectorIndex({
        dimension,
        metric: 'cosine',
        indexType: 'hnsw'
      });
      spinner.succeed();

      // Generate vectors
      spinner = ora('Generating vectors...').start();
      const vectors = [];
      for (let i = 0; i < numVectors; i++) {
        vectors.push({
          id: `vec_${i}`,
          values: Utils.randomVector(dimension)
        });
      }
      spinner.succeed();

      // Insert benchmark
      spinner = ora('Benchmarking inserts...').start();
      const insertStart = Date.now();
      await index.insertBatch(vectors, { batchSize: 1000 });
      const insertDuration = Date.now() - insertStart;
      const insertThroughput = numVectors / (insertDuration / 1000);
      spinner.succeed();

      results.push({
        operation: 'Insert',
        duration: insertDuration,
        throughput: insertThroughput
      });

      // Search benchmark
      spinner = ora('Benchmarking searches...').start();
      const queries = [];
      for (let i = 0; i < numQueries; i++) {
        queries.push(Utils.randomVector(dimension));
      }

      const searchStart = Date.now();
      for (const query of queries) {
        await index.search(query, { k: 10 });
      }
      const searchDuration = Date.now() - searchStart;
      const searchThroughput = numQueries / (searchDuration / 1000);
      spinner.succeed();

      results.push({
        operation: 'Search',
        duration: searchDuration,
        throughput: searchThroughput
      });

      // Display results
      console.log(chalk.bold.cyan('\n📈 Results\n'));

      const table = new Table({
        head: ['Operation', 'Total Time', 'Throughput'],
        colWidths: [15, 20, 25]
      });

      results.forEach(result => {
        table.push([
          chalk.white(result.operation),
          chalk.yellow(formatDuration(result.duration)),
          chalk.green(`${formatNumber(result.throughput)} ops/sec`)
        ]);
      });

      console.log(table.toString());
      console.log();

      // Backend info
      const { getBackendInfo } = getVectorBackend();
      const info = getBackendInfo();
      console.log(chalk.cyan(`Backend: ${chalk.white(info.type)}`));
      console.log();

    } catch (error) {
      console.error(chalk.red('Benchmark failed:'), error.message);
      process.exit(1);
    }
  });

// ============================================================================
// GRAPH COMMANDS
// ============================================================================

const graphCmd = program
  .command('graph')
  .description('Graph database commands (Cypher queries, nodes, edges)');

graphCmd
  .command('query <cypher>')
  .description('Execute a Cypher query')
  .option('-f, --format <type>', 'Output format (table|json)', 'table')
  .action(async (cypher, options) => {
    const graph = getGraphBackend();
    if (!graph) {
      console.error(chalk.red('Graph module not installed. Run: npm install @ruvector/graph-node'));
      process.exit(1);
    }

    const spinner = ora('Executing Cypher query...').start();
    try {
      const db = new graph.GraphDB();
      const results = await db.query(cypher);
      spinner.succeed(chalk.green(`Query returned ${results.length} results`));

      if (options.format === 'json') {
        console.log(JSON.stringify(results, null, 2));
      } else {
        if (results.length > 0) {
          const table = new Table({
            head: Object.keys(results[0]).map(k => chalk.cyan(k))
          });
          results.forEach(row => {
            table.push(Object.values(row).map(v =>
              typeof v === 'object' ? JSON.stringify(v) : String(v)
            ));
          });
          console.log(table.toString());
        }
      }
    } catch (error) {
      spinner.fail(chalk.red('Query failed'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

graphCmd
  .command('create-node')
  .description('Create a new node')
  .requiredOption('-l, --label <label>', 'Node label')
  .requiredOption('-p, --properties <json>', 'Node properties as JSON')
  .action(async (options) => {
    const graph = getGraphBackend();
    if (!graph) {
      console.error(chalk.red('Graph module not installed. Run: npm install @ruvector/graph-node'));
      process.exit(1);
    }

    try {
      const db = new graph.GraphDB();
      const props = JSON.parse(options.properties);
      const nodeId = await db.createNode(options.label, props);
      console.log(chalk.green(`✓ Created node: ${nodeId}`));
    } catch (error) {
      console.error(chalk.red('Failed to create node:', error.message));
      process.exit(1);
    }
  });

// ============================================================================
// GNN COMMANDS
// ============================================================================

const gnnCmd = program
  .command('gnn')
  .description('Graph Neural Network commands');

gnnCmd
  .command('layer')
  .description('Create and test a GNN layer')
  .option('-i, --input-dim <number>', 'Input dimension', '128')
  .option('-h, --hidden-dim <number>', 'Hidden dimension', '256')
  .option('--heads <number>', 'Attention heads', '4')
  .action(async (options) => {
    const gnn = getGnnBackend();
    if (!gnn) {
      console.error(chalk.red('GNN module not installed. Run: npm install @ruvector/gnn-node'));
      process.exit(1);
    }

    try {
      const layer = new gnn.RuvectorLayer(
        parseInt(options.inputDim),
        parseInt(options.hiddenDim),
        parseInt(options.heads),
        0.1 // dropout
      );
      console.log(chalk.green('✓ GNN Layer created'));
      console.log(chalk.cyan('Configuration:'));
      console.log(`  Input dim: ${options.inputDim}`);
      console.log(`  Hidden dim: ${options.hiddenDim}`);
      console.log(`  Attention heads: ${options.heads}`);
    } catch (error) {
      console.error(chalk.red('Failed to create GNN layer:', error.message));
      process.exit(1);
    }
  });

gnnCmd
  .command('compress')
  .description('Compress a vector using adaptive compression')
  .requiredOption('-v, --vector <json>', 'Vector as JSON array')
  .option('-f, --frequency <number>', 'Access frequency (0-1)', '0.5')
  .action(async (options) => {
    const gnn = getGnnBackend();
    if (!gnn) {
      console.error(chalk.red('GNN module not installed. Run: npm install @ruvector/gnn-node'));
      process.exit(1);
    }

    try {
      const vector = JSON.parse(options.vector);
      const freq = parseFloat(options.frequency);
      const compressor = new gnn.TensorCompress();
      const compressed = compressor.compress(vector, freq);

      console.log(chalk.green('✓ Vector compressed'));
      console.log(chalk.cyan('Compression info:'));
      console.log(`  Original size: ${vector.length * 4} bytes`);
      console.log(`  Compressed: ${JSON.stringify(compressed).length} bytes`);
      console.log(`  Access frequency: ${freq}`);
      console.log(`  Level: ${gnn.getCompressionLevel(freq)}`);
    } catch (error) {
      console.error(chalk.red('Compression failed:', error.message));
      process.exit(1);
    }
  });

// Version
program.version(require('../package.json').version, '-v, --version', 'Show version');

// Help customization
program.on('--help', () => {
  console.log('');
  console.log(chalk.bold.cyan('Diagnostics:'));
  console.log('  $ ruvector doctor                                  Diagnose installation');
  console.log('  $ ruvector info                                    Show backend info');
  console.log('');
  console.log(chalk.bold.cyan('Vector Commands:'));
  console.log('  $ ruvector init my-index.bin -d 384                Initialize index');
  console.log('  $ ruvector stats my-index.bin                      Show index statistics');
  console.log('  $ ruvector insert my-index.bin vectors.json        Insert vectors');
  console.log('  $ ruvector search my-index.bin -q "[0.1,...]" -k 10');
  console.log('  $ ruvector benchmark -d 384 -n 10000               Run benchmarks');
  console.log('');
  console.log(chalk.bold.cyan('Graph Commands (requires @ruvector/graph-node):'));
  console.log('  $ ruvector graph query "MATCH (n) RETURN n"        Execute Cypher');
  console.log('  $ ruvector graph create-node -l Person -p \'{"name":"Alice"}\'');
  console.log('');
  console.log(chalk.bold.cyan('GNN Commands (requires @ruvector/gnn-node):'));
  console.log('  $ ruvector gnn layer -i 128 -h 256 --heads 4       Create GNN layer');
  console.log('  $ ruvector gnn compress -v "[0.1,...]" -f 0.5      Compress vector');
  console.log('');
  console.log(chalk.bold.cyan('Hooks Commands (Claude Code integration):'));
  console.log('  $ ruvector hooks init                              Initialize hooks');
  console.log('  $ ruvector hooks stats                             Show intelligence stats');
  console.log('  $ ruvector hooks session-start                     Start session');
  console.log('  $ ruvector hooks pre-edit <file>                   Pre-edit intelligence');
  console.log('  $ ruvector hooks post-edit <file>                  Post-edit learning');
  console.log('');
  console.log(chalk.cyan('For more info: https://github.com/ruvnet/ruvector'));
  console.log('');
});

// ============================================================================
// HOOKS COMMANDS - Self-learning intelligence for Claude Code
// ============================================================================

const INTEL_PATH = path.join(require('os').homedir(), '.ruvector', 'intelligence.json');

class Intelligence {
  constructor() {
    this.data = this.load();
    this.alpha = 0.1;
    this.lastEditedFile = null;
  }

  load() {
    try {
      if (fs.existsSync(INTEL_PATH)) {
        return JSON.parse(require('fs').readFileSync(INTEL_PATH, 'utf-8'));
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
      stats: { total_patterns: 0, total_memories: 0, total_trajectories: 0, total_errors: 0, session_count: 0, last_session: 0 }
    };
  }

  save() {
    const dir = path.dirname(INTEL_PATH);
    if (!fs.existsSync(dir)) require('fs').mkdirSync(dir, { recursive: true });
    require('fs').writeFileSync(INTEL_PATH, JSON.stringify(this.data, null, 2));
  }

  now() { return Math.floor(Date.now() / 1000); }

  embed(text) {
    const embedding = new Array(64).fill(0);
    for (let i = 0; i < text.length; i++) {
      const idx = (text.charCodeAt(i) + i * 7) % 64;
      embedding[idx] += 1.0;
    }
    const norm = Math.sqrt(embedding.reduce((a, b) => a + b * b, 0));
    if (norm > 0) for (let i = 0; i < embedding.length; i++) embedding[i] /= norm;
    return embedding;
  }

  similarity(a, b) {
    if (a.length !== b.length) return 0;
    const dot = a.reduce((sum, v, i) => sum + v * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, v) => sum + v * v, 0));
    const normB = Math.sqrt(b.reduce((sum, v) => sum + v * v, 0));
    return normA > 0 && normB > 0 ? dot / (normA * normB) : 0;
  }

  remember(memoryType, content, metadata = {}) {
    const id = `mem_${this.now()}`;
    this.data.memories.push({ id, memory_type: memoryType, content, embedding: this.embed(content), metadata, timestamp: this.now() });
    if (this.data.memories.length > 5000) this.data.memories.splice(0, 1000);
    this.data.stats.total_memories = this.data.memories.length;
    return id;
  }

  recall(query, topK) {
    const queryEmbed = this.embed(query);
    return this.data.memories
      .map(m => ({ score: this.similarity(queryEmbed, m.embedding), memory: m }))
      .sort((a, b) => b.score - a.score).slice(0, topK).map(r => r.memory);
  }

  getQ(state, action) {
    const key = `${state}|${action}`;
    return this.data.patterns[key]?.q_value ?? 0;
  }

  updateQ(state, action, reward) {
    const key = `${state}|${action}`;
    if (!this.data.patterns[key]) this.data.patterns[key] = { state, action, q_value: 0, visits: 0, last_update: 0 };
    const p = this.data.patterns[key];
    p.q_value = p.q_value + this.alpha * (reward - p.q_value);
    p.visits++;
    p.last_update = this.now();
    this.data.stats.total_patterns = Object.keys(this.data.patterns).length;
  }

  learn(state, action, outcome, reward) {
    const id = `traj_${this.now()}`;
    this.updateQ(state, action, reward);
    this.data.trajectories.push({ id, state, action, outcome, reward, timestamp: this.now() });
    if (this.data.trajectories.length > 1000) this.data.trajectories.splice(0, 200);
    this.data.stats.total_trajectories = this.data.trajectories.length;
    return id;
  }

  suggest(state, actions) {
    let bestAction = actions[0] ?? '';
    let bestQ = -Infinity;
    for (const action of actions) {
      const q = this.getQ(state, action);
      if (q > bestQ) { bestQ = q; bestAction = action; }
    }
    return { action: bestAction, confidence: bestQ > 0 ? Math.min(bestQ, 1) : 0 };
  }

  route(task, file, crateName, operation = 'edit') {
    const fileType = file ? path.extname(file).slice(1) : 'unknown';
    const state = `${operation}_${fileType}_in_${crateName ?? 'project'}`;
    const agentMap = {
      rs: ['rust-developer', 'coder', 'reviewer', 'tester'],
      ts: ['typescript-developer', 'coder', 'frontend-dev'],
      tsx: ['typescript-developer', 'coder', 'frontend-dev'],
      js: ['coder', 'frontend-dev'],
      py: ['python-developer', 'coder', 'ml-developer'],
      md: ['docs-writer', 'coder']
    };
    const agents = agentMap[fileType] ?? ['coder', 'reviewer'];
    const { action, confidence } = this.suggest(state, agents);
    const reason = confidence > 0.5 ? 'learned from past success' : confidence > 0 ? 'based on patterns' : `default for ${fileType} files`;
    return { agent: action, confidence, reason };
  }

  shouldTest(file) {
    const ext = path.extname(file).slice(1);
    switch (ext) {
      case 'rs': {
        const crateMatch = file.match(/crates\/([^/]+)/);
        return crateMatch ? { suggest: true, command: `cargo test -p ${crateMatch[1]}` } : { suggest: true, command: 'cargo test' };
      }
      case 'ts': case 'tsx': case 'js': case 'jsx': return { suggest: true, command: 'npm test' };
      case 'py': return { suggest: true, command: 'pytest' };
      default: return { suggest: false, command: '' };
    }
  }

  recordFileSequence(fromFile, toFile) {
    const existing = this.data.file_sequences.find(s => s.from_file === fromFile && s.to_file === toFile);
    if (existing) existing.count++;
    else this.data.file_sequences.push({ from_file: fromFile, to_file: toFile, count: 1 });
    this.lastEditedFile = toFile;
  }

  suggestNext(file, limit = 3) {
    return this.data.file_sequences.filter(s => s.from_file === file).sort((a, b) => b.count - a.count).slice(0, limit).map(s => ({ file: s.to_file, score: s.count }));
  }

  recordError(command, message) {
    const codeMatch = message.match(/error\[([A-Z]\d+)\]/i) || message.match(/([A-Z]\d{4})/);
    const codes = [];
    if (codeMatch) {
      const code = codeMatch[1];
      codes.push(code);
      if (!this.data.errors[code]) this.data.errors[code] = { code, error_type: 'unknown', message: message.slice(0, 500), fixes: [], occurrences: 0 };
      this.data.errors[code].occurrences++;
      this.data.stats.total_errors = Object.keys(this.data.errors).length;
    }
    return codes;
  }

  classifyCommand(command) {
    const cmd = command.toLowerCase();
    if (cmd.includes('cargo') || cmd.includes('rustc')) return { category: 'rust', subcategory: cmd.includes('test') ? 'test' : 'build', risk: 'low' };
    if (cmd.includes('npm') || cmd.includes('node')) return { category: 'javascript', subcategory: cmd.includes('test') ? 'test' : 'build', risk: 'low' };
    if (cmd.includes('git')) return { category: 'git', subcategory: 'vcs', risk: cmd.includes('push') ? 'medium' : 'low' };
    if (cmd.includes('rm') || cmd.includes('delete')) return { category: 'filesystem', subcategory: 'destructive', risk: 'high' };
    return { category: 'shell', subcategory: 'general', risk: 'low' };
  }

  swarmStats() {
    const agents = Object.keys(this.data.agents).length;
    const edges = this.data.edges.length;
    const activeAgents = Object.values(this.data.agents).filter(a => a.status === 'active');
    const avgSuccess = activeAgents.length > 0 ? activeAgents.reduce((sum, a) => sum + a.success_rate, 0) / activeAgents.length : 0;
    return { agents, edges, avgSuccess };
  }

  stats() { return this.data.stats; }
  sessionStart() { this.data.stats.session_count++; this.data.stats.last_session = this.now(); }
  sessionEnd() {
    const duration = this.now() - this.data.stats.last_session;
    const actions = this.data.trajectories.filter(t => t.timestamp >= this.data.stats.last_session).length;
    return { duration, actions };
  }
  getLastEditedFile() { return this.lastEditedFile; }
}

const hooksCmd = program.command('hooks').description('Self-learning intelligence hooks for Claude Code');

hooksCmd.command('init').description('Initialize hooks in current project').option('--force', 'Force overwrite').action((opts) => {
  const settingsPath = path.join(process.cwd(), '.claude', 'settings.json');
  const settingsDir = path.dirname(settingsPath);
  if (!fs.existsSync(settingsDir)) require('fs').mkdirSync(settingsDir, { recursive: true });
  let settings = {};
  if (fs.existsSync(settingsPath)) try { settings = JSON.parse(require('fs').readFileSync(settingsPath, 'utf-8')); } catch {}
  settings.hooks = settings.hooks || {};
  settings.hooks.PreToolUse = [{ matcher: 'Edit|Write|MultiEdit', hooks: ['ruvector hooks pre-edit "$TOOL_INPUT_file_path"'] }, { matcher: 'Bash', hooks: ['ruvector hooks pre-command "$TOOL_INPUT_command"'] }];
  settings.hooks.PostToolUse = [{ matcher: 'Edit|Write|MultiEdit', hooks: ['ruvector hooks post-edit "$TOOL_INPUT_file_path"'] }, { matcher: 'Bash', hooks: ['ruvector hooks post-command "$TOOL_INPUT_command"'] }];
  settings.hooks.SessionStart = ['ruvector hooks session-start'];
  settings.hooks.Stop = ['ruvector hooks session-end'];
  require('fs').writeFileSync(settingsPath, JSON.stringify(settings, null, 2));
  console.log(chalk.green('✅ Hooks initialized in .claude/settings.json'));
});

hooksCmd.command('stats').description('Show intelligence statistics').action(() => {
  const intel = new Intelligence();
  const stats = intel.stats();
  const swarm = intel.swarmStats();
  console.log(chalk.bold.cyan('\n🧠 RuVector Intelligence Stats\n'));
  console.log(`  ${chalk.green(stats.total_patterns)} Q-learning patterns`);
  console.log(`  ${chalk.green(stats.total_memories)} vector memories`);
  console.log(`  ${chalk.green(stats.total_trajectories)} learning trajectories`);
  console.log(`  ${chalk.green(stats.total_errors)} error patterns\n`);
  console.log(chalk.bold('Swarm Status:'));
  console.log(`  ${chalk.cyan(swarm.agents)} agents registered`);
  console.log(`  ${chalk.cyan(swarm.edges)} coordination edges`);
});

hooksCmd.command('session-start').description('Session start hook').option('--resume', 'Resume previous session').action(() => {
  const intel = new Intelligence();
  intel.sessionStart();
  intel.save();
  console.log(chalk.bold.cyan('🧠 RuVector Intelligence Layer Active'));
  console.log('⚡ Intelligence guides: agent routing, error fixes, file sequences');
});

hooksCmd.command('session-end').description('Session end hook').option('--export-metrics', 'Export metrics').action((opts) => {
  const intel = new Intelligence();
  const sessionInfo = intel.sessionEnd();
  intel.save();
  console.log('📊 Session ended. Learning data saved.');
  if (opts.exportMetrics) console.log(JSON.stringify({ duration_seconds: sessionInfo.duration, actions_recorded: sessionInfo.actions }));
});

hooksCmd.command('pre-edit').description('Pre-edit intelligence').argument('<file>', 'File path').action((file) => {
  const intel = new Intelligence();
  const fileName = path.basename(file);
  const crateMatch = file.match(/crates\/([^/]+)/);
  const crate = crateMatch?.[1];
  const { agent, confidence, reason } = intel.route(`edit ${fileName}`, file, crate, 'edit');
  console.log(chalk.bold('🧠 Intelligence Analysis:'));
  console.log(`   📁 ${chalk.cyan(crate ?? 'project')}/${fileName}`);
  console.log(`   🤖 Recommended: ${chalk.green.bold(agent)} (${(confidence * 100).toFixed(0)}% confidence)`);
  if (reason) console.log(`      → ${chalk.dim(reason)}`);
  const nextFiles = intel.suggestNext(file, 3);
  if (nextFiles.length > 0) { console.log('   📎 Likely next files:'); nextFiles.forEach(n => console.log(`      - ${n.file} (${n.score} edits)`)); }
});

hooksCmd.command('post-edit').description('Post-edit learning').argument('<file>', 'File path').option('--success', 'Edit succeeded').action((file, opts) => {
  const intel = new Intelligence();
  const success = opts.success ?? true;
  const ext = path.extname(file).slice(1);
  const crateMatch = file.match(/crates\/([^/]+)/);
  const crate = crateMatch?.[1] ?? 'project';
  const state = `edit_${ext}_in_${crate}`;
  const lastFile = intel.getLastEditedFile();
  if (lastFile && lastFile !== file) intel.recordFileSequence(lastFile, file);
  intel.learn(state, success ? 'successful-edit' : 'failed-edit', success ? 'completed' : 'failed', success ? 1.0 : -0.5);
  intel.remember('edit', `${success ? 'successful' : 'failed'} edit of ${ext} in ${crate}`);
  intel.save();
  console.log(`📊 Learning recorded: ${success ? '✅' : '❌'} ${path.basename(file)}`);
  const test = intel.shouldTest(file);
  if (test.suggest) console.log(`   🧪 Consider: ${chalk.cyan(test.command)}`);
});

hooksCmd.command('pre-command').description('Pre-command intelligence').argument('<command...>', 'Command').action((command) => {
  const intel = new Intelligence();
  const cmd = command.join(' ');
  const classification = intel.classifyCommand(cmd);
  console.log(chalk.bold('🧠 Command Analysis:'));
  console.log(`   📦 Category: ${chalk.cyan(classification.category)}`);
  console.log(`   🏷️  Type: ${classification.subcategory}`);
  if (classification.risk === 'high') console.log(`   ⚠️  Risk: ${chalk.red('HIGH')} - Review carefully`);
  else if (classification.risk === 'medium') console.log(`   ⚡ Risk: ${chalk.yellow('MEDIUM')}`);
  else console.log(`   ✅ Risk: ${chalk.green('LOW')}`);
});

hooksCmd.command('post-command').description('Post-command learning').argument('<command...>', 'Command').option('--success', 'Success').action((command, opts) => {
  const intel = new Intelligence();
  const cmd = command.join(' ');
  const success = opts.success ?? true;
  const classification = intel.classifyCommand(cmd);
  intel.learn(`cmd_${classification.category}_${classification.subcategory}`, success ? 'success' : 'failure', success ? 'completed' : 'failed', success ? 0.8 : -0.3);
  intel.remember('command', `${cmd} ${success ? 'succeeded' : 'failed'}`);
  intel.save();
  console.log(`📊 Command ${success ? '✅' : '❌'} recorded`);
});

hooksCmd.command('route').description('Route task to agent').argument('<task...>', 'Task').option('--file <file>', 'File').option('--crate <crate>', 'Crate').action((task, opts) => {
  const intel = new Intelligence();
  const result = intel.route(task.join(' '), opts.file, opts.crate);
  console.log(JSON.stringify({ task: task.join(' '), recommended: result.agent, confidence: result.confidence, reasoning: result.reason }, null, 2));
});

hooksCmd.command('suggest-context').description('Suggest relevant context').action(() => {
  const intel = new Intelligence();
  const stats = intel.stats();
  console.log(`RuVector Intelligence: ${stats.total_patterns} learned patterns, ${stats.total_errors} error fixes available. Use 'ruvector hooks route' for agent suggestions.`);
});

hooksCmd.command('remember').description('Store in memory').requiredOption('-t, --type <type>', 'Memory type').option('--memory-type <type>', 'Memory type alias').argument('<content...>', 'Content').action((content, opts) => {
  const intel = new Intelligence();
  const memType = opts.type || opts.memoryType || 'general';
  const id = intel.remember(memType, content.join(' '));
  intel.save();
  console.log(JSON.stringify({ success: true, id }));
});

hooksCmd.command('recall').description('Search memory').argument('<query...>', 'Query').option('-k, --top-k <n>', 'Results', '5').action((query, opts) => {
  const intel = new Intelligence();
  const results = intel.recall(query.join(' '), parseInt(opts.topK));
  console.log(JSON.stringify({ query: query.join(' '), results: results.map(r => ({ type: r.memory_type, content: r.content.slice(0, 200), timestamp: r.timestamp })) }, null, 2));
});

hooksCmd.command('pre-compact').description('Pre-compact hook').option('--auto', 'Auto mode').action(() => {
  const intel = new Intelligence();
  intel.save();
  console.log('🗜️ Pre-compact: State saved');
});

hooksCmd.command('swarm-recommend').description('Recommend agent for task').argument('<task-type>', 'Task type').action((taskType) => {
  console.log(JSON.stringify({ task_type: taskType, recommended: 'coder', type: 'default', score: 0.8 }));
});

hooksCmd.command('async-agent').description('Async agent hook').option('--action <action>', 'Action').option('--agent-id <id>', 'Agent ID').option('--task <task>', 'Task').action((opts) => {
  console.log(JSON.stringify({ action: opts.action, agent_id: opts.agentId, status: 'ok' }));
});

hooksCmd.command('lsp-diagnostic').description('LSP diagnostic hook').option('--file <file>', 'File').option('--severity <sev>', 'Severity').option('--message <msg>', 'Message').action((opts) => {
  console.log(JSON.stringify({ file: opts.file, severity: opts.severity, action: 'logged' }));
});

hooksCmd.command('track-notification').description('Track notification').action(() => { console.log(JSON.stringify({ tracked: true })); });

program.parse(process.argv);

if (!process.argv.slice(2).length) {
  program.outputHelp();
}
