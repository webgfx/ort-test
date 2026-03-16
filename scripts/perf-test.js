/**
 * Unified perf test runner for ORT GenAI and llama.cpp benchmarks.
 *
 * Usage:
 *   node scripts/perf-test.js                                                   # Run both ORT and llama.cpp with defaults
 *   node scripts/perf-test.js --runtime ort                                       # ORT only
 *   node scripts/perf-test.js --runtime llamacpp                                  # llama.cpp only
 *   node scripts/perf-test.js --runtime ort,llamacpp                              # Both explicitly
 *   node scripts/perf-test.js --runtime ort --ort-backend webgpu                  # ORT WebGPU
 *   node scripts/perf-test.js --runtime llamacpp --llamacpp-backend cuda,vulkan    # llama.cpp CUDA + Vulkan
 *
 * Pass --runtime <name> --help to see runtime-specific options.
 */

// Extract --runtime from process.argv before modules parse it
const runtimeIdx = process.argv.indexOf('--runtime');
let runtimes;
if (runtimeIdx !== -1 && runtimeIdx + 1 < process.argv.length) {
  const val = process.argv[runtimeIdx + 1];
  process.argv.splice(runtimeIdx, 2);
  // Normalize each runtime name
  const runtimeAliases = { 'llama': 'llamacpp', 'llama.cpp': 'llamacpp' };
  runtimes = val.split(',').map(s => s.trim()).map(s => runtimeAliases[s] || s).filter(Boolean);
} else {
  runtimes = ['llamacpp', 'ort'];
}

const validRuntimes = ['ort', 'llamacpp'];
for (const r of runtimes) {
  if (!validRuntimes.includes(r)) {
    console.error(`Unknown runtime: ${r}`);
    console.error('Valid runtimes: ort, llamacpp');
    process.exit(1);
  }
}

// Handle --help for unified mode
if (runtimes.length > 1 && (process.argv.includes('--help') || process.argv.includes('-h'))) {
  console.log(`
Unified Perf Test Runner
========================

Usage: node scripts/perf-test.js [--runtime <runtimes>] [options]

Runtimes (comma-separated):
  ort        ORT GenAI benchmark
  llamacpp   llama.cpp benchmark
  (default: both)

Runtime-Specific Options:
  --ort-backend <ep>           ORT execution provider: webgpu, cuda, cpu (default: webgpu)
  --llamacpp-backend <list>    llama.cpp backend(s): cuda, vulkan, comma-separated (default: vulkan)

Shared Options:
  -m, --model <name>         Model name(s), comma-separated
  -pl, --prompt-length <n>   Prompt length
  -gl, --gen-length <n>      Generation length
  -r                         Repetitions / iterations
  -h, --help                 Show this help

Use --runtime <name> --help to see all options for a specific runtime.
`);
  process.exit(0);
}

// Require modules (their module-level code is safe to run without side effects)
const path = require('path');
const fs = require('fs');
const { main: runLlamaCpp } = require('./perf-test-llamacpp');
const { main: runOrt } = require('./perf-test-ort');
const { getSystemInfo } = require('./common');

const config = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'config.json'), 'utf8'));
const RESULTS_DIR = config.paths.results || path.resolve(path.join(__dirname, '..', 'gitignore', 'results'));

async function main() {
  const runLlama = runtimes.includes('llamacpp');
  const runOrtFlag = runtimes.includes('ort');
  const isMulti = runtimes.length > 1;
  const options = isMulti ? { lenient: true } : {};

  // Create shared result directory when running multiple runtimes
  let resultDir;
  if (isMulti) {
    const now = new Date();
    const timestamp = [
      now.getFullYear(),
      String(now.getMonth() + 1).padStart(2, '0'),
      String(now.getDate()).padStart(2, '0'),
      String(now.getHours()).padStart(2, '0'),
      String(now.getMinutes()).padStart(2, '0'),
      String(now.getSeconds()).padStart(2, '0'),
    ].join('');
    resultDir = path.join(RESULTS_DIR, timestamp);
    fs.mkdirSync(resultDir, { recursive: true });
    options.resultDir = resultDir;
  }

  const collected = {};

  if (runLlama) {
    if (isMulti) console.log(`\n${'#'.repeat(60)}\n# llama.cpp Benchmark\n${'#'.repeat(60)}\n`);
    try {
      collected.llamacpp = await runLlamaCpp(options);
    } catch (err) {
      if (isMulti) {
        console.error(`\nllama.cpp benchmark failed: ${err.message}\n`);
      } else {
        throw err;
      }
    }
  }

  if (runOrtFlag) {
    if (isMulti) console.log(`\n${'#'.repeat(60)}\n# ORT GenAI Benchmark\n${'#'.repeat(60)}\n`);
    try {
      collected.ort = runOrt(options);
    } catch (err) {
      if (isMulti) {
        console.error(`\nORT benchmark failed: ${err.message}\n`);
      } else {
        throw err;
      }
    }
  }

  // Write unified result file when running multiple runtimes
  if (isMulti && resultDir) {
    const sysInfo = getSystemInfo();
    const unified = {
      system: sysInfo,
      runtimes: {},
    };

    if (collected.llamacpp) {
      unified.runtimes.llamacpp = {
        config: collected.llamacpp.config,
        llamaCpp: collected.llamacpp.llamaCpp,
        results: collected.llamacpp.results,
      };
    }
    if (collected.ort) {
      unified.runtimes.ort = {
        config: collected.ort.config,
        results: collected.ort.results,
      };
    }

    const unifiedFile = path.join(resultDir, 'results.json');
    fs.writeFileSync(unifiedFile, JSON.stringify(unified, null, 2));
    console.log(`\nUnified results saved to: ${unifiedFile}`);
  }
}

main().catch(err => {
  console.error(`Error: ${err.message}`);
  process.exit(1);
});
