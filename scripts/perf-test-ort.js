/**
 * Run ORT GenAI perf test with result saving.
 * - For WebGPU (default): uses model_benchmark.exe (native C++ build)
 * - For CUDA/CPU: uses Python onnxruntime-genai package
 *
 * Usage:
 *   node scripts/perf-test-ort.js                                    # WebGPU, default model, sweep prompt lengths
 *   node scripts/perf-test-ort.js -m Qwen3-1.7B                     # Specific model
 *   node scripts/perf-test-ort.js --ep cuda                          # CUDA via Python
 *   node scripts/perf-test-ort.js -pl 256                            # Single prompt length
 *   node scripts/perf-test-ort.js --iterations 10 --warmup 3
 */

const { spawn, spawnSync, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { getSystemInfo } = require('./common');

const config = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'config.json'), 'utf8'));

const ORT_PATH = config.paths.onnxruntime;
const GENAI_PATH = config.paths['onnxruntime-genai'];
const BUILD_CONFIG = config.build.config;
const OS_DIR = process.platform === 'win32' ? 'Windows' : 'Linux';
const BIN_DIR = path.resolve(path.join(__dirname, '..', config.paths.bin));
const ORT_BACKUP_ROOT = config.paths['ort-backup'] || path.join(BIN_DIR, '..', 'ort-backup');
const MODEL_ROOT = config.paths.models;
const AI_MODEL_ROOT = config.paths['ai-models'] || MODEL_ROOT;
const RESULTS_DIR = config.paths.results || path.resolve(path.join(__dirname, '..', 'gitignore', 'results'));

// ORT install directory: <onnxruntime>/install/<config>
const ORT_INSTALL_DIR = path.join(ORT_PATH, 'install', BUILD_CONFIG);

const DEFAULT_PROMPT_LENGTHS = [128, 256, 512, 1024, 2048, 4096];

// ============================================================
// Copy binaries to dated backup directory
// ============================================================

function getBinarySources() {
  return {
    // From ORT install
    'onnxruntime.dll': path.join(ORT_INSTALL_DIR, 'bin', 'onnxruntime.dll'),

    // Dawn DLLs from ORT build (not included in cmake install)
    'dxcompiler.dll': path.join(ORT_PATH, 'build', OS_DIR, BUILD_CONFIG, BUILD_CONFIG, 'dxcompiler.dll'),
    'dxil.dll': path.join(ORT_PATH, 'build', OS_DIR, BUILD_CONFIG, BUILD_CONFIG, 'dxil.dll'),

    // From GenAI build
    'model_benchmark.exe': path.join(GENAI_PATH, 'build', OS_DIR, BUILD_CONFIG, 'benchmark', 'c', BUILD_CONFIG, 'model_benchmark.exe'),
    'onnxruntime-genai.dll': path.join(GENAI_PATH, 'build', OS_DIR, BUILD_CONFIG, BUILD_CONFIG, 'onnxruntime-genai.dll'),
  };
}

/**
 * Get the date string (yyyymmdd) from model_benchmark.exe's modification time.
 */
function getBenchmarkDate() {
  const sources = getBinarySources();
  const exeSrc = sources['model_benchmark.exe'];

  // Try source build output first
  if (fs.existsSync(exeSrc)) {
    const mtime = fs.statSync(exeSrc).mtime;
    return formatDate(mtime);
  }

  // Fallback: check existing BIN_DIR
  const localExe = path.join(BIN_DIR, 'model_benchmark.exe');
  if (fs.existsSync(localExe)) {
    const mtime = fs.statSync(localExe).mtime;
    return formatDate(mtime);
  }

  return null;
}

function formatDate(d) {
  return [
    d.getFullYear(),
    String(d.getMonth() + 1).padStart(2, '0'),
    String(d.getDate()).padStart(2, '0'),
  ].join('');
}

/**
 * Copy binaries to E:\...\backup\ort\<date> if the dated dir doesn't exist.
 */
function copyBinaries() {
  const dateStr = getBenchmarkDate();
  if (!dateStr) {
    console.log('No build outputs found to copy. Skipping backup.\n');
    return;
  }

  const backupDir = path.join(ORT_BACKUP_ROOT, dateStr);

  // If dated directory already has model_benchmark.exe, skip copying
  if (fs.existsSync(path.join(backupDir, 'model_benchmark.exe'))) {
    console.log(`Binaries already backed up at: ${backupDir}\n`);
    return;
  }

  const sources = getBinarySources();
  fs.mkdirSync(backupDir, { recursive: true });
  console.log(`Copying binaries to: ${backupDir}\n`);

  let success = 0;
  let failed = 0;

  for (const [name, src] of Object.entries(sources)) {
    const dest = path.join(backupDir, name);
    if (fs.existsSync(src)) {
      fs.copyFileSync(src, dest);
      const size = (fs.statSync(dest).size / 1024).toFixed(0);
      console.log(`  [COPIED] ${name} (${size} KB)`);
      success++;
    } else {
      // Try from local BIN_DIR as fallback
      const localSrc = path.join(BIN_DIR, name);
      if (fs.existsSync(localSrc)) {
        fs.copyFileSync(localSrc, dest);
        const size = (fs.statSync(dest).size / 1024).toFixed(0);
        console.log(`  [COPIED] ${name} from local bin (${size} KB)`);
        success++;
      } else {
        console.error(`  [MISSING] ${name} - expected at: ${src}`);
        failed++;
      }
    }
  }

  console.log(`\nDone: ${success} copied, ${failed} missing.`);
  if (failed > 0) {
    console.error('Some binaries are missing. Run "node scripts/build-ort.js all" first.');
    process.exit(1);
  }
}

/**
 * List available ORT backup versions (yyyymmdd dirs), newest first.
 */
function getAvailableVersions() {
  if (!fs.existsSync(ORT_BACKUP_ROOT)) return [];
  return fs.readdirSync(ORT_BACKUP_ROOT, { withFileTypes: true })
    .filter(d => d.isDirectory() && /^\d{8}$/.test(d.name)
      && fs.existsSync(path.join(ORT_BACKUP_ROOT, d.name, 'model_benchmark.exe')))
    .map(d => d.name)
    .sort((a, b) => b.localeCompare(a));  // newest first
}

/**
 * Resolve the bin directory for running benchmarks.
 * Uses the specified version or the latest available backup.
 */
function resolveBinDir(requestedVersion) {
  if (requestedVersion) {
    const dir = path.join(ORT_BACKUP_ROOT, requestedVersion);
    if (!fs.existsSync(path.join(dir, 'model_benchmark.exe'))) {
      const available = getAvailableVersions();
      console.error(`ORT version ${requestedVersion} not found.`);
      console.error(`Available: ${available.length > 0 ? available.join(', ') : '(none)'}`);
      process.exit(1);
    }
    return dir;
  }

  const versions = getAvailableVersions();
  if (versions.length === 0) {
    console.error(`No ORT backups found in ${ORT_BACKUP_ROOT}.`);
    console.error('Run "node scripts/build-ort.js all" first, then run this script to copy binaries.');
    process.exit(1);
  }

  return path.join(ORT_BACKUP_ROOT, versions[0]);
}

// ============================================================
// Parse CLI arguments
// ============================================================

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    models: [Object.keys(config.models)[0]],
    promptLengths: null,       // null = use DEFAULT_PROMPT_LENGTHS
    genLength: config.perf.genTokens,
    prompt: null,
    iterations: config.perf.iterations,
    warmup: config.perf.warmup,
    verbose: false,
    maxLength: 0,
    ep: null,  // execution provider: cuda, cpu (default: WebGPU via native build)
    ortVersion: null,  // ORT backup version (yyyymmdd), null = latest
    graphCapture: null, // null = don't change, true = enable, false = disable
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--model':
      case '-m': {
        const val = args[++i];
        const newModels = val.split(',').map(s => s.trim()).filter(Boolean);
        if (opts._modelOverridden) {
          opts.models.push(...newModels);
        } else {
          opts.models = newModels;
          opts._modelOverridden = true;
        }
        break;
      }
      case '--prompt-length':
      case '-pl':
        opts.promptLengths = [parseInt(args[++i])];
        break;
      case '--gen-length':
      case '-gl':
        opts.genLength = parseInt(args[++i]);
        break;
      case '--prompt':
        opts.prompt = args[++i];
        break;
      case '--iterations':
      case '-r':
        opts.iterations = parseInt(args[++i]);
        break;
      case '--warmup':
      case '-w':
        opts.warmup = parseInt(args[++i]);
        break;
      case '--max-length':
      case '-ml':
        opts.maxLength = parseInt(args[++i]);
        break;
      case '--ep':
      case '-e':
        opts.ep = args[++i];
        break;
      case '--ort-version':
      case '-V':
        opts.ortVersion = args[++i];
        break;
      case '--gc':
        opts.graphCapture = true;
        break;
      case '--no-gc':
        opts.graphCapture = false;
        break;
      case '--verbose':
      case '-v':
        opts.verbose = true;
        break;
      case '--help':
      case '-h':
        printHelp();
        process.exit(0);
      default:
        console.error(`Unknown option: ${args[i]}`);
        printHelp();
        process.exit(1);
    }
  }

  if (!opts.promptLengths) {
    opts.promptLengths = DEFAULT_PROMPT_LENGTHS;
  }

  return opts;
}

function printHelp() {
  console.log(`
Usage: node scripts/perf-test-ort.js [options]

Options:
  -m, --model <name>          Model name(s), comma-separated (default: ${Object.keys(config.models)[0]})
  -e, --ep <provider>         Execution provider: cuda, cpu (default: WebGPU via native build)
  -V, --ort-version <date>    ORT backup version to use, yyyymmdd (default: latest)
  --gc                        Enable graph capture in genai_config.json
  --no-gc                     Disable graph capture in genai_config.json
  -pl, --prompt-length <n>    Single prompt length (default: sweep ${DEFAULT_PROMPT_LENGTHS.join(',')})
  -gl, --gen-length <n>       Number of tokens to generate (default: ${config.perf.genTokens})
      --prompt <text>         Use specific prompt text instead of generated prompt
  -r, --iterations <n>        Number of benchmark iterations (default: ${config.perf.iterations})
  -w, --warmup <n>            Number of warmup iterations (default: ${config.perf.warmup})
  -ml, --max-length <n>       Max sequence length (0 = auto)
  -v, --verbose               Verbose output
  -h, --help                  Show this help
`);
}

// ============================================================
// Find model path
// ============================================================

/**
 * Resolve model name to a directory containing genai_config.json.
 * Searches: direct path, MODEL_ROOT/<name>, MODEL_ROOT/<name>/onnx, config.models entries.
 */
function findModelPath(modelName) {
  // Direct absolute/relative path
  if (fs.existsSync(path.join(modelName, 'genai_config.json'))) {
    return path.resolve(modelName);
  }
  // Direct path with /onnx or /webgpu subdirectory
  for (const sub of ['onnx', 'webgpu']) {
    const subDir = path.join(modelName, sub);
    if (fs.existsSync(path.join(subDir, 'genai_config.json'))) {
      return path.resolve(subDir);
    }
  }

  // Search in MODEL_ROOT and AI_MODEL_ROOT
  const roots = [MODEL_ROOT, AI_MODEL_ROOT].filter((v, i, a) => a.indexOf(v) === i);
  for (const root of roots) {
    const modelDir = path.join(root, modelName);
    if (fs.existsSync(path.join(modelDir, 'genai_config.json'))) {
      return modelDir;
    }
    for (const sub of ['onnx', 'webgpu']) {
      const subDir = path.join(modelDir, sub);
      if (fs.existsSync(path.join(subDir, 'genai_config.json'))) {
        return subDir;
      }
    }
  }

  // Check config.models
  const modelInfo = config.models[modelName];
  if (modelInfo) {
    const configPath = path.join(MODEL_ROOT, modelInfo.path);
    if (fs.existsSync(path.join(configPath, 'genai_config.json'))) {
      return configPath;
    }
    for (const sub of ['onnx', 'webgpu']) {
      const subDir = path.join(configPath, sub);
      if (fs.existsSync(path.join(subDir, 'genai_config.json'))) {
        return subDir;
      }
    }
  }

  return null;
}

// ============================================================
// Parse benchmark output
// ============================================================

function parseBenchmarkOutput(stdout, promptLength) {
  const result = {};

  // Prompt processing (time to first token)
  const ppAvgTs = stdout.match(/Prompt processing[\s\S]*?avg \(tokens\/s\):\s+([\d.]+)/);
  const ppStddev = stdout.match(/Prompt processing[\s\S]*?stddev \(us\):\s+([\d.]+)/);

  // Token generation
  const tgAvgUs = stdout.match(/Token generation[\s\S]*?avg \(us\):\s+([\d.]+)/);
  const tgAvgTs = stdout.match(/Token generation[\s\S]*?avg \(tokens\/s\):\s+([\d.]+)/);
  const tgStddev = stdout.match(/Token generation[\s\S]*?stddev \(us\):\s+([\d.]+)/);

  // E2E
  const e2eAvgMs = stdout.match(/E2E generation[\s\S]*?avg \(ms\):\s+([\d.]+)/);

  // Peak memory
  const peakMem = stdout.match(/Peak working set size \(bytes\):\s+(\d+)/);

  if (ppAvgTs) {
    result.plTs = parseFloat(ppAvgTs[1]);
    // Derive TTFT from tokens/s: TTFT = promptLength / plTs * 1000 (ms)
    result.ttftMs = promptLength / result.plTs * 1000;
  }
  if (ppStddev) result.ppStddevUs = parseFloat(ppStddev[1]);
  if (tgAvgUs) result.tgAvgUs = parseFloat(tgAvgUs[1]);
  if (tgAvgTs) result.tgTs = parseFloat(tgAvgTs[1]);
  if (tgStddev) result.tgStddevUs = parseFloat(tgStddev[1]);
  if (e2eAvgMs) result.e2eMs = parseFloat(e2eAvgMs[1]);
  if (peakMem) result.peakMemoryBytes = parseInt(peakMem[1]);

  return result;
}

// ============================================================
// Run a single benchmark
// ============================================================

function runBenchOnce(binDir, modelPath, pl, genLength, iterations, warmup, maxLength, ep, verbose) {
  const benchExe = path.join(binDir, 'model_benchmark.exe');
  const benchArgs = [
    '-i', modelPath,
    '-l', pl.toString(),
    '-g', genLength.toString(),
    '-r', iterations.toString(),
    '-w', warmup.toString(),
  ];

  if (ep) {
    benchArgs.push('-e', ep);
  }

  if (maxLength > 0) {
    benchArgs.push('-ml', maxLength.toString());
  }

  if (verbose) {
    benchArgs.push('-v');
  }

  const result = spawnSync(benchExe, benchArgs, {
    cwd: binDir,
    encoding: 'utf8',
    timeout: 600000,
    env: { ...process.env, PATH: `${binDir};${process.env.PATH}` },
  });

  if (result.status !== 0) {
    return { error: result.stderr || result.stdout || `Exit code ${result.status}` };
  }

  return { data: parseBenchmarkOutput(result.stdout, pl), raw: result.stdout };
}

// ============================================================
// Main
// ============================================================

function main() {
  const opts = parseArgs();

  // Copy new build outputs to backup (independent of benchmark run)
  let binDir = BIN_DIR;
  if (!opts.ep || opts.ep === 'webgpu' || opts.ep === 'dml') {
    copyBinaries();
    binDir = resolveBinDir(opts.ortVersion);
    console.log(`Using ORT binaries from: ${binDir}\n`);
  }

  // Resolve all model paths
  const modelEntries = [];
  for (const modelName of opts.models) {
    const modelPath = findModelPath(modelName);
    if (!modelPath) {
      console.error(`Model not found: ${modelName}`);
      console.error(`Searched: ${modelName}, ${MODEL_ROOT}/${modelName}, ${MODEL_ROOT}/${modelName}/onnx, config.models`);
      process.exit(1);
    }
    // Read and optionally set enableGraphCapture in genai_config.json
    let graphCapture = null;
    let originalGraphCapture = null;
    const genaiConfigPath = path.join(modelPath, 'genai_config.json');
    if (fs.existsSync(genaiConfigPath)) {
      try {
        const genaiConfig = JSON.parse(fs.readFileSync(genaiConfigPath, 'utf8'));
        const providerOpts = genaiConfig.model?.decoder?.session_options?.provider_options;
        if (Array.isArray(providerOpts)) {
          for (const po of providerOpts) {
            const provider = po?.webgpu || po?.dml;
            if (provider && 'enableGraphCapture' in provider) {
              const current = provider.enableGraphCapture === '1' || provider.enableGraphCapture === 1;
              originalGraphCapture = current;
              if (opts.graphCapture != null && opts.graphCapture !== current) {
                provider.enableGraphCapture = opts.graphCapture ? '1' : '0';
                fs.writeFileSync(genaiConfigPath, JSON.stringify(genaiConfig, null, 4));
                console.log(`  Updated enableGraphCapture to ${opts.graphCapture ? '1' : '0'} in ${genaiConfigPath}`);
              }
              graphCapture = opts.graphCapture != null ? opts.graphCapture : current;
              break;
            }
          }
        }
      } catch {}
    }
    modelEntries.push({ name: modelName, path: modelPath, graphCapture, originalGraphCapture, genaiConfigPath });
  }

  // For CUDA/CPU with Python runner, delegate (single prompt length only)
  if (opts.ep === 'cuda' || opts.ep === 'cpu') {
    for (const { name, path: modelPath } of modelEntries) {
      runPython({ ...opts, model: name }, modelPath);
    }
    return;
  }

  // Native WebGPU benchmark with sweep
  const sysInfo = getSystemInfo();
  const epName = opts.ep || 'webgpu';

  // Create timestamped result folder
  const now = new Date();
  const timestamp = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, '0'),
    String(now.getDate()).padStart(2, '0'),
    String(now.getHours()).padStart(2, '0'),
    String(now.getMinutes()).padStart(2, '0'),
    String(now.getSeconds()).padStart(2, '0'),
  ].join('');
  const resultDir = path.join(RESULTS_DIR, timestamp);
  fs.mkdirSync(resultDir, { recursive: true });

  console.log(`${'='.repeat(60)}`);
  console.log(`ORT GenAI Benchmark`);
  console.log(`Models:      ${opts.models.join(', ')}`);
  for (const me of modelEntries) {
    const gcLabel = me.graphCapture != null ? (me.graphCapture ? 'enabled' : 'disabled') : 'unknown';
    console.log(`  ${me.name}: ${me.path} (Graph Capture: ${gcLabel})`);
  }
  console.log(`EP:          ${epName}`);
  console.log(`Prompt lens: ${opts.promptLengths.join(', ')}`);
  console.log(`Gen length:  ${opts.genLength}`);
  console.log(`Iterations:  ${opts.iterations} (warmup: ${opts.warmup})`);
  console.log(`GPU:         ${sysInfo.gpu}`);
  console.log(`Results:     ${resultDir}`);
  console.log(`${'='.repeat(60)}\n`);

  const allResults = {
    system: sysInfo,
    config: {
      ep: epName,
      genLength: opts.genLength,
      runs: opts.iterations,
      warmup: opts.warmup,
      promptLengths: opts.promptLengths,
      graphCapture: modelEntries.length === 1 ? modelEntries[0].graphCapture : undefined,
    },
    results: [],
  };

  for (const { name: modelName, path: modelPath, graphCapture } of modelEntries) {
    console.log(`\n=== Model: ${modelName} ===\n`);

    for (const pl of opts.promptLengths) {
      process.stdout.write(`  prompt_length=${pl}, gl=${opts.genLength} ... `);

      const result = runBenchOnce(binDir, modelPath, pl, opts.genLength, opts.iterations, opts.warmup, opts.maxLength, opts.ep, opts.verbose);

      if (result.error) {
        console.log(`ERROR: ${result.error.split('\n')[0]}`);
        allResults.results.push({ model: modelName, ep: epName, pl, tg: opts.genLength, graphCapture, error: result.error });
      } else if (result.data) {
        const d = result.data;
        const record = {
          model: modelName,
          ep: epName,
          pl,
          tg: opts.genLength,
          graphCapture,
          ttftMs: d.ttftMs,
          plTs: d.plTs,
          tgTs: d.tgTs,
          tgStddevUs: d.tgStddevUs,
          e2eMs: d.e2eMs,
          peakMemoryBytes: d.peakMemoryBytes,
        };
        allResults.results.push(record);

        const parts = [];
        if (d.ttftMs != null) parts.push(`TTFT: ${d.ttftMs.toFixed(1)} ms`);
        if (d.tgTs != null) parts.push(`TPS: ${d.tgTs.toFixed(1)} t/s`);
        if (d.e2eMs != null) parts.push(`E2E: ${d.e2eMs.toFixed(0)} ms`);
        console.log(parts.join('  |  '));
      }
    }
  }

  // Write JSON results
  const resultFile = path.join(resultDir, 'ort-results.json');
  fs.writeFileSync(resultFile, JSON.stringify(allResults, null, 2));

  // Write human-readable summary
  const summaryLines = [
    `ORT GenAI Benchmark Results`,
    `${'='.repeat(60)}`,
    ``,
    `System Information`,
    `  CPU:        ${sysInfo.cpu}`,
    `  CPU Cores:  ${sysInfo.cpuCores}`,
    `  GPU:        ${sysInfo.gpu}`,
    `  GPU Driver: ${sysInfo.gpuDriver}`,
    `  GPU Memory: ${sysInfo.gpuMemoryMB || 'N/A'} MB`,
    `  OS:         ${sysInfo.os}`,
    `  RAM:        ${sysInfo.totalMemoryGB} GB`,
    `  Timestamp:  ${sysInfo.timestamp}`,
    ``,
    `Test Configuration`,
    `  EP:         ${epName}`,
    `  Gen Length: ${opts.genLength}`,
    `  Runs:       ${opts.iterations}`,
    `  Warmup:     ${opts.warmup}`,
    ``,
    `Results`,
    `${'='.repeat(92)}`,
    `${'Model'.padEnd(22)} ${'EP'.padEnd(10)} ${'PL'.padEnd(6)} ${'TTFT (ms)'.padEnd(12)} ${'TPS (t/s)'.padEnd(12)} ${'PL (t/s)'.padEnd(12)} ${'E2E (ms)'.padEnd(10)}`,
    `${'-'.repeat(92)}`,
  ];

  for (const r of allResults.results) {
    if (r.error) {
      summaryLines.push(`${(r.model || '').padEnd(22)} ${(r.ep || '').padEnd(10)} ${String(r.pl).padEnd(6)} ERROR: ${r.error.split('\n')[0]}`);
    } else if (r.ttftMs !== undefined) {
      summaryLines.push(
        `${(r.model || '').padEnd(22)} ${(r.ep || '').padEnd(10)} ${String(r.pl).padEnd(6)} ${String(r.ttftMs?.toFixed(2) || '').padEnd(12)} ${String(r.tgTs?.toFixed(2) || '').padEnd(12)} ${String(r.plTs?.toFixed(2) || '').padEnd(12)} ${String(r.e2eMs?.toFixed(0) || '').padEnd(10)}`
      );
    }
  }

  const summaryFile = path.join(resultDir, 'ort-results.txt');
  fs.writeFileSync(summaryFile, summaryLines.join('\n'));

  console.log(`\nResults saved to:`);
  console.log(`  JSON: ${resultFile}`);
  console.log(`  Text: ${summaryFile}`);

  // Restore enableGraphCapture to original value if changed
  for (const me of modelEntries) {
    if (opts.graphCapture != null && me.originalGraphCapture != null && opts.graphCapture !== me.originalGraphCapture) {
      try {
        const genaiConfig = JSON.parse(fs.readFileSync(me.genaiConfigPath, 'utf8'));
        const providerOpts = genaiConfig.model?.decoder?.session_options?.provider_options;
        if (Array.isArray(providerOpts)) {
          for (const po of providerOpts) {
            const provider = po?.webgpu || po?.dml;
            if (provider && 'enableGraphCapture' in provider) {
              provider.enableGraphCapture = me.originalGraphCapture ? '1' : '0';
              fs.writeFileSync(me.genaiConfigPath, JSON.stringify(genaiConfig, null, 4));
              console.log(`Restored enableGraphCapture to ${me.originalGraphCapture ? '1' : '0'} in ${me.genaiConfigPath}`);
              break;
            }
          }
        }
      } catch {}
    }
  }
}

/**
 * Run via Python onnxruntime-genai package (CUDA / CPU EP)
 * Requires: pip install onnxruntime-genai-cuda onnxruntime-gpu
 */
function runPython(opts, modelPath) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Model:       ${opts.model}`);
  console.log(`EP:          ${opts.ep}`);
  console.log(`Runner:      Python onnxruntime-genai`);
  console.log(`Model path:  ${modelPath}`);
  console.log(`Gen tokens:  ${opts.genLength}`);
  console.log(`Iterations:  ${opts.iterations} (warmup: ${opts.warmup})`);
  console.log(`${'='.repeat(60)}\n`);

  const pyArgs = [
    path.join(__dirname, 'perf-test-ort.py'),
    '-m', opts.model,
    '-e', opts.ep,
    '-g', opts.genLength.toString(),
    '-r', opts.iterations.toString(),
    '-w', opts.warmup.toString(),
    '-l', (opts.promptLengths[0] || 16).toString(),
  ];

  if (opts.verbose) {
    pyArgs.push('-v');
  }

  const child = spawn('python', pyArgs, {
    cwd: path.join(__dirname, '..'),
    stdio: 'inherit',
  });

  child.on('close', (code) => {
    if (code !== 0) {
      console.error(`\nPython perf test exited with code ${code}`);
      process.exit(code);
    }
  });

  child.on('error', (err) => {
    console.error(`Failed to start Python: ${err.message}`);
    process.exit(1);
  });
}

main();
