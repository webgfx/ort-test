/**
 * Run llama.cpp perf test using llama-bench, with result saving.
 *
 * Usage:
 *   node scripts/perf-test-llamacpp.js                              # CUDA + Vulkan, default prompt lengths
 *   node scripts/perf-test-llamacpp.js --backend cuda               # CUDA only
 *   node scripts/perf-test-llamacpp.js --backend vulkan             # Vulkan only
 *   node scripts/perf-test-llamacpp.js -pl 512 -gl 128              # Single prompt length
 *   node scripts/perf-test-llamacpp.js --model Qwen3-1.7B-Q8_0
 */

const { execSync, spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const https = require('https');
const { getSystemInfo } = require('./common');

const config = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'config.json'), 'utf8'));

const LLAMA_CPP_ROOT = config.paths['llama.cpp'] || 'E:\\workspace\\project\\test\\llama.cpp';
const MODEL_ROOT = config.paths.models;
const AI_MODEL_ROOT = config.paths['ai-models'] || MODEL_ROOT;
const RESULTS_DIR = config.paths.results || path.resolve(path.join(__dirname, '..', 'gitignore', 'results'));

const DEFAULT_PROMPT_LENGTHS = [128, 256, 512, 1024, 2048, 4096];

// ============================================================
// llama.cpp version resolution
// ============================================================

/**
 * Directory layout: LLAMA_CPP_ROOT/<version>/<backend>/
 *   e.g. llama.cpp/b8200/cuda/llama-bench.exe
 *        llama.cpp/b8200/vulkan/llama-bench.exe
 */
function getAvailableVersions() {
  if (!fs.existsSync(LLAMA_CPP_ROOT)) return [];
  return fs.readdirSync(LLAMA_CPP_ROOT, { withFileTypes: true })
    .filter(d => d.isDirectory() && /^b\d+$/.test(d.name))
    .map(d => ({ name: d.name, number: parseInt(d.name.slice(1)) }))
    .sort((a, b) => b.number - a.number);  // newest first
}

function resolveVersion(requested) {
  const versions = getAvailableVersions();
  if (versions.length === 0) {
    throw new Error(`No llama.cpp versions found in ${LLAMA_CPP_ROOT}. Expected subdirs like b8200/.`);
  }

  if (requested) {
    // Allow "b8200" or "8200"
    const normalized = requested.startsWith('b') ? requested : `b${requested}`;
    const found = versions.find(v => v.name === normalized);
    if (!found) {
      throw new Error(`llama.cpp version ${normalized} not found. Available: ${versions.map(v => v.name).join(', ')}`);
    }
    return found.name;
  }

  // Default: latest (highest build number)
  return versions[0].name;
}

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    backend: null,             // null = run both cuda and vulkan
    models: ['Qwen3-1.7B-Q8_0'],  // supports multiple models
    promptLengths: null,       // null = use DEFAULT_PROMPT_LENGTHS
    gl: 128,
    reps: 5,
    ngl: 99,
    version: null,             // null = use latest local
    updateLlama: false,         // download latest before running
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--backend':
      case '-b':
        opts.backend = args[++i];
        break;
      case '--model':
      case '-m': {
        const val = args[++i];
        // Support comma-separated models or multiple -m flags
        const newModels = val.split(',').map(s => s.trim()).filter(Boolean);
        if (opts._modelOverridden) {
          opts.models.push(...newModels);
        } else {
          opts.models = newModels;
          opts._modelOverridden = true;
        }
        break;
      }
      case '-pl':
        opts.promptLengths = [parseInt(args[++i])];
        break;
      case '-gl':
        opts.gl = parseInt(args[++i]);
        break;
      case '--reps':
      case '-r':
        opts.reps = parseInt(args[++i]);
        break;
      case '--ngl':
        opts.ngl = parseInt(args[++i]);
        break;
      case '--version':
      case '-V':
        opts.version = args[++i];
        break;
      case '--update-llama':
        opts.updateLlama = true;
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
Usage: node scripts/perf-test-llamacpp.js [options]

Options:
  -b, --backend <name>       Backend: cuda, vulkan, or omit for both (default: both)
  -m, --model <name>         Model name(s), comma-separated or repeated (default: Qwen3-1.7B-Q8_0)
  -V, --version <ver>        llama.cpp version, e.g. b8200 (default: latest local)
  --update-llama             Download latest llama.cpp before running
  -pl <n>                    Single prompt length (default: sweep 128,512,1024,2048,4096)
  -gl <n>                    Generation length (default: 128)
  -r, --reps <n>             Repetitions (default: 5)
  --ngl <n>                  GPU layers (default: 99 = all)
  -h, --help                 Show this help
`);
}

// ============================================================
// Download llama.cpp releases from GitHub
// ============================================================

const GITHUB_API = 'https://api.github.com/repos/ggerganov/llama.cpp/releases';
const CUDA_PATTERN = /llama-b\d+-bin-win-cuda-([\d.]+)-x64\.zip$/;
const VULKAN_PATTERN = /llama-b\d+-bin-win-vulkan-x64\.zip$/;

function getSystemCudaVersion() {
  try {
    const out = execSync('nvcc --version', { encoding: 'utf8', timeout: 5000 });
    const m = out.match(/release ([\d.]+)/);
    return m ? m[1] : null;
  } catch { return null; }
}

function pickCudaAsset(assets) {
  const cudaAssets = assets.filter(a => CUDA_PATTERN.test(a.name))
    .map(a => {
      const m = a.name.match(CUDA_PATTERN);
      return { asset: a, cudaVer: m[1], major: parseInt(m[1]) };
    })
    .sort((a, b) => b.major - a.major || b.cudaVer.localeCompare(a.cudaVer));

  if (cudaAssets.length === 0) return null;

  const sysCuda = getSystemCudaVersion();
  if (sysCuda) {
    const sysMajor = parseInt(sysCuda);
    const match = cudaAssets.find(c => c.major <= sysMajor);
    if (match) return match.asset;
  }
  return cudaAssets[0].asset;
}

function httpGet(url) {
  return new Promise((resolve, reject) => {
    const options = { headers: { 'User-Agent': 'ort-test-downloader' } };
    https.get(url, options, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        return httpGet(res.headers.location).then(resolve, reject);
      }
      if (res.statusCode !== 200) {
        return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
      }
      const chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => resolve(Buffer.concat(chunks)));
      res.on('error', reject);
    }).on('error', reject);
  });
}

async function httpGetJson(url) {
  const data = await httpGet(url);
  return JSON.parse(data.toString());
}

function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const options = { headers: { 'User-Agent': 'ort-test-downloader' } };
    const file = fs.createWriteStream(dest);
    const request = (u) => {
      https.get(u, options, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          return request(res.headers.location);
        }
        if (res.statusCode !== 200) {
          file.close();
          fs.unlinkSync(dest);
          return reject(new Error(`HTTP ${res.statusCode}`));
        }
        const totalBytes = parseInt(res.headers['content-length'] || '0');
        let downloaded = 0;
        res.on('data', (chunk) => {
          downloaded += chunk.length;
          if (totalBytes > 0) {
            const pct = ((downloaded / totalBytes) * 100).toFixed(0);
            process.stdout.write(`\r  Downloading... ${pct}% (${(downloaded / 1024 / 1024).toFixed(1)} MB)`);
          }
        });
        res.pipe(file);
        file.on('finish', () => {
          file.close();
          console.log(`\r  Downloaded ${(downloaded / 1024 / 1024).toFixed(1)} MB`);
          resolve();
        });
      }).on('error', (err) => {
        file.close();
        fs.unlinkSync(dest);
        reject(err);
      });
    };
    request(url);
  });
}

function extractZip(zipPath, destDir) {
  fs.mkdirSync(destDir, { recursive: true });
  execSync(`powershell -Command "Expand-Archive -Path '${zipPath}' -DestinationPath '${destDir}' -Force"`, {
    stdio: 'inherit',
    timeout: 120000,
  });
}

async function downloadLlamaCpp(version) {
  let release;
  if (version) {
    const tag = version.startsWith('b') ? version : `b${version}`;
    console.log(`Fetching release ${tag}...`);
    try {
      release = await httpGetJson(`${GITHUB_API}/tags/${tag}`);
    } catch {
      console.error(`Release ${tag} not found.`);
      process.exit(1);
    }
  } else {
    console.log('Fetching latest release...');
    const releases = await httpGetJson(`${GITHUB_API}?per_page=1`);
    if (releases.length === 0) {
      console.error('No releases found.');
      process.exit(1);
    }
    release = releases[0];
  }

  const tag = release.tag_name;
  const date = release.published_at?.slice(0, 10) || 'unknown';
  console.log(`Release: ${tag} (${date})\n`);

  const versionDir = path.join(LLAMA_CPP_ROOT, tag);

  const assets = {};
  const cudaAsset = pickCudaAsset(release.assets);
  if (cudaAsset) assets.cuda = cudaAsset;
  const vulkanAsset = release.assets.find(a => VULKAN_PATTERN.test(a.name));
  if (vulkanAsset) assets.vulkan = vulkanAsset;

  if (Object.keys(assets).length === 0) {
    console.error('No matching Windows x64 binary assets found in this release.');
    process.exit(1);
  }

  const tmpDir = path.join(LLAMA_CPP_ROOT, '.tmp');
  fs.mkdirSync(tmpDir, { recursive: true });

  for (const [backend, asset] of Object.entries(assets)) {
    const destDir = path.join(versionDir, backend);
    if (fs.existsSync(destDir) && fs.readdirSync(destDir).length > 0) {
      console.log(`  ${backend}: already exists at ${destDir}, skipping.`);
      continue;
    }

    console.log(`  ${backend}: ${asset.name} (${(asset.size / 1024 / 1024).toFixed(1)} MB)`);
    const zipPath = path.join(tmpDir, asset.name);

    await downloadFile(asset.browser_download_url, zipPath);

    console.log('  Extracting...');
    const extractDir = path.join(tmpDir, `${tag}-${backend}`);
    extractZip(zipPath, extractDir);

    const extracted = fs.readdirSync(extractDir);
    let sourceDir = extractDir;
    if (extracted.length === 1 && fs.statSync(path.join(extractDir, extracted[0])).isDirectory()) {
      sourceDir = path.join(extractDir, extracted[0]);
    }

    fs.mkdirSync(destDir, { recursive: true });
    for (const file of fs.readdirSync(sourceDir)) {
      fs.renameSync(path.join(sourceDir, file), path.join(destDir, file));
    }

    fs.rmSync(extractDir, { recursive: true, force: true });
    fs.unlinkSync(zipPath);

    console.log(`  Installed to ${destDir}`);
  }

  try { fs.rmdirSync(tmpDir); } catch {}

  console.log(`\nDone! ${tag} installed at ${versionDir}`);
}

// ============================================================
// Find model
// ============================================================

function findModel(modelName) {
  const roots = [MODEL_ROOT, AI_MODEL_ROOT].filter((v, i, a) => a.indexOf(v) === i);
  const candidates = [];

  for (const root of roots) {
    candidates.push(
      path.join(root, `${modelName.split('-Q')[0]}-GGUF`, `${modelName}.gguf`),
      path.join(root, `${modelName}-GGUF`, `${modelName}.gguf`),
      path.join(root, modelName, `${modelName}.gguf`),
      path.join(root, `${modelName}.gguf`),
    );

    // Search subdirectories for matching GGUF files (case-insensitive)
    if (fs.existsSync(root)) {
      const dirs = fs.readdirSync(root, { withFileTypes: true }).filter(d => d.isDirectory());
      for (const dir of dirs) {
        const dirPath = path.join(root, dir.name);
        try {
          const files = fs.readdirSync(dirPath).filter(f => f.endsWith('.gguf'));
          const match = files.find(f => f === `${modelName}.gguf`) || files.find(f => f.includes(modelName));
          if (match) candidates.push(path.join(dirPath, match));
        } catch {}
      }
    }
  }

  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

// ============================================================
// Detect llama.cpp version
// ============================================================

function getLlamaCppVersion(backend, modelPath, versionDir) {
  const backendDir = path.join(versionDir, backend);
  const llamaBench = path.join(backendDir, 'llama-bench.exe');

  if (!fs.existsSync(llamaBench)) return null;

  // Run a minimal bench (1 rep, 1 token) just to get version fields from JSON
  const result = spawnSync(llamaBench, [
    '-m', modelPath, '-p', '1', '-n', '0', '-r', '1', '-ngl', '0', '-o', 'json',
  ], {
    cwd: backendDir,
    encoding: 'utf8',
    timeout: 60000,
    env: { ...process.env, PATH: `${backendDir};${process.env.PATH}` },
  });

  if (result.status !== 0) return null;

  try {
    const stdout = result.stdout.trim();
    let data;
    try { data = JSON.parse(stdout); } catch {}
    if (!data) {
      const idx = stdout.indexOf('[');
      if (idx >= 0) data = JSON.parse(stdout.slice(idx));
    }
    if (data && data.length > 0) {
      return {
        buildNumber: data[0].build_number,
        buildCommit: data[0].build_commit,
        version: `b${data[0].build_number} (${data[0].build_commit})`,
      };
    }
  } catch {}
  return null;
}

// ============================================================
// Run llama-bench and parse output
// ============================================================

function runBench(backend, modelPath, pp, tg, reps, ngl, versionDir) {
  const backendDir = path.join(versionDir, backend);
  const llamaBench = path.join(backendDir, 'llama-bench.exe');

  if (!fs.existsSync(llamaBench)) {
    return { error: `llama-bench not found at ${llamaBench}` };
  }

  const benchArgs = [
    '-m', modelPath,
    '-p', pp.toString(),
    '-n', tg.toString(),
    '-r', reps.toString(),
    '-ngl', ngl.toString(),
    '-o', 'json',
  ];

  const result = spawnSync(llamaBench, benchArgs, {
    cwd: backendDir,
    encoding: 'utf8',
    timeout: 600000,
    env: { ...process.env, PATH: `${backendDir};${process.env.PATH}` },
  });

  if (result.status !== 0) {
    return { error: result.stderr || `Exit code ${result.status}` };
  }

  // Parse JSON output — llama-bench -o json outputs a pretty-printed JSON array
  try {
    const stdout = result.stdout.trim();
    // Try parsing the whole stdout as JSON first
    try {
      return { data: JSON.parse(stdout) };
    } catch {}
    // Fallback: find the '[' and take everything from there
    const idx = stdout.indexOf('[');
    if (idx >= 0) {
      return { data: JSON.parse(stdout.slice(idx)) };
    }
    return { raw: stdout };
  } catch {
    return { raw: result.stdout };
  }
}

// ============================================================
// Main
// ============================================================

async function main() {
  const opts = parseArgs();
  const backends = opts.backend ? [opts.backend] : ['cuda', 'vulkan'];

  // Download latest if --update-llama is specified
  if (opts.updateLlama) {
    console.log('[update-llama] Downloading latest llama.cpp release...\n');
    try {
      await downloadLlamaCpp(opts.version);
    } catch (err) {
      console.error(`[update-llama] Download failed: ${err.message}`);
      process.exit(1);
    }
    console.log('');
  }

  // Resolve llama.cpp version
  const llamaCppVersion = resolveVersion(opts.version);
  const versionDir = path.join(LLAMA_CPP_ROOT, llamaCppVersion);

  // Filter backends to those that actually exist in this version
  const availableBackends = backends.filter(b => {
    const dir = path.join(versionDir, b);
    return fs.existsSync(path.join(dir, 'llama-bench.exe'));
  });
  if (availableBackends.length === 0) {
    console.error(`No backends found in ${versionDir}. Available subdirs: ${fs.existsSync(versionDir) ? fs.readdirSync(versionDir).join(', ') : '(dir missing)'}`);
    process.exit(1);
  }

  // Find models
  const modelEntries = [];
  for (const modelName of opts.models) {
    const modelPath = findModel(modelName);
    if (!modelPath) {
      console.error(`Model not found: ${modelName}`);
      process.exit(1);
    }
    modelEntries.push({ name: modelName, path: modelPath });
  }

  // System info
  const sysInfo = getSystemInfo();

  // Detect llama.cpp version per backend (use first model for version detection)
  const llamaCppVersions = {};
  for (const backend of availableBackends) {
    const ver = getLlamaCppVersion(backend, modelEntries[0].path, versionDir);
    if (ver) llamaCppVersions[backend] = ver;
  }
  // Use first available version as primary
  const primaryVersion = Object.values(llamaCppVersions)[0];

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
  console.log(`llama.cpp Benchmark`);
  console.log(`Models:      ${opts.models.join(', ')}`);
  for (const me of modelEntries) {
    console.log(`  ${me.name}: ${me.path}`);
  }
  console.log(`Backends:    ${availableBackends.join(', ')}`);
  console.log(`Prompt lens: ${opts.promptLengths.join(', ')}`);
  console.log(`Gen length:  ${opts.gl}`);
  console.log(`Reps:        ${opts.reps}`);
  console.log(`GPU:         ${sysInfo.gpu}`);
  console.log(`llama.cpp:   ${llamaCppVersion} ${primaryVersion ? `(${primaryVersion.buildCommit})` : ''}`);
  console.log(`Results:     ${resultDir}`);
  console.log(`${'='.repeat(60)}\n`);

  const allResults = {
    system: sysInfo,
    llamaCpp: {
      version: llamaCppVersion,
      versions: llamaCppVersions,
      buildNumber: primaryVersion?.buildNumber,
      buildCommit: primaryVersion?.buildCommit,
    },
    config: {
      models: opts.models,
      genLength: opts.gl,
      runs: opts.reps,
      warmup: 1,
      ngl: opts.ngl,
      promptLengths: opts.promptLengths,
      backends: availableBackends,
    },
    results: [],
  };

  for (const { name: modelName, path: modelPath } of modelEntries) {
    console.log(`\n=== Model: ${modelName} ===`);

    for (const backend of availableBackends) {
      console.log(`\n--- Backend: ${backend.toUpperCase()} ---\n`);

      for (const pl of opts.promptLengths) {
        process.stdout.write(`  prompt_length=${pl}, tg=${opts.gl} ... `);

        const result = runBench(backend, modelPath, pl, opts.gl, opts.reps, opts.ngl, versionDir);

        if (result.error) {
          console.log(`ERROR: ${result.error}`);
          allResults.results.push({ model: modelName, backend, pl, tg: opts.gl, error: result.error });
        } else if (result.data) {
          // Combine pp (prefill) and tg (generation) into one record per prompt length
          const ppEntry = result.data.find(e => e.n_prompt > 0);
          const tgEntry = result.data.find(e => e.n_gen > 0);

          // TTFT = prefill time in ms (avg_ns / 1e6)
          const ttftMs = ppEntry ? ppEntry.avg_ns / 1e6 : null;
          const plTs = ppEntry ? ppEntry.avg_ts : null;
          const tgTs = tgEntry ? tgEntry.avg_ts : null;

          const record = {
            model: modelName,
            backend,
            pl,
            tg: opts.gl,
            ttftMs,           // Time to first token (prefill time in ms)
            plTs,             // Prefill throughput (tokens/s)
            plStddevTs: ppEntry?.stddev_ts,
            tgTs,             // Generation throughput (tokens/s)
            tgStddevTs: tgEntry?.stddev_ts,
            modelType: (ppEntry || tgEntry)?.model_type,
            modelSize: (ppEntry || tgEntry)?.model_size,
            nParams: (ppEntry || tgEntry)?.model_n_params,
            nGpuLayers: (ppEntry || tgEntry)?.n_gpu_layers,
            ppSamplesTs: ppEntry?.samples_ts,
            tgSamplesTs: tgEntry?.samples_ts,
          };
          allResults.results.push(record);

          const parts = [];
          if (ttftMs != null) parts.push(`TTFT: ${ttftMs.toFixed(1)} ms`);
          if (tgTs != null) parts.push(`TPS: ${tgTs.toFixed(1)} t/s`);
          console.log(parts.join('  |  '));
        } else if (result.raw) {
          console.log('(raw output)');
          allResults.results.push({ model: modelName, backend, pl, tg: opts.gl, raw: result.raw });
        }
      }
    }
  }

  // Write results
  const resultFile = path.join(resultDir, 'llamacpp-results.json');
  fs.writeFileSync(resultFile, JSON.stringify(allResults, null, 2));

  // Also write a human-readable summary
  const summaryLines = [
    `llama.cpp Benchmark Results`,
    `${'='.repeat(60)}`,
    ``,
    `System Information`,
    `  CPU:        ${sysInfo.cpu}`,
    `  CPU Cores:  ${sysInfo.cpuCores}`,
    `  GPU:        ${sysInfo.gpu}`,
    `  GPU Driver: ${sysInfo.gpuDriver}`,
    `  GPU Memory: ${sysInfo.gpuMemoryMB} MB`,
    `  OS:         ${sysInfo.os}`,
    `  RAM:        ${sysInfo.totalMemoryGB} GB`,
    `  Timestamp:  ${sysInfo.timestamp}`,
    ``,
    `Test Configuration`,
    `  llama.cpp:  ${llamaCppVersion} ${primaryVersion ? `(${primaryVersion.buildCommit})` : ''}`,
    `  Gen Length: ${opts.gl}`,
    `  Runs:       ${opts.reps}`,
    `  Warmup:     1 (llama-bench default)`,
    `  GPU Layers: ${opts.ngl}`,
    ``,
    `Results`,
    `${'='.repeat(70)}`,
    `${'Model'.padEnd(22)} ${'Backend'.padEnd(10)} ${'PL'.padEnd(6)} ${'TTFT (ms)'.padEnd(12)} ${'TPS (t/s)'.padEnd(12)} ${'PL (t/s)'.padEnd(12)} ${'TG stddev'.padEnd(10)}`,
    `${'-'.repeat(92)}`,
  ];

  for (const r of allResults.results) {
    if (r.error) {
      summaryLines.push(`${(r.model || '').padEnd(22)} ${(r.backend || '').padEnd(10)} ${String(r.pl).padEnd(6)} ERROR: ${r.error}`);
    } else if (r.ttftMs !== undefined) {
      summaryLines.push(
        `${(r.model || '').padEnd(22)} ${(r.backend || '').padEnd(10)} ${String(r.pl).padEnd(6)} ${String(r.ttftMs?.toFixed(2) || '').padEnd(12)} ${String(r.tgTs?.toFixed(2) || '').padEnd(12)} ${String(r.plTs?.toFixed(2) || '').padEnd(12)} ${String(r.tgStddevTs?.toFixed(2) || '').padEnd(10)}`
      );
    }
  }

  const summaryFile = path.join(resultDir, 'llamacpp-results.txt');
  fs.writeFileSync(summaryFile, summaryLines.join('\n'));

  console.log(`\nResults saved to:`);
  console.log(`  JSON: ${resultFile}`);
  console.log(`  Text: ${summaryFile}`);
}

main().catch(err => {
  console.error(`Error: ${err.message}`);
  process.exit(1);
});
