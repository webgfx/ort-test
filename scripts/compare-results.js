/**
 * Compare benchmark results across multiple runs and generate an HTML report with charts.
 *
 * Usage:
 *   node scripts/compare-results.js 20260305153602 20260305141935
 *   node scripts/compare-results.js --list                          # List available results
 *   node scripts/compare-results.js --latest 3                      # Compare latest 3 runs
 */

const path = require('path');
const fs = require('fs');

const config = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'config.json'), 'utf8'));
const RESULTS_DIR = config.paths.results || path.resolve(path.join(__dirname, '..', 'gitignore', 'results'));

// ============================================================
// Load results
// ============================================================

function loadResult(dirName) {
  const dir = path.join(RESULTS_DIR, dirName);
  if (!fs.existsSync(dir)) {
    console.error(`Result directory not found: ${dir}`);
    process.exit(1);
  }

  const files = fs.readdirSync(dir).filter(f => f.endsWith('-results.json'));
  if (files.length === 0) {
    console.error(`No result JSON files found in ${dir}`);
    process.exit(1);
  }

  const allResults = [];
  for (const file of files) {
    const data = JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
    const source = file.replace('-results.json', ''); // 'ort', 'llamacpp', etc.
    allResults.push({ source, dirName, data });
  }
  return allResults;
}

function getAvailableResults() {
  if (!fs.existsSync(RESULTS_DIR)) return [];
  return fs.readdirSync(RESULTS_DIR, { withFileTypes: true })
    .filter(d => d.isDirectory() && /^\d{14}$/.test(d.name))
    .map(d => {
      const files = fs.readdirSync(path.join(RESULTS_DIR, d.name))
        .filter(f => f.endsWith('-results.json'));
      return { name: d.name, files };
    })
    .filter(d => d.files.length > 0)
    .sort((a, b) => b.name.localeCompare(a.name));
}

function formatTimestamp(ts) {
  // 20260305153602 -> 2026-03-05 15:36:02
  return `${ts.slice(0,4)}-${ts.slice(4,6)}-${ts.slice(6,8)} ${ts.slice(8,10)}:${ts.slice(10,12)}:${ts.slice(12,14)}`;
}

// ============================================================
// Normalize results into comparable series
// ============================================================

function normalizeResults(loaded) {
  const series = [];

  for (const { source, dirName, data } of loaded) {
    for (const r of data.results) {
      if (r.error) continue;

      // Build a label for this configuration
      let configLabel;
      if (source === 'llamacpp') {
        configLabel = `${r.model} / ${r.backend} (llama.cpp)`;
      } else if (source === 'ort') {
        configLabel = `${r.model} / ${r.ep} (ORT)`;
      } else {
        configLabel = `${r.model} / ${source}`;
      }

      // Find or create series
      const seriesKey = `${dirName}|${configLabel}`;
      let s = series.find(x => x.key === seriesKey);
      if (!s) {
        s = {
          key: seriesKey,
          label: `${configLabel} [${formatTimestamp(dirName)}]`,
          shortLabel: configLabel,
          dirName,
          source,
          points: [],
        };
        series.push(s);
      }

      s.points.push({
        pp: r.pp,
        ttftMs: r.ttftMs,
        tgTs: r.tgTs,
        ppTs: r.ppTs,
        e2eMs: r.e2eMs,
      });
    }
  }

  // Sort points by pp within each series
  for (const s of series) {
    s.points.sort((a, b) => a.pp - b.pp);
  }

  return series;
}

// ============================================================
// Generate HTML report
// ============================================================

function generateHtml(series, outputPath) {
  // Collect all unique pp values
  const allPPs = [...new Set(series.flatMap(s => s.points.map(p => p.pp)))].sort((a, b) => a - b);

  // Color palette
  const colors = [
    '#4285F4', '#EA4335', '#34A853', '#FBBC05',
    '#8E24AA', '#00ACC1', '#FF7043', '#5C6BC0',
    '#43A047', '#E53935', '#1E88E5', '#FDD835',
  ];

  // Build datasets for each chart
  function buildDatasets(metric) {
    return series.map((s, i) => {
      const color = colors[i % colors.length];
      const data = allPPs.map(pp => {
        const pt = s.points.find(p => p.pp === pp);
        return pt ? pt[metric] : null;
      });
      return {
        label: s.label,
        data,
        borderColor: color,
        backgroundColor: color + '33',
        pointBackgroundColor: color,
        pointRadius: 5,
        pointHoverRadius: 7,
        borderWidth: 2.5,
        tension: 0.3,
        fill: false,
      };
    });
  }

  // Build bar datasets for comparison at each pp
  function buildBarDatasets(metric) {
    return series.map((s, i) => {
      const color = colors[i % colors.length];
      const data = allPPs.map(pp => {
        const pt = s.points.find(p => p.pp === pp);
        return pt ? pt[metric] : null;
      });
      return {
        label: s.label,
        data,
        backgroundColor: color + 'CC',
        borderColor: color,
        borderWidth: 1,
      };
    });
  }

  const ttftDatasets = JSON.stringify(buildDatasets('ttftMs'));
  const tpsDatasets = JSON.stringify(buildDatasets('tgTs'));
  const ttftBarDatasets = JSON.stringify(buildBarDatasets('ttftMs'));
  const tpsBarDatasets = JSON.stringify(buildBarDatasets('tgTs'));
  const ppLabels = JSON.stringify(allPPs.map(p => `PP=${p}`));

  // Summary table
  const tableRows = series.flatMap(s =>
    s.points.map(p => `
      <tr>
        <td>${s.shortLabel}</td>
        <td>${s.dirName}</td>
        <td>${p.pp}</td>
        <td>${p.ttftMs != null ? p.ttftMs.toFixed(2) : '-'}</td>
        <td>${p.tgTs != null ? p.tgTs.toFixed(2) : '-'}</td>
        <td>${p.ppTs != null ? p.ppTs.toFixed(2) : '-'}</td>
        <td>${p.e2eMs != null ? p.e2eMs.toFixed(0) : '-'}</td>
      </tr>`)
  ).join('');

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Benchmark Comparison</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: #0d1117;
      color: #c9d1d9;
      padding: 24px;
    }
    h1 {
      font-size: 28px;
      font-weight: 600;
      color: #f0f6fc;
      margin-bottom: 8px;
    }
    .subtitle {
      color: #8b949e;
      font-size: 14px;
      margin-bottom: 32px;
    }
    .chart-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-bottom: 40px;
    }
    .chart-card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 20px;
    }
    .chart-card h2 {
      font-size: 16px;
      font-weight: 600;
      color: #f0f6fc;
      margin-bottom: 16px;
    }
    .chart-card canvas {
      max-height: 360px;
    }
    .chart-card.full-width {
      grid-column: 1 / -1;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 12px;
      overflow: hidden;
    }
    th {
      background: #21262d;
      color: #f0f6fc;
      font-weight: 600;
      text-align: left;
      padding: 10px 14px;
      border-bottom: 1px solid #30363d;
    }
    td {
      padding: 8px 14px;
      border-bottom: 1px solid #21262d;
    }
    tr:hover td { background: #1c2128; }
    tr:last-child td { border-bottom: none; }
    .table-section {
      margin-top: 40px;
    }
    .table-section h2 {
      font-size: 18px;
      font-weight: 600;
      color: #f0f6fc;
      margin-bottom: 12px;
    }
    @media (max-width: 900px) {
      .chart-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <h1>Benchmark Comparison</h1>
  <p class="subtitle">Generated on ${new Date().toISOString().slice(0, 19).replace('T', ' ')} &mdash; Comparing ${series.map(s => s.dirName).filter((v, i, a) => a.indexOf(v) === i).join(', ')}</p>

  <div class="chart-grid">
    <div class="chart-card">
      <h2>TTFT (Time to First Token) — Lower is Better</h2>
      <canvas id="ttftLine"></canvas>
    </div>
    <div class="chart-card">
      <h2>TPS (Token Generation Speed) — Higher is Better</h2>
      <canvas id="tpsLine"></canvas>
    </div>
    <div class="chart-card">
      <h2>TTFT by Prompt Length</h2>
      <canvas id="ttftBar"></canvas>
    </div>
    <div class="chart-card">
      <h2>TPS by Prompt Length</h2>
      <canvas id="tpsBar"></canvas>
    </div>
  </div>

  <div class="table-section">
    <h2>Detailed Results</h2>
    <table>
      <thead>
        <tr>
          <th>Configuration</th>
          <th>Run</th>
          <th>PP</th>
          <th>TTFT (ms)</th>
          <th>TPS (t/s)</th>
          <th>PP (t/s)</th>
          <th>E2E (ms)</th>
        </tr>
      </thead>
      <tbody>${tableRows}
      </tbody>
    </table>
  </div>

  <script>
    const ppLabels = ${ppLabels};

    const chartDefaults = {
      color: '#c9d1d9',
      borderColor: '#30363d',
      font: { family: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif' }
    };

    Chart.defaults.color = chartDefaults.color;
    Chart.defaults.font.family = chartDefaults.font.family;

    const gridOpts = {
      color: '#21262d',
      drawBorder: false,
    };

    // TTFT Line Chart
    new Chart(document.getElementById('ttftLine'), {
      type: 'line',
      data: { labels: ppLabels, datasets: ${ttftDatasets} },
      options: {
        responsive: true,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'bottom', labels: { padding: 16, usePointStyle: true, pointStyle: 'circle' } },
          tooltip: { callbacks: { label: ctx => ctx.dataset.label + ': ' + (ctx.parsed.y?.toFixed(2) || '-') + ' ms' } },
        },
        scales: {
          x: { grid: gridOpts, title: { display: true, text: 'Prompt Length' } },
          y: { grid: gridOpts, title: { display: true, text: 'TTFT (ms)' }, beginAtZero: true },
        },
      },
    });

    // TPS Line Chart
    new Chart(document.getElementById('tpsLine'), {
      type: 'line',
      data: { labels: ppLabels, datasets: ${tpsDatasets} },
      options: {
        responsive: true,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'bottom', labels: { padding: 16, usePointStyle: true, pointStyle: 'circle' } },
          tooltip: { callbacks: { label: ctx => ctx.dataset.label + ': ' + (ctx.parsed.y?.toFixed(2) || '-') + ' t/s' } },
        },
        scales: {
          x: { grid: gridOpts, title: { display: true, text: 'Prompt Length' } },
          y: { grid: gridOpts, title: { display: true, text: 'TPS (tokens/s)' }, beginAtZero: true },
        },
      },
    });

    // TTFT Bar Chart
    new Chart(document.getElementById('ttftBar'), {
      type: 'bar',
      data: { labels: ppLabels, datasets: ${ttftBarDatasets} },
      options: {
        responsive: true,
        plugins: {
          legend: { position: 'bottom', labels: { padding: 16, usePointStyle: true, pointStyle: 'rect' } },
          tooltip: { callbacks: { label: ctx => ctx.dataset.label + ': ' + (ctx.parsed.y?.toFixed(2) || '-') + ' ms' } },
        },
        scales: {
          x: { grid: gridOpts },
          y: { grid: gridOpts, title: { display: true, text: 'TTFT (ms)' }, beginAtZero: true },
        },
      },
    });

    // TPS Bar Chart
    new Chart(document.getElementById('tpsBar'), {
      type: 'bar',
      data: { labels: ppLabels, datasets: ${tpsBarDatasets} },
      options: {
        responsive: true,
        plugins: {
          legend: { position: 'bottom', labels: { padding: 16, usePointStyle: true, pointStyle: 'rect' } },
          tooltip: { callbacks: { label: ctx => ctx.dataset.label + ': ' + (ctx.parsed.y?.toFixed(2) || '-') + ' t/s' } },
        },
        scales: {
          x: { grid: gridOpts },
          y: { grid: gridOpts, title: { display: true, text: 'TPS (tokens/s)' }, beginAtZero: true },
        },
      },
    });
  </script>
</body>
</html>`;

  fs.writeFileSync(outputPath, html);
  return outputPath;
}

// ============================================================
// CLI
// ============================================================

function main() {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
Usage: node scripts/compare-results.js [options] <run1> <run2> [run3...]

Arguments:
  <run>                       Result folder name (timestamp), e.g. 20260305153602

Options:
  --list, -l                  List available result folders
  --latest <n>                Compare the latest n runs (default: 2)
  -o, --output <file>         Output HTML file (default: comparison.html in results dir)
  -h, --help                  Show this help
`);
    return;
  }

  if (args.includes('--list') || args.includes('-l')) {
    const results = getAvailableResults();
    if (results.length === 0) {
      console.log(`No results found in ${RESULTS_DIR}`);
      return;
    }
    console.log(`Available results in ${RESULTS_DIR}:\n`);
    for (const r of results) {
      console.log(`  ${r.name}  ${formatTimestamp(r.name)}  [${r.files.join(', ')}]`);
    }
    return;
  }

  // Parse --latest
  const latestIdx = args.findIndex(a => a === '--latest');
  let runNames = [];

  if (latestIdx >= 0) {
    const n = parseInt(args[latestIdx + 1]) || 2;
    const available = getAvailableResults();
    runNames = available.slice(0, n).map(r => r.name);
    if (runNames.length < 2) {
      console.error(`Need at least 2 results to compare, found ${runNames.length}.`);
      process.exit(1);
    }
  } else {
    // Filter out option flags
    let outputPath = null;
    for (let i = 0; i < args.length; i++) {
      if (args[i] === '-o' || args[i] === '--output') {
        outputPath = args[++i];
      } else if (!args[i].startsWith('-')) {
        runNames.push(args[i]);
      }
    }
  }

  if (runNames.length < 1) {
    console.error('Please provide at least one result folder to compare.');
    console.error('Usage: node scripts/compare-results.js <run1> <run2> [run3...]');
    process.exit(1);
  }

  // Parse output path
  const outputIdx = args.findIndex(a => a === '-o' || a === '--output');
  const outputPath = outputIdx >= 0 ? args[outputIdx + 1] : path.join(RESULTS_DIR, 'comparison.html');

  console.log(`Loading results from ${runNames.length} run(s)...\n`);

  // Load all results
  const allLoaded = [];
  for (const name of runNames) {
    const loaded = loadResult(name);
    allLoaded.push(...loaded);
    for (const l of loaded) {
      const count = l.data.results.filter(r => !r.error).length;
      console.log(`  ${name}/${l.source}: ${count} data points`);
    }
  }

  // Normalize and generate
  const series = normalizeResults(allLoaded);
  console.log(`\n${series.length} series to compare:\n`);
  for (const s of series) {
    console.log(`  ${s.label} (${s.points.length} points)`);
  }

  const htmlPath = generateHtml(series, outputPath);
  console.log(`\nReport saved to: ${htmlPath}`);

  return htmlPath;
}

const htmlPath = main();

// Try to open in browser
if (htmlPath) {
  try {
    require('child_process').execSync(`start "" "${htmlPath}"`, { stdio: 'ignore' });
  } catch {}
}
