/**
 * Shared utilities for perf test scripts.
 */

const { execSync } = require('child_process');
const os = require('os');
const path = require('path');
const fs = require('fs');

/**
 * Load config.json merged with local_config.json (if exists).
 * local_config.json overrides config.json values (shallow merge per section).
 */
function loadConfig() {
  const projectRoot = path.join(__dirname, '..');
  const configPath = path.join(projectRoot, 'config.json');
  const localConfigPath = path.join(projectRoot, 'local_config.json');

  const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));

  if (fs.existsSync(localConfigPath)) {
    const local = JSON.parse(fs.readFileSync(localConfigPath, 'utf8'));
    // Deep merge: override each section
    for (const [key, value] of Object.entries(local)) {
      if (typeof value === 'object' && !Array.isArray(value) && config[key] && typeof config[key] === 'object') {
        config[key] = { ...config[key], ...value };
      } else {
        config[key] = value;
      }
    }
  }

  // Resolve relative paths in config.paths to absolute (relative to project root)
  if (config.paths) {
    for (const [key, value] of Object.entries(config.paths)) {
      if (typeof value === 'string' && !path.isAbsolute(value)) {
        config.paths[key] = path.resolve(projectRoot, value);
      }
    }
  }

  return config;
}

/**
 * Collect system information (CPU, GPU, memory, OS).
 * Uses nvidia-smi when available for GPU details.
 */
function getSystemInfo() {
  const info = {
    timestamp: new Date().toISOString(),
    os: `${os.type()} ${os.release()} ${os.arch()}`,
    cpu: os.cpus()[0]?.model || 'Unknown',
    cpuCores: os.cpus().length,
    totalMemoryGB: (os.totalmem() / 1024 / 1024 / 1024).toFixed(1),
    gpu: 'Unknown',
    gpuDriver: 'Unknown',
  };

  // Get GPU info via WMI (works for NVIDIA, AMD, Intel)
  // Fetch all controllers, skip virtual/remote adapters and disabled devices (ConfigManagerErrorCode=22)
  try {
    const wmi = execSync(
      'powershell -NoProfile -Command "Get-WmiObject Win32_VideoController | Select-Object Name,DriverVersion,AdapterRAM,ConfigManagerErrorCode | ConvertTo-Json"',
      { encoding: 'utf8', timeout: 10000, stdio: 'pipe' }
    ).trim();
    const parsed = JSON.parse(wmi);
    const controllers = Array.isArray(parsed) ? parsed : [parsed];
    const virtualNames = /microsoft remote display|microsoft basic display|hyper-v video|indirect display/i;
    const obj = controllers.find(c =>
      c && c.Name &&
      c.ConfigManagerErrorCode !== 22 &&
      !virtualNames.test(c.Name)
    ) || controllers[0];
    if (obj && obj.Name) {
      info.gpu = obj.Name.trim();
      if (obj.DriverVersion) info.gpuDriver = obj.DriverVersion.trim();
      if (obj.AdapterRAM && obj.AdapterRAM > 0) {
        info.gpuMemoryMB = Math.round(obj.AdapterRAM / 1024 / 1024).toString();
      }
    }
  } catch {}

  return info;
}

module.exports = { getSystemInfo, loadConfig };
