import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
if (process.platform !== 'darwin' || process.arch !== 'arm64') {
  console.error('Qwen MLX needs native ARM Node.js and Python on an Apple Silicon Mac. Do not run under Rosetta.');
  process.exit(1);
}
const root = fileURLToPath(new URL('..', import.meta.url));
const python = fileURLToPath(new URL('../.venv-qwen/bin/python', import.meta.url));
if (!existsSync(python)) {
  const result = spawnSync('python3', ['-m', 'venv', '.venv-qwen'], { cwd: root, stdio: 'inherit' });
  if (result.status !== 0) process.exit(1);
}
const result = spawnSync(python, ['scripts/setup_qwen.py', ...process.argv.slice(2)], { cwd: root, stdio: 'inherit' });
process.exit(result.status ?? 1);
