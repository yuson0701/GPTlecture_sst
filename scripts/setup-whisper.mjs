import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
if (process.platform !== 'darwin' || process.arch !== 'arm64') {
  console.error('This Whisper GPU installer requires native ARM Node.js on an Apple Silicon Mac.');
  process.exit(1);
}
const root = fileURLToPath(new URL('..', import.meta.url));
const python = fileURLToPath(new URL('../.venv-whisper/bin/python', import.meta.url));
if (!existsSync(python)) {
  const result = spawnSync('python3', ['-m', 'venv', '.venv-whisper'], { cwd: root, stdio: 'inherit' });
  if (result.status !== 0) process.exit(1);
}
const result = spawnSync(python, ['scripts/setup_whisper.py', ...process.argv.slice(2)], { cwd: root, stdio: 'inherit' });
process.exit(result.status ?? 1);
