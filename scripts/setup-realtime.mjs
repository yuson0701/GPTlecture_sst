import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('..', import.meta.url));
const python = fileURLToPath(new URL(process.platform === 'win32' ? '../.venv/Scripts/python.exe' : '../.venv/bin/python', import.meta.url));
if (!existsSync(python)) {
  const result = spawnSync(process.platform === 'win32' ? 'python' : 'python3', ['-m', 'venv', '.venv'], { cwd: root, stdio: 'inherit' });
  if (result.status !== 0) { console.error('Install Python 3.10–3.12 first.'); process.exit(1); }
}
const result = spawnSync(python, ['scripts/setup_realtime.py'], { cwd: root, stdio: 'inherit' });
process.exit(result.status ?? 1);
