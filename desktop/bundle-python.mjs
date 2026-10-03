// Run on an ARM Mac with uv installed. Copy a relocatable Python distribution,
// not a venv whose interpreter still points to the build machine.
import { spawnSync } from 'node:child_process';
import { cpSync, mkdirSync, rmSync } from 'node:fs';
import { resolve, join } from 'node:path';
if (process.platform !== 'darwin' || process.arch !== 'arm64') throw Error('Build the Python runtime on an Apple Silicon Mac.');
const managed = resolve('build/python-install');
const env = { ...process.env, UV_PYTHON_INSTALL_DIR: managed };
function run(command, args, capture = false) {
  const result = spawnSync(command, args, { env, encoding: 'utf8', stdio: capture ? ['ignore', 'pipe', 'inherit'] : 'inherit' });
  if (result.status !== 0) throw Error(`${command} failed (${result.status})`);
  return result.stdout?.trim();
}
run('uv', ['python', 'install', '3.14']);
const sourcePython = run('uv', ['python', 'find', '--managed-python', '3.14'], true);
const prefix = run(sourcePython, ['-c', 'import sys; print(sys.prefix)'], true);
const destination = resolve('build/python');
rmSync(destination, { recursive: true, force: true }); mkdirSync('build', { recursive: true });
cpSync(prefix, destination, { recursive: true, dereference: false });
const python = join(destination, 'bin/python3.14');
run('uv', ['pip', 'install', '--python', python, '--break-system-packages', '-r', 'requirements-qwen-mac.txt']);
run(python, ['-c', 'import mlx.core, mlx_audio, transformers, faster_whisper; from importlib.metadata import version; print({p:version(p) for p in ["mlx", "mlx-audio", "transformers", "huggingface-hub"]})']);
// Check relocation again before packaging.
const relocated = resolve('build/python-relocation-test');
rmSync(relocated, { recursive: true, force: true });
cpSync(destination, relocated, { recursive: true, dereference: false });
run(join(relocated, 'bin/python3.14'), ['-c', 'import sys, mlx.core, faster_whisper; print("Relocated runtime OK:", sys.prefix)']);
rmSync(relocated, { recursive: true, force: true });
