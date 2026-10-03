import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export class LocalTranscriber {
  constructor(env = process.env) { this.env = env; this.pending = new Map(); this.serial = 0; this.child = null; }
  start() {
    if (this.child) return;
    const venv = fileURLToPath(new URL(process.platform === 'win32' ? '../.venv/Scripts/python.exe' : '../.venv/bin/python', import.meta.url));
    const python = this.env.PYTHON_BIN || (existsSync(venv) ? venv : process.platform === 'win32' ? 'python' : 'python3');
    const child = spawn(python, ['-u', fileURLToPath(new URL('./worker.py', import.meta.url))], { env: { ...process.env, ...this.env, HF_HUB_OFFLINE: '1', HF_HUB_DISABLE_TELEMETRY: '1' }, stdio: ['pipe', 'pipe', 'ignore'] });
    this.child = child;
    createInterface({ input: child.stdout }).on('line', line => {
      let message; try { message = JSON.parse(line); } catch { return; }
      const item = this.pending.get(message.id); if (!item) return;
      clearTimeout(item.timer); this.pending.delete(message.id);
      if (message.error) item.reject(new Error(message.error)); else item.resolve(message.result);
    });
    const failed = () => { if (this.child !== child) return; this.child = null; this.rejectAll('Python 음성 처리기가 종료되었습니다. Python 설치 및 로컬 모델 설정을 확인하세요.'); };
    child.on('error', failed); child.on('exit', failed); child.stdin.on('error', failed);
  }
  rejectAll(message) { for (const item of this.pending.values()) { clearTimeout(item.timer); item.reject(new Error(message)); } this.pending.clear(); }
  request(action, data = {}) {
    if (this.pending.size >= 3) return Promise.reject(new Error('로컬 음성 처리기가 사용 중입니다. 잠시 후 다시 시도하세요.'));
    this.start(); const id = ++this.serial;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.close('로컬 음성 처리 시간이 초과되었습니다. 더 작은 모델을 사용하세요.'); }, 120000);
      this.pending.set(id, { resolve, reject, timer });
      this.child.stdin.write(JSON.stringify({ ...data, action, id }) + '\n');
    });
  }
  close(message = '로컬 음성 처리기가 종료되었습니다.') { const child = this.child; this.child = null; this.rejectAll(message); child?.kill(); }
}
