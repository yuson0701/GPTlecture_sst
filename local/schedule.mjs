import { readFile, mkdir, writeFile, rename, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';

const fail = message => Object.assign(new Error(message), { status: 400 });
export class ScheduleStore {
  constructor(env = {}) {
    this.path = env.LECTURE_SCHEDULE_FILE || fileURLToPath(new URL('../data/schedule.json', import.meta.url));
    const venv = fileURLToPath(new URL('../.venv-qwen/bin/python', import.meta.url));
    this.python = env.PYTHON_BIN || (existsSync(venv) ? venv : 'python3');
    this.busy = false;
  }
  async get() {
    try { return JSON.parse(await readFile(this.path, 'utf8')); }
    catch (error) { if (error.code === 'ENOENT') return { schema: 1, lectures: [] }; throw error; }
  }
  async import({ filename, xlsx } = {}) {
    if (this.busy) throw fail('시간표를 불러오는 중입니다. 잠시 후 다시 시도하세요.');
    if (typeof filename !== 'string' || filename.length > 250 || !filename.toLowerCase().endsWith('.xlsx') ||
        typeof xlsx !== 'string' || xlsx.length > 2800000 || xlsx.length % 4 || !/^[A-Za-z0-9+/]+={0,2}$/.test(xlsx)) throw fail('2 MB 이하의 .xlsx 시간표를 선택하세요.');
    const bytes = Buffer.from(xlsx, 'base64');
    if (bytes.length > 2 * 1024 * 1024) throw fail('시간표는 2 MB 이하만 지원합니다.');
    this.busy = true;
    try {
      const result = await new Promise((resolve, reject) => {
        const child = spawn(this.python, [fileURLToPath(new URL('../scripts/import_schedule.py', import.meta.url)), '--stdin'], { stdio: ['pipe', 'pipe', 'pipe'] });
        child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8');
        let stdout = '', stderr = '';
        const timer = setTimeout(() => { child.kill(); reject(fail('시간표 처리 시간이 초과되었습니다.')); }, 20000);
        child.on('error', () => { clearTimeout(timer); reject(fail('시간표를 읽을 Python이 없습니다. 모델 설치를 먼저 완료하세요.')); });
        child.stdin.on('error', () => {});
        child.stdout.on('data', chunk => { stdout += chunk; if (stdout.length > 4000000) { child.kill(); reject(fail('시간표가 너무 큽니다.')); } });
        child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-1000); });
        child.on('close', code => { clearTimeout(timer); if (code !== 0) reject(fail(stderr.trim() || '시간표를 읽지 못했습니다.')); else { try { resolve(JSON.parse(stdout)); } catch { reject(fail('시간표 변환 결과를 읽지 못했습니다.')); } } });
        child.stdin.end(bytes);
      });
      result.sourceName = basename(filename).normalize('NFC');
      await mkdir(dirname(this.path), { recursive: true });
      const temporary = `${this.path}.${randomUUID()}.tmp`;
      try { await writeFile(temporary, JSON.stringify(result), { mode: 0o600 }); await rename(temporary, this.path); }
      finally { await rm(temporary, { force: true }); }
      return result;
    } finally { this.busy = false; }
  }
}
