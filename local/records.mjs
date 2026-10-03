import { mkdir, readFile, readdir, open, rename, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
const validId = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;
const fail = (message, status = 400) => Object.assign(new Error(message), { status });
const text = (x, max) => typeof x === 'string' && x.length <= max;
const seconds = x => Number.isFinite(x) && x >= 0;
export function validateRecord(value) {
  if (!value || !Number.isSafeInteger(value.revision) || value.revision < 0 || !text(value.title, 120) || !text(value.glossary, 1500) || !seconds(value.elapsed) || !['idle', 'connecting', 'live', 'stopping', 'demo'].includes(value.state)) throw fail('Invalid lecture metadata');
  if (!Array.isArray(value.transcript) || value.transcript.length > 50000 || value.transcript.some(x => !x || !text(x.id, 150) || !text(x.text, 20000) || !seconds(x.seconds) || typeof x.final !== 'boolean')) throw fail('Invalid transcript');
  if (new Set(value.transcript.map(x => x.id)).size !== value.transcript.length) throw fail('Duplicate transcript IDs');
  if (!Array.isArray(value.summarized) || value.summarized.length > 100000 || value.summarized.some(x => !text(x, 180))) throw fail('Invalid summary references');
  if (!Array.isArray(value.blocks) || value.blocks.length > 10000 || value.blocks.some(x => !x || !Number.isSafeInteger(x.id) || x.id < 1 || !text(x.source, 10000) || !['pending','failed','done'].includes(x.state) || !Array.isArray(x.items) || !x.items.length || x.items.length > 1000 || x.items.some(i => !i || !seconds(i.seconds) || (i.id !== undefined && !text(i.id, 180)) || (i.text !== undefined && !text(i.text, 20000))) || ['title','text','cleaned','warning'].some(k => x[k] !== undefined && !text(x[k], 20000)))) throw fail('Invalid summary blocks');
  if (new Set(value.blocks.map(x => x.id)).size !== value.blocks.length) throw fail('Duplicate block IDs');
  // Copy only the record fields; audio and client timestamps are never stored.
  return { title: value.title, glossary: value.glossary, elapsed: value.elapsed, state: value.state, transcript: value.transcript.map(x => ({ id: x.id, text: x.text, seconds: x.seconds, final: x.final, failed: x.failed === true })),
    blocks: value.blocks.map(x => ({ id: x.id, source: x.source, state: x.state,
      method: ['chatgpt', 'ollama', 'extractive', 'demo'].includes(x.method) ? x.method : undefined,
      title: x.title, text: x.text, cleaned: x.cleaned, warning: x.warning,
      items: x.items.map(i => ({ id: i.id, seconds: i.seconds, text: i.text })) })), summarized: [...value.summarized] };
}
export class RecordStore {
  constructor(directory) { this.directory = directory; this.pending = Promise.resolve(); }
  path(id) { if (!validId.test(id)) throw fail('Invalid record ID'); return join(this.directory, `${id}.json`); }
  async get(id) {
    try { return JSON.parse(await readFile(this.path(id), 'utf8')); }
    catch (error) { if (error.code === 'ENOENT') throw fail('기록을 찾을 수 없습니다.', 404); throw error; }
  }
  async list() {
    let names;
    try { names = await readdir(this.directory); } catch (e) { if (e.code === 'ENOENT') return []; throw e; }
    const records = await Promise.all(names.filter(name => name.endsWith('.json') && validId.test(name.slice(0, -5))).map(name => this.get(name.slice(0, -5))));
    return records.map(({ id, title, elapsed, state, createdAt, updatedAt, transcript }) => ({ id, title, elapsed, state, createdAt, updatedAt, segments: transcript.length, preview: transcript.find(x => x.text)?.text.slice(0, 160) || '' })).sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  }
  save(id, value) {
    const destination = this.path(id), clean = validateRecord(value), expected = value.revision;
    const work = this.pending.then(async () => {
      let current;
      try { current = await this.get(id); } catch (e) { if (e.status !== 404) throw e; }
      if ((current?.revision || 0) !== expected) throw fail('다른 탭에서 수정한 기록입니다. 현재 내용을 내보낸 뒤 다시 열어 주세요.', 409);
      const now = new Date().toISOString();
      const record = { ...clean, id, revision: expected + 1, createdAt: current?.createdAt || now, updatedAt: now };
      await mkdir(this.directory, { recursive: true, mode: 0o700 });
      const temporary = join(this.directory, `.${randomUUID()}.tmp`);
      try {
        const file = await open(temporary, 'wx', 0o600);
        try { await file.writeFile(JSON.stringify(record)); await file.sync(); } finally { await file.close(); }
        await rename(temporary, destination);
      } finally { await rm(temporary, { force: true }); }
      return { id, revision: record.revision, createdAt: record.createdAt, updatedAt: now };
    });
    this.pending = work.catch(() => {});
    return work;
  }
}
