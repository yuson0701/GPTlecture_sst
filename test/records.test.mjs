import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { RecordStore } from '../local/records.mjs';
import { createApp } from '../server.mjs';
const record = () => ({ revision: 0, title: '경제학 강의', glossary: '기회비용', elapsed: 3600, state: 'idle', transcript: [{ id: 'one', seconds: 0, final: true, text: '기회비용을 배웁니다.' }], blocks: [{ id: 1, source: '기회비용을 배웁니다.', items: [{ id: 'one:0', seconds: 0, text: '기회비용을 배웁니다.' }], state: 'done', method: 'chatgpt', sections: [{ title: '기회비용', bullets: ['최선의 대안의 가치입니다.'] }], attempts: [{ input_tokens: 100, output_tokens: 50, total_tokens: 150 }, null], text: '선택의 비용을 설명합니다.', cleaned: '기회비용을 배웁니다.' }], summarized: ['one:0'] });
async function directory(t) { const path = await mkdtemp(join(tmpdir(), 'lecture-records-')); t.after(() => rm(path, { recursive: true, force: true })); return path; }
test('records survive a new store instance with all source/summary pairs', async t => {
  const dir = await directory(t), id = randomUUID();
  const first = await new RecordStore(dir).save(id, { ...record(), audio: 'must not persist' });
  const store = new RecordStore(dir), saved = await store.get(id);
  assert.equal(saved.blocks[0].cleaned, record().blocks[0].cleaned);
  assert.deepEqual(saved.blocks[0].sections, record().blocks[0].sections);
  assert.deepEqual(saved.blocks[0].attempts, record().blocks[0].attempts);
  assert.deepEqual(saved.summarized, ['one:0']); assert.equal(saved.elapsed, 3600);
  assert.equal(saved.audio, undefined); assert.equal(first.revision, 1);
  const second = await store.save(id, { ...saved, title: '수정된 제목' });
  assert.equal(second.revision, 2); assert.equal(second.createdAt, first.createdAt);
  assert.equal((await store.list())[0].title, '수정된 제목');
});
test('concurrent and stale updates cannot overwrite a newer lecture', async t => {
  const store = new RecordStore(await directory(t)), id = randomUUID();
  await store.save(id, record());
  const results = await Promise.allSettled([store.save(id, { ...record(), revision: 1, title: 'A' }), store.save(id, { ...record(), revision: 1, title: 'B' })]);
  assert.equal(results.filter(x => x.status === 'fulfilled').length, 1);
  assert.equal(results.find(x => x.status === 'rejected').reason.status, 409);
  assert.equal((await store.get(id)).title, 'A');
});
test('invalid records, missing files and traversal leave existing data intact', async t => {
  const dir = await directory(t), store = new RecordStore(dir), id = randomUUID();
  await store.save(id, record()); const before = await readFile(join(dir, id + '.json'), 'utf8');
  assert.throws(() => store.save('../escape', record()), /Invalid/);
  assert.throws(() => store.save(id, { ...record(), transcript: [{}] }), /Invalid/);
  assert.equal(await readFile(join(dir, id + '.json'), 'utf8'), before);
  await assert.rejects(store.get(randomUUID()), e => e.status === 404);
  await writeFile(join(dir, '.interrupted.tmp'), 'incomplete');
  assert.equal((await store.list()).length, 1);
});
test('record API requires same-origin writes and does not expose the data directory', async t => {
  const dir = await directory(t);
  const server = createApp({ env: { LECTURE_DATA_DIR: dir }, transcriber: { request: async () => ({ available: false }), close() {} } });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  const base = `http://127.0.0.1:${server.address().port}`, id = randomUUID();
  const post = origin => fetch(`${base}/api/records/${id}`, { method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json' }, body: JSON.stringify(record()) });
  assert.equal((await post('https://other.example')).status, 403);
  assert.equal((await post(base)).status, 200);
  assert.equal((await post(base)).status, 409);
  // An hour of notes can exceed the older speech endpoint's 700 KB body limit.
  const longLecture = { ...record(), revision: 1, transcript: Array.from({ length: 3600 }, (_, i) => ({ id: `line-${i}`, seconds: i, final: true, text: '긴 강의 기록입니다. '.repeat(25) })) };
  const largeSave = await fetch(`${base}/api/records/${id}`, { method: 'POST', headers: { Origin: base, 'Content-Type': 'application/json' }, body: JSON.stringify(longLecture) });
  assert.equal(largeSave.status, 200);
  const reopened = await (await fetch(`${base}/api/records/${id}`)).json();
  assert.equal(reopened.transcript.length, 3600); assert.equal(reopened.transcript.at(-1).seconds, 3599);
  assert.equal((await (await fetch(base + '/api/records')).json()).records.length, 1);
  assert.equal((await fetch(base + '/data/lectures/' + id + '.json')).status, 404);
  assert.equal((await fetch(base + '/api/records', { headers: { Origin: 'https://other.example' } })).status, 403);
});
