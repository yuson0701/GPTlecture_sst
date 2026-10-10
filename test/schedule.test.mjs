import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { ScheduleStore } from '../local/schedule.mjs';
import { suggestVocabulary, normalizeTerms } from '../public/lecture-vocabulary.js';
import { createApp } from '../server.mjs';

test('vocabulary follows lecture topic, deduplicates terms and bounds prompt size', () => {
  const cardiac = suggestVocabulary({ title: '심장근 세포의 수축조절', course: '심순' });
  assert.ok(cardiac.includes('preload')); assert.ok(cardiac.includes('미오신'));
  assert.ok(!cardiac.includes('COPD'));
  assert.ok(suggestVocabulary({ title: '심장과 심전도', course: '심순' }).includes('ECG'));
  assert.ok(suggestVocabulary({ title: '천식', course: '호흡' }).includes('asthma'));
  assert.deepEqual(suggestVocabulary({ title: '새 주제', course: '미분류' }), ['새 주제']);
  assert.deepEqual(normalizeTerms('ECG, ecg, \n 심전도,심전도'), ['ECG', '심전도']);
  const many = suggestVocabulary({ title: '심전도 근육 심장근 혈압 폐렴 간염 소장 골절', course: '심순' });
  assert.ok(many.length <= 50); assert.ok(many.join(', ').length <= 1500);
});

test('XLSX import survives restart, keeps only schedule fields and preserves previous data on failure', async t => {
  const dir = await mkdtemp(join(tmpdir(), 'lecture-schedule-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const env = { LECTURE_SCHEDULE_FILE: join(dir, 'schedule.json') };
  const store = new ScheduleStore(env);
  assert.deepEqual((await store.get()).lectures, []);
  const xlsx = execFileSync(store.python, ['test/schedule_test.py', '--fixture']).toString('base64');
  const imported = await store.import({ filename: 'schedule.xlsx', xlsx });
  assert.equal(imported.lectures.length, 2);
  assert.equal(imported.sourceName, 'schedule.xlsx');
  assert.doesNotMatch(JSON.stringify(imported), /PRIVATE|DO NOT IMPORT/);
  assert.deepEqual(await new ScheduleStore(env).get(), imported);
  const before = await readFile(env.LECTURE_SCHEDULE_FILE, 'utf8');
  await assert.rejects(store.import({ filename: 'bad.xlsx', xlsx: Buffer.from('invalid').toString('base64') }));
  await assert.rejects(store.import({ filename: 'bad.xls', xlsx }));
  assert.equal(await readFile(env.LECTURE_SCHEDULE_FILE, 'utf8'), before);
});

test('schedule API rejects foreign origins and selected vocabulary reaches the speech worker unchanged', async t => {
  const lecture = { id: 'a'.repeat(24), date: '2026-10-12', period: '1', course: '심순', title: '심장근 세포의 수축조절', professor: '' };
  const calls = [];
  const server = createApp({ env: {}, schedule: { get: async () => ({ lectures: [lecture] }), import: async () => { throw Object.assign(Error('Invalid workbook'), { status: 400 }); } },
    transcriber: { request: async (action, body) => { calls.push({ action, body }); return { text: '심근 수축' }; }, close() {} } });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  const base = `http://127.0.0.1:${server.address().port}`;
  const post = (path, body, origin = base) => fetch(base + path, { method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  assert.equal((await fetch(base + '/api/schedule', { headers: { Origin: 'https://elsewhere.example' } })).status, 403);
  assert.equal((await post('/api/schedule/import', {}, 'https://elsewhere.example')).status, 403);
  assert.equal((await post('/api/schedule/import', {})).status, 400);
  const selected = (await (await fetch(base + '/api/schedule')).json()).lectures[0];
  const glossary = suggestVocabulary(selected).join(', ');
  assert.equal((await post('/api/transcribe', { glossary, audio: Buffer.alloc(32000).toString('base64') })).status, 200);
  assert.equal(calls[0].body.glossary, glossary);
  for (const path of ['/lecture-picker.js', '/lecture-vocabulary.js']) assert.equal((await fetch(base + path)).status, 200);
  assert.equal((await fetch(base + '/data/schedule.json')).status, 404);
});
