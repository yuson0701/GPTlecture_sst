import { test } from 'node:test';
import assert from 'node:assert/strict';
import { StreamingFrames, StreamingQueue } from '../public/streaming.js';
import { createApp } from '../server.mjs';

test('packets contain new audio only, ordered exactly once, including final flush', () => {
  const packets = [], frames = new StreamingFrames(16000, packet => packets.push(packet));
  frames.push(new Float32Array(3500).fill(0.5)); frames.flush(); frames.flush();
  assert.equal(packets.length, 2);
  assert.deepEqual(packets.map(p => p.sequence), [0, 1]);
  assert.deepEqual(packets.map(p => Buffer.from(p.audio, 'base64').length), [6400, 600]);
  assert.equal(packets[0].final, false); assert.equal(packets[1].final, true);
});
test('one hour of capture has bounded buffering and no overlap or sample loss', () => {
  let count = 0, bytes = 0, finalCount = 0;
  const frames = new StreamingFrames(16000, packet => { count++; bytes += Buffer.from(packet.audio, 'base64').length; if (packet.final) finalCount++; });
  const block = new Float32Array(3200);
  for (let i = 0; i < 18000; i++) frames.push(block);
  frames.flush();
  assert.equal(bytes, 3600 * 16000 * 2); assert.equal(count, 18001); assert.equal(finalCount, 1);
  assert.equal(frames.buffer.length, 3200);
});
test('stream queue never coalesces incremental packets and retries at the same sequence', async () => {
  const seen = []; let fail = true;
  const queue = new StreamingQueue(async packet => { if (packet.sequence === 1 && fail) throw Error('network'); seen.push(packet.sequence); });
  for (let sequence = 0; sequence < 4; sequence++) queue.enqueue({ sequence, final: false });
  await queue.settle(); assert.deepEqual(seen, [0]); assert.deepEqual(queue.items.map(x => x.sequence), [1, 2, 3]);
  fail = false; await queue.retry(); assert.deepEqual(seen, [0, 1, 2, 3]);
});
test('streaming routes pass sequence/session; live summaries require explicit cloud consent', async t => {
  const calls = [];
  const server = createApp({ env: { STT_BACKEND: 'sherpa' }, transcriber: { request: async (action, data) => { calls.push({ action, data }); return action === 'status' ? { available: true } : { events: [] }; }, close() {} } });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  const base = `http://127.0.0.1:${server.address().port}`;
  const post = (path, body) => fetch(base + path, { method: 'POST', headers: { Origin: base, 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const config = await (await fetch(base + '/api/config')).json(); assert.equal(config.streaming, true);
  const response = await post('/api/transcribe', { audio: '', final: true, session: 'test', sequence: 0, glossary: '' });
  assert.equal(response.status, 200); assert.equal(calls.at(-1).data.session, 'test'); assert.equal(calls.at(-1).data.sequence, 0);
  assert.equal((await post('/api/transcribe', { audio: '', final: true, session: 'test', sequence: -1, glossary: '' })).status, 400);
  const summary = await (await post('/api/summary', { transcript: '기회비용은 포기한 최선의 대안의 가치입니다.', previous: '', live: true })).json();
  assert.equal(summary.code, 'consent_required'); assert.equal(summary.summary, undefined);
});
