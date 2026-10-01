import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createApp } from '../server.mjs';
import { extractiveSummary, localSummary } from '../local/summary.mjs';
import { AudioQueue, Segmenter, pcmBase64, removeOverlap } from '../public/audio.js';
import { Transcript } from '../public/transcript.js';

const noWorker = { request: async () => ({ available: false }), close() {} };
async function serve(t, options = {}) {
  const server = createApp({ env: {}, transcriber: noWorker, ...options });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  const base = `http://127.0.0.1:${server.address().port}`;
  return { base, post: (path, body, origin = base) => fetch(base + path, { method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) };
}
const audio = Buffer.alloc(16000 * 2).toString('base64');
test('serves no-key demo and advertises local-only model configuration', async t => {
  const { base } = await serve(t);
  assert.equal((await fetch(base)).status, 200);
  assert.equal((await fetch(base + '/.env')).status, 404);
  const config = await (await fetch(base + '/api/config')).json();
  assert.equal(config.configured, false); assert.equal(config.local, true); assert.match(config.model, /faster-whisper/);
  const page = await fetch(base); assert.match(page.headers.get('content-security-policy'), /connect-src 'self';/);
  assert.doesNotMatch(await page.text(), /api\.openai\.com|OPENAI_API_KEY/);
});
test('rejects cross-origin calls and malformed or oversized PCM before worker invocation', async t => {
  const { post } = await serve(t, { transcriber: { request() { throw Error('Must not run'); }, close() {} } });
  assert.equal((await post('/api/session', { glossary: '' }, 'https://evil.example')).status, 403);
  assert.equal((await post('/api/transcribe', { glossary: '', audio: 'not-base64' })).status, 400);
  assert.equal((await post('/api/transcribe', { glossary: '', audio: Buffer.alloc(480002).toString('base64') })).status, 400);
  assert.equal((await post('/api/summary', { transcript: 3, previous: '' })).status, 400);
});
test('loads local model then sends PCM and glossary to the local worker', async t => {
  const calls = [];
  const { post } = await serve(t, { transcriber: { request: async (action, body) => { calls.push({ action, body }); return action === 'load' ? { loaded: true } : { text: '기회비용입니다.' }; }, close() {} } });
  assert.equal((await post('/api/session', { glossary: '기회비용' })).status, 200);
  assert.deepEqual(await (await post('/api/transcribe', { glossary: '기회비용', audio })).json(), { text: '기회비용입니다.' });
  assert.equal(calls[0].action, 'load'); assert.equal(calls[1].action, 'transcribe'); assert.equal(calls[1].body.audio, audio); assert.equal(calls[1].body.glossary, '기회비용');
});
test('worker setup failure is actionable and does not return fake transcription', async t => {
  const { post } = await serve(t, { transcriber: { request: async () => { throw Error('모델 설치 필요'); }, close() {} } });
  const response = await post('/api/session', { glossary: '' }); assert.equal(response.status, 503); assert.deepEqual(await response.json(), { error: '모델 설치 필요' });
});
test('Ollama receives rolling context at loopback only, without any API credentials', async t => {
  const { post } = await serve(t, { fetcher: async (url, options) => {
    assert.equal(url, 'http://127.0.0.1:11434/api/chat'); assert.equal(options.headers.Authorization, undefined);
    const body = JSON.parse(options.body); assert.equal(body.model, 'qwen2.5:7b'); assert.equal(body.stream, false);
    assert.deepEqual(JSON.parse(body.messages[1].content), { previous_summary: '이전 요약', new_transcript: '새 강의' });
    return Response.json({ message: { content: '누적 요약' } });
  } });
  const result = await (await post('/api/summary', { transcript: '새 강의', previous: '이전 요약' })).json(); assert.equal(result.summary, '누적 요약'); assert.equal(result.method, 'ollama');
});
test('missing Ollama produces explicitly labeled extractive notes', async () => {
  const text = '기회비용은 포기한 최선의 대안의 가치입니다. 매몰비용은 이미 지출한 비용입니다.';
  const result = await localSummary({ previous: '', transcript: text }, { env: {}, fetcher: async () => { throw Error('not running'); } });
  assert.equal(result.method, 'extractive'); assert.match(result.warning, /Ollama/); assert.match(result.summary, /기회비용/);
  assert.ok(result.summary.split('\n').every(line => text.includes(line.replace(/^• /, ''))));
});
test('extractive mode never calls a model; cloud model tags are blocked', async () => {
  const fetcher = () => { throw Error('must not be called'); };
  const input = { previous: '', transcript: '이것은 네트워크를 사용하지 않는 강의 요약 테스트입니다.' };
  assert.equal((await localSummary(input, { env: { SUMMARY_MODE: 'extractive' }, fetcher })).method, 'extractive');
  assert.equal((await localSummary(input, { env: { OLLAMA_MODEL: 'qwen3:cloud' }, fetcher })).method, 'extractive');
  assert.ok(extractiveSummary('', input.transcript).length);
});
test('PCM encoding downsamples 48 kHz to 16 kHz signed little endian', () => {
  const encoded = pcmBase64(new Float32Array(48000).fill(0.5), 48000); const bytes = Buffer.from(encoded, 'base64');
  assert.equal(bytes.length, 32000); assert.equal(bytes.readInt16LE(0), 16384);
  assert.equal(Buffer.from(pcmBase64(new Float32Array(16000).fill(-1), 16000), 'base64').readInt16LE(0), -32768);
});
test('speech queues its first preview at 350 ms and finalizes after a pause', () => {
  const chunks = [], segmenter = new Segmenter(16000, chunk => chunks.push(chunk));
  for (let i = 0; i < 18; i++) segmenter.push(new Float32Array(320).fill(0.1));
  assert.equal(chunks.length, 1); assert.equal(chunks[0].final, false);
  assert.ok(Buffer.from(chunks[0].audio, 'base64').length <= 16000 * 2 * 0.37);
  for (let i = 0; i < 25; i++) segmenter.push(new Float32Array(320));
  assert.equal(chunks.length, 2); assert.equal(chunks[1].final, true); assert.equal(chunks[1].id, chunks[0].id);
  for (let i = 0; i < 100; i++) segmenter.push(new Float32Array(320));
  segmenter.flush(); assert.equal(chunks.length, 2);
});
test('silence and isolated clicks create no transcription requests', () => {
  const chunks = [], segmenter = new Segmenter(16000, chunk => chunks.push(chunk));
  for (let i = 0; i < 1000; i++) segmenter.push(new Float32Array(320));
  segmenter.push(new Float32Array(320).fill(0.3));
  for (let i = 0; i < 100; i++) segmenter.push(new Float32Array(320));
  segmenter.flush(); assert.deepEqual(chunks, []);
  assert.ok(segmenter.preLength <= 3200);
});
test('continuous speech has bounded context and overlapping finalized sections', () => {
  const chunks = [], segmenter = new Segmenter(16000, chunk => chunks.push(chunk));
  for (let i = 0; i < 550; i++) segmenter.push(new Float32Array(320).fill(0.1));
  segmenter.flush(); const finals = chunks.filter(x => x.final);
  assert.equal(finals.length, 2); assert.equal(finals[0].seconds, 0); assert.equal(finals[1].seconds, 7.6); assert.equal(finals[1].overlap, true);
  assert.ok(chunks.every(x => Buffer.from(x.audio, 'base64').length <= 16000 * 2 * 8.02));
});
test('overlap deduplication removes repeated boundary words only', () => {
  assert.equal(removeOverlap('이것이 기회비용의 정의입니다', '기회비용의 정의입니다. 다음은 매몰비용입니다.'), '다음은 매몰비용입니다.');
  assert.equal(removeOverlap('경제학', '다음 강의입니다.'), '다음 강의입니다.');
});
test('queue preserves order and retains failed audio for retry without duplicating success', async () => {
  const seen = []; let fail = true;
  const queue = new AudioQueue(async item => { if (item === 2 && fail) throw Error('temporary failure'); seen.push(item); });
  queue.enqueue(1); queue.enqueue(2); queue.enqueue(3); await queue.settle();
  assert.deepEqual(seen, [1]); assert.deepEqual(queue.items, [2, 3]); assert.ok(queue.failed);
  fail = false; await queue.retry(); assert.deepEqual(seen, [1, 2, 3]); assert.equal(queue.items.length, 0); assert.equal(queue.failed, null);
});
test('queue enforces backlog bound rather than accumulating unbounded audio', async () => {
  let release; const queue = new AudioQueue(() => new Promise(resolve => { release = resolve; }), () => {}, 1);
  assert.equal(queue.enqueue(1), true); assert.equal(queue.enqueue(2), false); release(); await queue.settle();
});
test('pending transcript entries update in place and remain chronologically ordered', () => {
  const transcript = new Transcript(); transcript.set('b', { seconds: 3 }); transcript.set('a', { seconds: 0, text: '첫 문장', final: true }); transcript.set('b', { text: '둘째 문장', final: true });
  assert.equal(transcript.items.size, 2); assert.deepEqual(transcript.ordered().map(x => x.text), ['첫 문장', '둘째 문장']);
});
test('stop after a forced split does not retranscribe overlap alone', () => {
  const chunks = [], segmenter = new Segmenter(16000, chunk => chunks.push(chunk));
  for (let i = 0; i < 400; i++) segmenter.push(new Float32Array(320).fill(0.1));
  segmenter.flush(); assert.equal(chunks.filter(x => x.final).length, 1);
});
test('slow processing coalesces previews and replaces the last preview with final audio', async () => {
  const processed = []; let release;
  const queue = new AudioQueue(async item => {
    processed.push(item);
    if (processed.length === 1) await new Promise(resolve => { release = resolve; });
  });
  queue.enqueue({ id: 'one', final: false, text: 'a' });
  for (const text of ['ab', 'abc', 'abcd']) queue.enqueue({ id: 'one', final: false, text });
  assert.equal(queue.items.length, 2); assert.equal(queue.items[1].text, 'abcd');
  queue.enqueue({ id: 'one', final: true, text: 'corrected' });
  release(); await queue.settle();
  assert.deepEqual(processed.map(x => x.text), ['a', 'corrected']);
});
test('preview failure does not stop recording; full final audio is still processed', async () => {
  const processed = [];
  const queue = new AudioQueue(async item => { if (!item.final) throw Error('preview failure'); processed.push(item.id); });
  queue.enqueue({ id: 'one', final: false }); queue.enqueue({ id: 'one', final: true });
  await queue.settle(); assert.deepEqual(processed, ['one']); assert.equal(queue.failed, null);
});
test('full queue skips previews but retains final audio for retry', async () => {
  let release; const seen = [];
  const queue = new AudioQueue(async item => { seen.push(item.id); if (item.id === 'first') await new Promise(resolve => { release = resolve; }); }, () => {}, 2);
  queue.enqueue({ id: 'first', final: true }); queue.enqueue({ id: 'second', final: true });
  assert.equal(queue.enqueue({ id: 'preview', final: false }), true);
  assert.equal(queue.enqueue({ id: 'third', final: true }), false);
  assert.equal(queue.items.length, 2); release(); await queue.settle(); assert.deepEqual(seen, ['first', 'second']);
});
