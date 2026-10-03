import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createApp } from '../server.mjs';
import { chatgptSummary } from '../local/summary.mjs';
import { createChatGPTService, publicChatGPTError } from '../local/chatgpt.mjs';
import { keychainEncryption } from '../local/credential-encryption.mjs';
import { ChatGPTError } from '../vendor/siwc-local/dist/index.js';

const paragraph = { title: '기회비용', bullets: ['포기한 최선의 대안의 가치입니다.'], cleaned: '기회비용은 포기한 최선의 대안의 가치입니다.' };
const input = { previous: '이전 문단은 보내지 마세요', block: true, paragraph: true, transcript: '기회 비용은 포기한 최선의 대안의 가치입니다.', consent: true, model: 'available-model' };
const connected = { status: 'connected', sharing: true, identity: { email: 'test@example.com' } };
const fakeClient = overrides => ({
  getSession: async () => connected, signIn: async () => connected, cancelSignIn() {}, disconnect: async () => {},
  listModels: async () => [{ slug: input.model, displayName: 'Available model' }],
  streamResponse: async () => ({ text: JSON.stringify(paragraph) }), ...overrides,
});
async function serve(t, client = fakeClient()) {
  const chatgpt = createChatGPTService({ client });
  const server = createApp({ env: { SUMMARY_MODE: 'ollama', OLLAMA_MODEL: 'old-model' }, chatgpt,
    transcriber: { request: async () => ({ available: true }), close() {} } });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  const base = `http://127.0.0.1:${server.address().port}`;
  return { base,
    post: (path, body = {}, origin = base) => fetch(base + path, { method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
    get: path => fetch(base + path, { headers: { 'X-Lecture-Client': '1' } }),
  };
}

test('explicit consent and a catalog model are required before any text leaves', async t => {
  let calls = 0;
  const { post } = await serve(t, fakeClient({ streamResponse: async () => { calls++; return { text: JSON.stringify(paragraph) }; } }));
  assert.equal((await post('/api/summary', { ...input, consent: false })).status, 400);
  assert.equal((await post('/api/summary', { ...input, consent: undefined })).status, 400);
  assert.equal((await post('/api/summary', { ...input, model: 'made-up-model' })).status, 503);
  assert.equal(calls, 0);
  const response = await post('/api/summary', input);
  assert.equal(response.status, 200); assert.equal((await response.json()).method, 'chatgpt'); assert.equal(calls, 1);
});

test('one source block only reaches ChatGPT; old Ollama configuration is ignored', async t => {
  const { post } = await serve(t, fakeClient({ streamResponse: async options => {
    assert.deepEqual(options.input, [{ role: 'user', content: JSON.stringify({ transcript: input.transcript }) }]);
    assert.match(options.instructions, /추측해서 고치지/);
    assert.equal(options.model, input.model); assert.ok(options.signal instanceof AbortSignal);
    assert.equal(options.audio, undefined); assert.equal(options.previous, undefined);
    return { text: JSON.stringify(paragraph) };
  } }));
  const result = await (await post('/api/summary', { ...input, audio: 'not-sent', glossary: 'not-sent' })).json();
  assert.equal(result.cleaned, paragraph.cleaned); assert.equal(result.title, paragraph.title);
  assert.equal(result.summary, '• ' + paragraph.bullets[0]);
});

test('auth routes reject cross-origin state changes and untrusted status requests', async t => {
  let signedIn = 0;
  const { post, get, base } = await serve(t, fakeClient({ signIn: async () => { signedIn++; return connected; } }));
  assert.equal((await post('/api/chatgpt/sign-in', {}, 'https://evil.example')).status, 403);
  assert.equal((await post('/api/chatgpt/disconnect', {}, 'https://evil.example')).status, 403);
  assert.equal((await fetch(base + '/api/chatgpt/status')).status, 403);
  assert.equal(signedIn, 0);
  assert.equal((await post('/api/chatgpt/sign-in')).status, 202);
  const state = await (await get('/api/chatgpt/status')).json();
  assert.equal(state.identity.email, connected.identity.email);
  assert.equal(signedIn, 1);
});

test('eligibility and usage errors return actionable failures, never extracted success', async t => {
  const { post } = await serve(t, fakeClient({ streamResponse: async () => { throw new ChatGPTError('subscription_sharing_usage_limit_exceeded', 'limit'); } }));
  const response = await post('/api/summary', input), body = await response.json();
  assert.equal(response.status, 503); assert.match(body.error, /한도/);
  assert.equal(body.summary, undefined); assert.equal(body.method, undefined);
});

test('malformed, empty, oversized and incomplete model outputs remain failures', async () => {
  for (const text of ['not json', 'null', '{}', JSON.stringify({ ...paragraph, bullets: [] }), JSON.stringify({ ...paragraph, cleaned: ' ' }), JSON.stringify({ ...paragraph, cleaned: 'x'.repeat(10001) })]) {
    await assert.rejects(chatgptSummary(input, { chatgpt: { respond: async () => ({ text }) } }), { code: 'invalid_paragraph' });
  }
  await assert.rejects(chatgptSummary(input, { chatgpt: { respond: async () => { throw new ChatGPTError('stream_interrupted', 'Interrupted'); } } }), { code: 'stream_interrupted' });
  assert.doesNotMatch(JSON.stringify(publicChatGPTError(Error('secret-token'))), /secret-token/);
});

test('login cancellation, reconsent and disconnect use supported SDK operations', async () => {
  let finish, consent, cancelled = 0, disconnected = 0;
  const service = createChatGPTService({ client: fakeClient({
    signIn: options => { consent = options.reconsent; return new Promise(resolve => { finish = resolve; }); },
    cancelSignIn: () => { cancelled++; finish?.(connected); },
    disconnect: async () => { disconnected++; },
  }) });
  service.startSignIn({ reconsent: true }); await Promise.resolve();
  assert.equal((await service.status()).status, 'connecting'); assert.equal(consent, true);
  await service.cancel(); assert.equal(cancelled, 1);
  await service.disconnect(); assert.equal(disconnected, 1);
});

test('credentials are authenticated ciphertext and survive a new provider instance', async () => {
  let stored, writes = 0;
  const loadEntry = async () => ({ getSecret: async () => stored, setSecret: async value => { writes++; stored = value; } });
  const provider = () => keychainEncryption({ platform: 'darwin', loadEntry });
  const encrypted = await provider().encrypt('synthetic-access-and-refresh-tokens');
  assert.ok(!Buffer.from(encrypted).includes(Buffer.from('synthetic')));
  assert.equal(await provider().decrypt(encrypted), 'synthetic-access-and-refresh-tokens');
  assert.equal(writes, 1);
  const tampered = Buffer.from(encrypted); tampered[15] ^= 1;
  await assert.rejects(provider().decrypt(tampered));
  stored = undefined;
  await assert.rejects(provider().decrypt(encrypted)); assert.equal(writes, 1, 'missing key must not overwrite saved credentials');
  const locked = keychainEncryption({ platform: 'darwin', loadEntry: async () => ({ getSecret: async () => { throw Error('locked'); }, setSecret: async () => { writes++; } }) });
  await assert.rejects(locked.encrypt('secret')); assert.equal(writes, 1);
  assert.equal(await keychainEncryption({ platform: 'linux' }).isAvailable(), false);
});

test('concurrent summaries are bounded and client disconnect aborts the model request', async t => {
  let modelSignal, resolveStarted;
  const started = new Promise(resolve => { resolveStarted = resolve; });
  const { post, base } = await serve(t, fakeClient({ streamResponse: options => new Promise((resolve, reject) => {
    modelSignal = options.signal; resolveStarted();
    modelSignal.addEventListener('abort', () => reject(new ChatGPTError('cancelled', 'Cancelled')), { once: true });
  }) }));
  const controller = new AbortController();
  const first = fetch(base + '/api/summary', { method: 'POST', headers: { Origin: base, 'Content-Type': 'application/json' }, body: JSON.stringify(input), signal: controller.signal }).catch(() => {});
  await started;
  assert.equal((await post('/api/summary', input)).status, 429);
  controller.abort(); await first;
  await new Promise(resolve => setTimeout(resolve, 50));
  assert.equal(modelSignal.aborted, true);
});
