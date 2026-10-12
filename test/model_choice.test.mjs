import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ModelTranscriber } from '../local/model-choice.mjs';

test('switching waits for Qwen exit before loading Whisper and keeps adapter configuration', async () => {
  const calls = []; let release;
  const manager = new ModelTranscriber({ STT_BACKEND: 'qwen-mlx', PYTHON_BIN: '/custom/qwen', QWEN_ASR_ADAPTER: '/custom/adapter', WHISPER_PYTHON_BIN: '/custom/whisper' }, {
    platform: 'darwin', factory: env => ({
      request: async (action, data) => { calls.push({ action, env, data }); return { text: 'ECG' }; },
      shutdown: async () => { calls.push({ exit: env.STT_BACKEND }); if (env.STT_BACKEND === 'qwen-mlx') await new Promise(resolve => { release = resolve; }); },
      close() {},
    }),
  });
  await manager.request('load', { sttModel: 'qwen' });
  const switched = manager.request('load', { sttModel: 'whisper' });
  assert.equal(calls.length, 2); assert.equal(calls[1].exit, 'qwen-mlx');
  await assert.rejects(manager.request('load', { sttModel: 'qwen' }), /처리 중/);
  release(); await switched;
  assert.equal(calls[0].env.PYTHON_BIN, '/custom/qwen');
  assert.equal(calls[0].env.QWEN_ASR_ADAPTER, '/custom/adapter');
  assert.equal(calls[2].env.PYTHON_BIN, '/custom/whisper');
  assert.equal(calls[2].env.STT_BACKEND, 'mlx');
  await manager.request('transcribe', { sttModel: 'whisper', language: 'auto', glossary: 'ECG' });
  assert.equal(calls.length, 4, 'same model is reused');
  assert.equal(calls[3].data.glossary, 'ECG'); assert.equal(calls[3].data.language, 'auto');
});

test('readiness probes never load models or replace an active worker', async () => {
  let loads = 0, closes = 0;
  const manager = new ModelTranscriber({ STT_BACKEND: 'qwen-mlx' }, { platform: 'darwin', factory: env => ({
    request: async action => { if (action === 'load') loads++; return { available: env.STT_BACKEND === 'qwen-mlx' }; },
    shutdown: async () => { closes++; }, close() {},
  }) });
  await manager.request('load'); const active = manager.active;
  const models = await manager.catalog();
  assert.equal(loads, 1); assert.equal(closes, 2); assert.equal(manager.active, active);
  assert.equal(models[0].available, true); assert.equal(models[1].available, false);
  assert.equal(models[1].setup, 'npm run setup:whisper');
  assert.ok(models.every(x => !('python' in x)));
  await assert.rejects(manager.request('load', { sttModel: '../../shell' }), /지원하지/);
});

test('failed loads close the selected worker, do not fall back, and can retry', async () => {
  const requests = []; let fail = true, closed = 0;
  const manager = new ModelTranscriber({}, { factory: env => ({
    request: async () => { requests.push(env.STT_BACKEND); if (fail) throw Error('missing weights'); return { loaded: true }; },
    shutdown: async () => { closed++; }, close() {},
  }) });
  await assert.rejects(manager.request('load', { sttModel: 'qwen' }), /missing weights/);
  assert.equal(closed, 1); assert.equal(manager.active, null);
  fail = false; await manager.request('load', { sttModel: 'qwen' });
  assert.deepEqual(requests, ['qwen-mlx', 'qwen-mlx']);
});

test('a recoverable decode failure keeps the loaded model for the next audio clip', async () => {
  let instances = 0, closed = 0, fail = true;
  const manager = new ModelTranscriber({}, { factory: () => {
    instances++;
    return { child: {}, close() {}, shutdown: async () => { closed++; },
      request: async action => { if (action === 'transcribe' && fail) throw Error('repetition'); return { text: 'ECG' }; } };
  } });
  await manager.request('load', { sttModel: 'qwen' });
  await assert.rejects(manager.request('transcribe', { sttModel: 'qwen' }), /repetition/);
  fail = false;
  assert.deepEqual(await manager.request('transcribe', { sttModel: 'qwen' }), { text: 'ECG' });
  assert.equal(instances, 1); assert.equal(closed, 0);
});
