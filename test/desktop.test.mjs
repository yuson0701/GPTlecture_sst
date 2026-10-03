import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { runtimePaths, runtimeEnv, trustedPage, usableModel } from '../desktop/runtime.mjs';
test('packaged desktop uses bundled Python and durable user data, never the app directory', () => {
  const paths = runtimePaths({ packaged: true, resources: '/Applications/Lecture Note.app/Contents/Resources', root: '/Applications/Lecture Note.app/Contents/Resources/app', userData: '/Users/example/Library/Application Support/Lecture Note' });
  const env = runtimeEnv(paths, { base: '/cache/base', adapter: '/cache/adapter' }, {});
  assert.equal(env.PYTHON_BIN, '/Applications/Lecture Note.app/Contents/Resources/python/bin/python3.14');
  assert.equal(env.LECTURE_DATA_DIR, '/Users/example/Library/Application Support/Lecture Note/lectures');
  assert.equal(env.STT_BACKEND, 'qwen-mlx'); assert.equal(env.QWEN_ASR_ADAPTER, '/cache/adapter');
  assert.equal(env.PYTHONNOUSERSITE, '1');
});
test('privileged desktop IPC accepts only the app origin or exact setup page', () => {
  const origin = 'http://127.0.0.1:34345', setup = 'file:///app/desktop/setup.html';
  assert.equal(trustedPage(origin + '/', origin, setup), true);
  assert.equal(trustedPage(setup, origin, setup), true);
  for (const url of ['https://evil.example', 'http://127.0.0.1:34346', 'file:///tmp/evil.html', 'not a url']) assert.equal(trustedPage(url, origin, setup), false);
  assert.equal(trustedPage('file:///tmp/evil.html', '', setup), false);
});
test('missing model files route first launch back to setup', async t => {
  const root = await mkdtemp(join(tmpdir(), 'lecture-model-test-')); t.after(() => rm(root, { recursive: true, force: true }));
  const config = { schema: 1, base: join(root, 'base'), adapter: join(root, 'adapter') };
  assert.equal(Boolean(usableModel(config)), false);
  await mkdir(config.base); await mkdir(config.adapter);
  await writeFile(join(config.base, 'config.json'), '{}'); await writeFile(join(config.adapter, 'selected.safetensors'), 'fixture');
  assert.equal(usableModel(config), true); assert.equal(Boolean(usableModel({})), false);
});
