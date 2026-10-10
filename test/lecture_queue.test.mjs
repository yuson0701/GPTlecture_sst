import { test } from 'node:test';
import assert from 'node:assert/strict';
import { LectureQueue } from '../public/lecture-queue.js';

class MemorySpool {
  constructor() { this.rows = new Map(); this.quota = false; }
  async put(item) { if (this.quota) throw Error('quota'); this.rows.set(item.id, { ...item }); }
  async get(id) { return this.rows.get(id); }
  async delete(id) { this.rows.delete(id); }
  async list() { return [...this.rows.values()].map(({ audio, ...meta }) => meta); }
}
const clip = i => ({ id: String(i), seconds: i * 6, final: true, audio: 'pcm-' + i });

test('a repetition failure preserves audio but does not block the next final', async () => {
  const storage = new MemorySpool(), seen = []; let bad = true;
  const queue = new LectureQueue(async item => {
    if (item.id === '1' && bad) throw Error('repetition/token limit');
    seen.push(item.id);
  }, () => {}, storage);
  for (let i = 0; i < 4; i++) assert.equal(queue.enqueue(clip(i)), true);
  await queue.settle();
  assert.deepEqual(seen, ['0', '2', '3']);
  assert.equal(queue.failed, null); assert.equal(queue.failures.length, 1);
  assert.deepEqual([...storage.rows.keys()], ['1']);
  bad = false; await queue.retry();
  assert.deepEqual(seen, ['0', '2', '3', '1']);
  assert.equal(storage.rows.size, 0); assert.equal(queue.failures.length, 0);
});

test('three hours of clip arrivals survive a stalled request without an audio-memory backlog', async () => {
  const storage = new MemorySpool(), seen = [];
  let release, entered;
  const started = new Promise(resolve => { entered = resolve; });
  const queue = new LectureQueue(async item => {
    if (item.id === '0') { entered(); await new Promise(resolve => { release = resolve; }); }
    seen.push(item.id);
  }, () => {}, storage);
  queue.enqueue(clip(0)); await started;
  for (let i = 1; i < 1800; i++) {
    queue.enqueue({ ...clip(i), final: false });
    assert.equal(queue.enqueue(clip(i)), true);
    await queue.writes;
  }
  assert.equal(queue.items.length, 1800);
  assert.ok(queue.items.every(item => !('audio' in item)));
  assert.equal(storage.rows.size, 1800);
  release(); await queue.settle();
  assert.deepEqual(seen, Array.from({ length: 1800 }, (_, i) => String(i)));
  assert.equal(storage.rows.size, 0); assert.equal(queue.failed, null);
});

test('reopening a lecture restores failed or interrupted clips for explicit retry', async () => {
  const storage = new MemorySpool();
  await storage.put(clip(5)); await storage.put(clip(6));
  const seen = [], queue = new LectureQueue(async item => seen.push(item.audio), () => {}, storage);
  await queue.restore();
  assert.equal(queue.failures.length, 2);
  assert.equal(seen.length, 0);
  await queue.retry();
  assert.deepEqual(seen, ['pcm-5', 'pcm-6']); assert.equal(storage.rows.size, 0);
});

test('settle waits for an uncommitted final and storage failure retains its audio', async () => {
  const storage = new MemorySpool(); storage.quota = true;
  const seen = [], queue = new LectureQueue(async item => seen.push(item.id), () => {}, storage);
  queue.enqueue(clip(0)); await queue.settle();
  assert.ok(queue.failed); assert.equal(queue.items[0].audio, 'pcm-0');
  storage.quota = false; await queue.retry();
  assert.deepEqual(seen, ['0']); assert.equal(queue.items.length, 0);
});

test('failed transcript save keeps audio until a successful retry', async () => {
  const storage = new MemorySpool(); let saveFails = true;
  const queue = new LectureQueue(async () => { if (saveFails) throw Error('record save failed'); }, () => {}, storage);
  queue.enqueue(clip(0)); await queue.settle();
  assert.equal(storage.rows.size, 1);
  saveFails = false; await queue.retry(); assert.equal(storage.rows.size, 0);
});

test('draft failures are disposable, and a failed final is tried only once per retry', async () => {
  const storage = new MemorySpool(); let attempts = 0;
  const queue = new LectureQueue(async () => { attempts++; throw Error('decode failed'); }, () => {}, storage);
  queue.enqueue({ ...clip(0), final: false }); await queue.settle();
  assert.equal(queue.failures.length, 0);
  queue.enqueue(clip(0)); queue.enqueue(clip(1)); await queue.settle();
  assert.equal(attempts, 3); assert.equal(queue.failures.length, 2);
  await queue.retry();
  assert.equal(attempts, 5); assert.equal(queue.failures.length, 2);
  assert.equal(queue.failed, null);
});

test('a quarantined failure does not suppress previews of later speech', async () => {
  const storage = new MemorySpool(), drafts = [];
  const queue = new LectureQueue(async item => {
    if (item.final) throw Error('bad clip');
    drafts.push(item.id);
  }, () => {}, storage);
  queue.enqueue(clip(0)); await queue.settle();
  queue.enqueue({ ...clip(1), final: false }); await queue.settle();
  assert.deepEqual(drafts, ['1']);
  assert.equal(queue.failures.length, 1);
});
