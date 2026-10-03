import { test } from 'node:test';
import assert from 'node:assert/strict';
import { AudioQueue, SpeechActivity } from '../public/audio.js';

test('five captured seconds of silence trigger once after speech, not on startup or clicks', () => {
  for (const rate of [16000, 48000]) {
    const pauses = [], detector = new SpeechActivity(rate, pause => pauses.push(pause));
    const feed = (seconds, level) => { for (let i = 0; i < Math.round(seconds * 50); i++) detector.push(new Float32Array(rate / 50).fill(level)); };
    feed(10, 0); feed(0.02, 0.2); feed(6, 0);
    assert.equal(pauses.length, 0);
    feed(0.5, 0.1); feed(4.98, 0); assert.equal(pauses.length, 0);
    feed(0.02, 0); assert.equal(pauses.length, 1);
    assert.ok(pauses[0].throughSeconds > 20);
    feed(20, 0); assert.equal(pauses.length, 1);
    feed(0.5, 0.1); feed(4, 0); feed(0.5, 0.1); feed(4, 0);
    assert.equal(pauses.length, 1, 'speech resets the silence interval');
    feed(1, 0); assert.equal(pauses.length, 2);
  }
});

test('brief finalized backlog keeps audio in order and skips drafts until finals drain', async () => {
  let release; const processed = [];
  const queue = new AudioQueue(async item => {
    processed.push(item.id);
    if (item.id === 'first') await new Promise(resolve => { release = resolve; });
  });
  queue.enqueue({ id: 'first', final: true });
  assert.equal(queue.waiting.length, 0, 'in-flight audio is not waiting');
  for (let i = 0; i < 12; i++) {
    assert.equal(queue.enqueue({ id: `draft-${i}`, final: false }), true);
    assert.equal(queue.enqueue({ id: `final-${i}`, final: true }), true);
  }
  assert.equal(queue.waiting.length, 12); assert.equal(queue.failed, null);
  assert.ok(queue.waiting.every(item => item.final));
  release(); await queue.settle();
  assert.deepEqual(processed, ['first', ...Array.from({ length: 12 }, (_, i) => `final-${i}`)]);
  assert.equal(queue.waiting.length, 0);
  queue.enqueue({ id: 'recovered-preview', final: false }); await queue.settle();
  assert.equal(processed.at(-1), 'recovered-preview');
});

test('final audio removes unrelated queued drafts without cancelling the in-flight request', async () => {
  let release; const seen = [];
  const queue = new AudioQueue(async item => {
    seen.push(item.id);
    if (seen.length === 1) await new Promise(resolve => { release = resolve; });
  });
  queue.enqueue({ id: 'active-draft', final: false });
  queue.enqueue({ id: 'old-draft', final: false });
  queue.enqueue({ id: 'final', final: true });
  assert.deepEqual(queue.items.map(x => x.id), ['active-draft', 'final']);
  release(); await queue.settle();
  assert.deepEqual(seen, ['active-draft', 'final']);
});

test('audio buffering remains bounded even when individual items are large', async () => {
  let release; const queue = new AudioQueue(() => new Promise(resolve => { release = resolve; }));
  assert.equal(queue.enqueue({ id: 'first', final: true, audio: 'A'.repeat(2000000) }), true);
  assert.equal(queue.enqueue({ id: 'second', final: true, audio: 'A'.repeat(2000000) }), false);
  assert.equal(queue.items.length, 1);
  release(); await queue.settle();
});
