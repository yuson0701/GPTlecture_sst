import { test } from 'node:test';
import assert from 'node:assert/strict';
import { sourceBatch, summaryDue, automaticSummaryBatch, usageTotals, blockMarkdown } from '../public/notes.js';
import { streamResponse } from '../vendor/siwc-local/dist/responses.js';
import { chatgptSummary } from '../local/summary.mjs';
import { publicChatGPTError } from '../local/chatgpt.mjs';
import { validateRecord } from '../local/records.mjs';

const usage = { input_tokens: 1200, output_tokens: 300, total_tokens: 1500 };
test('a delayed Whisper final cannot starve the deadline for completed text', () => {
  const items = [0, 8, 16].map(seconds => ({ seconds, text: '칼슘 채널과 심근 수축의 관계에 대한 강의 내용입니다. '.repeat(4) }));
  const audio = [{ seconds: 16, final: true }, { seconds: 24, final: false }];
  assert.deepEqual(automaticSummaryBatch(items, audio, 30, 30000, 0), [], 'pause must initially wait for its final');
  assert.deepEqual(automaticSummaryBatch(items, audio, 30, 90000, 0), items.slice(0, 2), 'send only the completed prefix at the deadline');
  assert.deepEqual(automaticSummaryBatch(items, [], 30, 30000, 0), items, 'drained pause sends the entire block');
  assert.deepEqual(automaticSummaryBatch(items, [{ seconds: 0, final: true }], 30, 90000, 0), [], 'do not summarize past unfinished audio');
  assert.deepEqual(automaticSummaryBatch(items, [{ seconds: 24, final: false }], 20, 30000, 0), items, 'later drafts do not hold earlier finals');
});

test('complete fenced summary JSON is accepted but broken responses remain failures', async () => {
  const sections = [{ title: '칼슘 조절', bullets: ['Troponin C에 칼슘이 결합합니다.'] }];
  const run = text => chatgptSummary({ transcript: '강의 원문', model: 'mock', consent: true }, {
    chatgpt: { respond: async () => ({ text, usage }) },
  });
  assert.deepEqual((await run('```json\n' + JSON.stringify({ sections }) + '\n```')).sections, sections);
  for (const text of ['```json\n{"sections": [', 'Here is your note: ' + JSON.stringify({ sections }), '{}']) {
    await assert.rejects(run(text), error => error.code === 'invalid_paragraph' && error.usage.total_tokens === 1500);
  }
});
test('automatic summaries prefer pauses and sentence endings, with a 90-second deadline', () => {
  const start = 100000, sentence = [{ seconds: 0, text: '강의 내용을 설명합니다. '.repeat(12) }];
  const fragment = [{ seconds: 0, text: '아직 설명이 이어지고 있어서 '.repeat(12) }];
  assert.equal(summaryDue(start + 20000, start, sentence), false);
  assert.equal(summaryDue(start + 59999, start, sentence), false);
  assert.equal(summaryDue(start + 60000, start, sentence), true);
  assert.equal(summaryDue(start + 60000, start, fragment), false);
  assert.equal(summaryDue(start + 89999, start, fragment), false);
  assert.equal(summaryDue(start + 90000, start, fragment), true);
  assert.equal(summaryDue(start + 10000, start, sentence, { paused: true }), true);
  assert.equal(summaryDue(start + 90000, start, [{ text: '네.' }], { paused: true }), false);
  assert.equal(summaryDue(start + 90000, start, []), false);
  // A long initial silence does not count as time spent explaining new content.
  assert.equal(summaryDue(start + 120000, start + 100000, sentence), false);
  const afterLongGap = [{ seconds: 0, text: '네.' }, { ...sentence[0], seconds: 300 }];
  assert.equal(sourceBatch(afterLongGap).length, 2, 'a tiny leftover must not block later lecture content');
  const items = Array.from({ length: 120 }, (_, seconds) => ({ seconds, text: '강의 문장입니다.' }));
  assert.equal(sourceBatch(items).length, 90);
  assert.equal(sourceBatch(items.slice(90)).length, 30);
  assert.equal(sourceBatch(items.map(x => ({ ...x, text: '가'.repeat(3000) }))).length, 2);
});

test('SSE response usage is returned only when reported counts are valid', async t => {
  for (const reported of [usage, undefined, { ...usage, total_tokens: 1 }, { ...usage, input_tokens: -2 }]) {
    t.mock.method(globalThis, 'fetch', async () => new Response(
      `data: ${JSON.stringify({ type: 'response.output_text.delta', delta: '정리' })}\n\n` +
      `data: ${JSON.stringify({ type: 'response.completed', response: { usage: reported } })}\n\n`,
      { headers: { 'Content-Type': 'text/event-stream' } }));
    const result = await streamResponse('synthetic-token', { model: 'mock', input: 'test' }, new AbortController().signal);
    assert.equal(result.text, '정리');
    assert.deepEqual(result.usage, reported === usage ? usage : undefined);
    t.mock.restoreAll();
  }
});

test('section headings and explanatory bullets survive parsing and Markdown export', async () => {
  const sections = [
    { title: '진단의 기초 과정', bullets: ['증상과 병력을 수집합니다.', '수집한 정보를 종합하여 질환을 추론합니다.'] },
    { title: '신체 진찰 실습의 목표', bullets: ['정상적인 신체 소리를 익히는 것이 목표입니다.'] },
  ];
  const result = await chatgptSummary({ transcript: '의학 강의', model: 'mock', consent: true }, { chatgpt: { respond: async () => ({ text: JSON.stringify({ sections }), usage }) } });
  assert.deepEqual(result.sections, sections); assert.deepEqual(result.usage, usage);
  const markdown = blockMarkdown(result);
  assert.match(markdown, /### 진단의 기초 과정/); assert.match(markdown, /### 신체 진찰 실습의 목표/);
  assert.match(markdown, /- 증상과 병력을 수집합니다./);
});

test('invalid paraphrases still report consumed tokens and retries add distinct attempts', async () => {
  let failure;
  try { await chatgptSummary({ transcript: '강의', model: 'mock', consent: true }, { chatgpt: { respond: async () => ({ text: '{}', usage }) } }); }
  catch (error) { failure = error; }
  assert.equal(failure.code, 'invalid_paragraph');
  assert.deepEqual(publicChatGPTError(failure).usage, usage);
  const blocks = [{ attempts: [usage, null, usage] }, { method: 'chatgpt' }, { method: 'demo' }];
  assert.deepEqual(usageTotals(blocks), { input_tokens: 2400, output_tokens: 600, total_tokens: 3000, unknown: 2, requests: 4 });
});

test('saved lectures preserve sections, retries, unknown usage and old records', () => {
  const record = { revision: 0, title: '강의', glossary: '', elapsed: 90, state: 'idle', transcript: [], summarized: [], blocks: [{ id: 1, source: '강의 원문', state: 'done', method: 'chatgpt', items: [{ seconds: 0 }], sections: [{ title: '주제', bullets: ['설명입니다.'] }], attempts: [usage, null, usage] }] };
  const restored = validateRecord(JSON.parse(JSON.stringify(record)));
  assert.deepEqual(restored.blocks[0].sections, record.blocks[0].sections);
  assert.deepEqual(usageTotals(restored.blocks), usageTotals(record.blocks));
  delete record.blocks[0].sections; delete record.blocks[0].attempts;
  assert.equal(usageTotals(validateRecord(record).blocks).unknown, 1);
  record.blocks[0].attempts = [{ ...usage, total_tokens: -1 }];
  assert.throws(() => validateRecord(record), /token usage/);
});

test('incomplete responses retain reported usage without returning partial notes', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response(`data: ${JSON.stringify({ type: 'response.incomplete', response: { usage } })}\n\n`, { headers: { 'Content-Type': 'text/event-stream' } }));
  await assert.rejects(streamResponse('synthetic', { model: 'mock', input: 'test' }, new AbortController().signal), error => error.code === 'response_incomplete' && error.usage.total_tokens === 1500);
});
