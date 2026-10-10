export const PARAPHRASE_SECONDS = 90;

export function reportedUsage(value) {
  if (!value || !['input_tokens', 'output_tokens', 'total_tokens'].every(k => Number.isSafeInteger(value[k]) && value[k] >= 0) || value.input_tokens + value.output_tokens !== value.total_tokens) return null;
  return { input_tokens: value.input_tokens, output_tokens: value.output_tokens, total_tokens: value.total_tokens };
}
export function usageTotals(blocks) {
  const totals = { input_tokens: 0, output_tokens: 0, total_tokens: 0, unknown: 0, requests: 0 };
  for (const block of blocks) for (const attempt of block.attempts || (block.method === 'chatgpt' ? [null] : [])) {
    totals.requests++;
    const usage = reportedUsage(attempt);
    if (!usage) totals.unknown++;
    else for (const key of ['input_tokens', 'output_tokens', 'total_tokens']) totals[key] += usage[key];
  }
  return totals;
}
export function sourceBatch(items) {
  const batch = []; let length = 0, characters = 0;
  for (const item of items) {
    if (batch.length && (length + item.text.length + 20 > 6400 || (characters >= 80 && item.seconds - batch[0].seconds >= PARAPHRASE_SECONDS))) break;
    batch.push(item); length += item.text.length + 20; characters += item.text.replace(/\s/g, '').length;
  }
  return batch;
}
export function summaryDue(now, firstPendingAt, items, { paused = false } = {}) {
  if (!items.length || items.reduce((n, item) => n + item.text.replace(/\s/g, '').length, 0) < 80) return false;
  const age = now - firstPendingAt;
  const sentenceEnd = /[.!?。！？][”’"')\]]*$/.test(items.at(-1).text.trim());
  return paused || age >= PARAPHRASE_SECONDS * 1000 || (age >= 60000 && sentenceEnd);
}

export function validSections(sections) {
  const text = (v, max) => typeof v === 'string' && v.trim().length > 0 && v.length <= max;
  return Array.isArray(sections) && sections.length > 0 && sections.length <= 5 && sections.every(s => s && text(s.title, 120) && Array.isArray(s.bullets) && s.bullets.length > 0 && s.bullets.length <= 6 && s.bullets.every(b => text(b, 1500))) && JSON.stringify(sections).length <= 10000;
}
export function blockMarkdown(block) {
  return block.sections?.map(s => `### ${s.title}\n\n${s.bullets.map(b => `- ${b}`).join('\n')}`).join('\n\n') || `${block.title ? `### ${block.title}\n\n` : ''}${block.text || '(아직 처리되지 않음)'}`;
}
