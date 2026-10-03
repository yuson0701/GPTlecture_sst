export const PARAPHRASE_SECONDS = 75;

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
  const batch = []; let length = 0;
  for (const item of items) {
    if (batch.length && (length + item.text.length + 20 > 6400 || item.seconds - batch[0].seconds >= PARAPHRASE_SECONDS)) break;
    batch.push(item); length += item.text.length + 20;
  }
  return batch;
}
export function summaryDue(now, lastRequest) { return now - lastRequest >= PARAPHRASE_SECONDS * 1000; }

export function validSections(sections) {
  const text = (v, max) => typeof v === 'string' && v.trim().length > 0 && v.length <= max;
  return Array.isArray(sections) && sections.length > 0 && sections.length <= 5 && sections.every(s => s && text(s.title, 120) && Array.isArray(s.bullets) && s.bullets.length > 0 && s.bullets.length <= 6 && s.bullets.every(b => text(b, 1500))) && JSON.stringify(sections).length <= 10000;
}
export function blockMarkdown(block) {
  return block.sections?.map(s => `### ${s.title}\n\n${s.bullets.map(b => `- ${b}`).join('\n')}`).join('\n\n') || `${block.title ? `### ${block.title}\n\n` : ''}${block.text || '(아직 처리되지 않음)'}`;
}
