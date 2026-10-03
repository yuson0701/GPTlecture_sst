import { sourceBatch, summaryDue, reportedUsage, usageTotals, blockMarkdown } from './notes.js';
import { initChatGPT, summaryOptions, summaryReady, pauseChatGPT } from './chatgpt.js';
import { Transcript, timestamp } from './transcript.js';
import { Microphone, AudioQueue, removeOverlap, Segmenter } from './audio.js';
import { StreamingFrames, StreamingQueue } from './streaming.js';
const $ = id => document.getElementById(id);
let transcript = new Transcript(), microphone, queue, timer, demoTimer, mode = 'idle', started = 0, elapsed = 0, summaryBlocks = [], configured = false, summarizing = false;
const summarized = new Set();
let lastSummaryAt = 0;
let qwen = false, streaming = false, streamingRows = new Map(), finalRows = 0, streamSession = null;
const demoLines = ['오늘은 경제학의 기본 개념인 기회비용에 대해 알아보겠습니다. 기회비용은 어떤 선택을 했을 때 포기한 대안 중 가장 가치 있는 것의 가치입니다.', '예를 들어 두 시간 동안 아르바이트를 하면 2만 원을 벌 수 있지만, 그 시간에 시험 공부를 선택했다면 포기한 2만 원이 기회비용에 포함됩니다.', '여기서 중요한 것은 모든 대안의 가치를 더하는 것이 아니라, 포기한 대안 중 가장 좋은 하나만 고려한다는 점입니다.', '이미 지출해서 회수할 수 없는 비용은 매몰비용이라고 합니다. 합리적인 의사결정에서는 매몰비용보다 앞으로 발생할 비용과 편익을 비교해야 합니다.'];
const demoParaphrases = ['기회비용은 선택 때문에 포기한 대안 중 가장 가치 있는 하나를 뜻합니다.', '공부 때문에 아르바이트를 하지 못했다면, 벌 수 있었던 2만 원이 기회비용에 포함됩니다.', '기회비용은 포기한 모든 대안의 합이 아니라, 가장 좋은 대안 하나의 가치입니다.', '매몰비용은 이미 써서 돌려받을 수 없는 돈입니다. 선택할 때는 앞으로의 비용과 이익을 비교해야 합니다.'];
let recordId = null, revision = 0, changeVersion = 0, savedVersion = 0, saving = null, restoring = false, recordLoaded = false, uiBusy = false;
function changed() { if (recordId && !restoring) { changeVersion++; $('save-status').textContent = '저장 대기 중'; } }
function snapshot() { return { revision, title: $('title').value, glossary: $('glossary').value, elapsed, state: mode, transcript: transcript.ordered(), blocks: summaryBlocks, summarized: [...summarized] }; }
async function saveRecord() {
  if (saving) return saving;
  if (!recordId || changeVersion === savedVersion) return true;
  const id = recordId, version = changeVersion, data = snapshot();
  $('save-status').textContent = '저장 중…';
  saving = (async () => {
    try {
      const result = await api(`/api/records/${id}`, data);
      revision = result.revision; savedVersion = version;
      $('save-status').textContent = savedVersion === changeVersion ? '이 컴퓨터에 저장됨' : '저장 대기 중';
      $('save-retry').hidden = true; return true;
    } catch (e) {
      $('save-status').textContent = `저장 실패 · ${e.message}`; $('save-retry').hidden = false; return false;
    } finally { saving = null; }
  })();
  return saving;
}
async function flushRecord() {
  if (saving && !await saving) return false;
  while (recordId && savedVersion !== changeVersion) if (!await saveRecord()) return false;
  return true;
}
async function createRecord() {
  if (!recordId) { recordId = crypto.randomUUID(); revision = 0; changeVersion = 1; savedVersion = 0; }
  return flushRecord();
}
async function readRecords(path) {
  const response = await fetch(path, { signal: AbortSignal.timeout(15000) });
  const data = await response.json(); if (!response.ok) throw Error(data.error || '기록을 불러오지 못했습니다.'); return data;
}
function showWorkspace(show) { document.body.classList.toggle('in-lecture', show); $('workspace').hidden = !show; $('library').hidden = show; $('new-from-note').hidden = !show; }
async function refreshLibrary() {
  $('library-message').textContent = '기록을 불러오는 중…';
  try {
    const { records } = await readRecords('/api/records');
    $('recent-records').replaceChildren();
    for (const record of records) {
      const card = document.createElement('button'); card.className = 'record-card';
      const heading = document.createElement('h3'); heading.textContent = record.title || '제목 없는 강의';
      const meta = document.createElement('span'); meta.textContent = `${new Date(record.updatedAt).toLocaleString('ko-KR')} · ${timestamp(record.elapsed)} · ${record.segments}개 구간${record.state !== 'idle' ? ' · 진행 중이거나 중단된 기록' : ''}`;
      const preview = document.createElement('p'); preview.textContent = record.preview || '아직 전사된 내용이 없습니다.';
      card.append(heading, meta, preview); card.onclick = () => void runAction(() => openRecord(record.id)); $('recent-records').append(card);
    }
    $('library-message').textContent = records.length ? `${records.length}개의 강의가 저장되어 있습니다.` : '아직 저장된 강의가 없습니다. 새 강의를 시작해 보세요.';
  } catch (e) { $('library-message').textContent = e.message; }
}
async function openRecord(id) {
  if (mode !== 'idle' || summarizing || !await flushRecord()) return;
  try {
    const data = await readRecords(`/api/records/${id}`);
    restoring = true; reset(); recordId = id; revision = data.revision; changeVersion = savedVersion = 0; recordLoaded = true;
    $('lecture-date').textContent = new Date(data.createdAt).toLocaleDateString('ko-KR');
    $('title').value = data.title; $('glossary').value = data.glossary; elapsed = data.elapsed;
    data.transcript.forEach(item => transcript.set(item.id, item)); data.summarized.forEach(id => summarized.add(id));
    summaryBlocks = data.blocks.map(block => block.state === 'pending' ? { ...block, state: 'failed', warning: '정리 도중 종료된 문단입니다. 지금 정리로 다시 시도하세요.' } : block);
    render(); summaryBlocks.forEach(renderSummary); $('timer').textContent = timestamp(elapsed);
    $('status').textContent = '저장된 강의'; $('save-status').textContent = '이 컴퓨터에 저장됨';
    $('notice').textContent = '저장된 기록입니다. 새 녹음은 새 강의에서 시작하세요. 미확정 초안은 그대로 표시됩니다. 음성은 저장되지 않아 다시 인식할 수 없습니다.';
    showWorkspace(true); controls();
  } catch (e) { $('library-message').textContent = e.message; }
  finally { restoring = false; }
}
async function newLecture() {
  if (mode !== 'idle' || summarizing || !canReset() || !await flushRecord()) return false;
  recordId = null; revision = 0; changeVersion = savedVersion = 0; recordLoaded = false;
  reset(); $('lecture-date').textContent = new Date().toLocaleDateString('ko-KR'); $('title').value = ''; $('glossary').value = ''; $('save-status').textContent = '녹음을 시작하면 자동 저장됩니다';
  $('save-retry').hidden = true; $('status').textContent = configured ? '시작할 준비가 됐어요' : '로컬 음성 모델 설치 필요';
  showWorkspace(true); controls(); $('title').focus(); return true;
}
$('new-lecture').onclick = () => newLecture();
$('new-from-note').onclick = () => newLecture();
$('home').onclick = async () => { if (mode !== 'idle' || summarizing || !canReset() || !await flushRecord()) return; showWorkspace(false); await refreshLibrary(); };
$('save-retry').onclick = () => void flushRecord();
$('title').addEventListener('input', changed); $('glossary').addEventListener('input', changed);
setInterval(() => { if (recordId && mode !== 'idle') changed(); void saveRecord(); }, 2000);
function error(message) { if (message) { $('settings-panel').hidden = false; $('settings-toggle').setAttribute('aria-expanded', 'true'); } $('error').textContent = message; $('error').hidden = !message; }
function updateLive() {
  const pending = transcript.ordered().filter(x => !x.failed && (!x.final || !summarized.has(`${x.id}:0`)));
  $('live-text').textContent = pending.map(x => x.text).join(' ') || (transcript.items.size ? '다음 말씀을 기다리고 있습니다.' : '강의를 시작하면 여기에 말씀이 나타납니다.');
}
for (const name of ['notes', 'script']) $(name + '-tab').onclick = () => {
  for (const tab of ['notes', 'script']) { $(tab + '-view').hidden = tab !== name; $(tab + '-tab').setAttribute('aria-pressed', String(tab === name)); }
};
function controls() {
  renderUsage();
  updateLive();
  const active = mode !== 'idle' || uiBusy;
  $('start').disabled = active || !configured || summarizing || recordLoaded || transcript.items.size > 0;
  $('new-lecture').disabled = active || summarizing; $('home').disabled = active || summarizing; $('new-from-note').disabled = active || summarizing;
  $('demo').disabled = active || summarizing;
  $('stop').disabled = !['live', 'demo'].includes(mode);
  $('retry').hidden = !queue?.failed || mode !== 'idle';
  $('title').disabled = active; $('glossary').disabled = active || streaming;
  $('export').disabled = !transcript.items.size;
  $('summarize').disabled = !summaryReady() || summarizing || mode === 'demo' || !pendingSummary().length;
  $('indicator').className = ['live', 'demo'].includes(mode) ? 'live' : '';
}
async function api(path, data) {
  const response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data), signal: AbortSignal.timeout(130000) });
  const result = await response.json(); if (!response.ok) throw Object.assign(new Error(result.error || '요청 실패'), { usage: reportedUsage(result.usage) }); return result;
}
function render() {
  changed();
  const target = $('transcript'), atBottom = target.scrollHeight - target.scrollTop - target.clientHeight < 100;
  const items = transcript.ordered().filter(x => x.text);
  const existing = new Map([...target.querySelectorAll('.segment')].map(row => [row.dataset.id, row]));
  const rows = new Map(existing);
  if (!existing.size && items.length) target.replaceChildren();
  for (const item of items) {
    let row = existing.get(item.id);
    if (!row) {
      row = document.createElement('div'); row.dataset.id = item.id;
      row.append(document.createElement('time'), document.createElement('p')); target.append(row); rows.set(item.id, row);
    }
    row.className = `segment ${item.final ? '' : 'pending'}`;
    row.firstChild.textContent = timestamp(item.seconds);
    if (row.lastChild.textContent !== item.text) row.lastChild.textContent = item.text;
    existing.delete(item.id);
  }
  for (const row of existing.values()) row.remove();
  // Usually append-only; keep rare failure/retry insertions chronological too.
  items.forEach((item, index) => {
    const row = rows.get(item.id);
    if (row && target.children[index] !== row) target.insertBefore(row, target.children[index] || null);
  });
  if (atBottom) target.scrollTop = target.scrollHeight;
  $('count').textContent = `${items.filter(x => x.final).length}개 구간${items.some(x => !x.final) ? ' · 실시간 초안' : ''}`; controls();
}

function renderStreaming(events) {
  if (events.length) changed();
  const target = $('transcript'), atBottom = target.scrollHeight - target.scrollTop - target.clientHeight < 100;
  for (const event of events) {
    if (!event.text) continue;
    const previous = transcript.items.get(event.id);
    transcript.set(event.id, event);
    if (event.final && !previous?.final) finalRows++;
    let row = streamingRows.get(event.id);
    if (!row) {
      if (!streamingRows.size) target.replaceChildren();
      row = document.createElement('div'); row.append(document.createElement('time'), document.createElement('p'));
      row.firstChild.textContent = timestamp(event.seconds); row.dataset.id = event.id;
      target.append(row); streamingRows.set(event.id, row);
    }
    row.className = `segment ${event.final ? '' : 'pending'}`;
    row.lastChild.textContent = event.text;
  }
  if (atBottom && events.length) target.scrollTop = target.scrollHeight;
  $('count').textContent = `${finalRows}개 구간 · 실시간 스트리밍`;
  if (events.length) controls();
}
const summaryLabels = { chatgpt: 'ChatGPT · 문단별 요점과 바꿔쓰기', ollama: '문단 요약 · AI가 정리한 내용', extractive: '핵심 문장 추출 · 바꿔쓰기 아님', demo: '쉽게 풀어쓴 내용 · 샘플' };
function renderUsage() {
  const u = usageTotals(summaryBlocks), format = n => n.toLocaleString('ko-KR');
  $('token-total').textContent = `${u.unknown ? '확인된 ' : ''}${format(u.total_tokens)} 토큰`;
  $('token-detail').textContent = `이 강의 · 입력 ${format(u.input_tokens)} / 출력 ${format(u.output_tokens)} · ${u.requests}회 요청${u.unknown ? ` · ${u.unknown}회 사용량 미확인` : ''}`;
}
function renderSummary(block) {
  changed(); renderUsage();
  const target = $('summary');
  let card = target.querySelector(`[data-block="${block.id}"]`);
  if (!card) {
    if (!target.querySelector('.summary-block')) target.replaceChildren();
    card = document.createElement('article'); card.className = 'summary-block'; card.dataset.block = block.id;
    const status = document.createElement('div'); status.className = 'block-status';
    const result = document.createElement('div'); result.className = 'summary-copy';
    const warning = document.createElement('p'); warning.className = 'block-warning';
    const original = document.createElement('details'); const toggle = document.createElement('summary');
    toggle.textContent = `원문 보기 · ${timestamp(block.items[0].seconds)}`;
    const source = document.createElement('p'); source.className = 'block-source'; source.textContent = block.source;
    const cleaned = document.createElement('p'); cleaned.className = 'cleaned-source';
    original.append(toggle, source, cleaned); card.append(status, result, warning, original); target.append(card);
  }
  const result = card.querySelector('.summary-copy'); result.replaceChildren();
  const sections = block.sections || (block.text ? [{ title: block.title || '강의 요점', bullets: block.text.split('\n').map(x => x.replace(/^[•-]\s*/, '').trim()).filter(Boolean) }] : []);
  for (const section of sections) {
    const heading = document.createElement('h3'); heading.textContent = section.title;
    const list = document.createElement('ul');
    for (const bullet of section.bullets) { const li = document.createElement('li'); li.textContent = bullet; list.append(li); }
    result.append(heading, list);
  }
  card.querySelector('.cleaned-source').textContent = block.cleaned ? `다듬은 문장 · AI 편집\n${block.cleaned}` : '';
  card.querySelector('.cleaned-source').hidden = !block.cleaned;
  card.setAttribute('aria-busy', String(block.state === 'pending'));
  card.querySelector('.block-status').textContent = block.state === 'pending' ? '주제별 요점을 정리하는 중…' : block.state === 'failed' ? '처리 실패 · 지금 정리로 재시도' : `${timestamp(block.items[0].seconds)} · ${summaryLabels[block.method] || '저장된 노트'}`;
  card.querySelector('.block-warning').textContent = block.warning || '';
  card.querySelector('.block-warning').hidden = !block.warning;
}
function reset() {
  transcript = new Transcript(); streamingRows = new Map(); finalRows = 0; streamSession = null; summarized.clear(); summaryBlocks = []; elapsed = 0; queue = null; $('latency').textContent = ''; error('');
  $('summary').textContent = '강의 내용을 기다리고 있습니다.'; $('transcript').textContent = '말씀하시면 여기에 실시간 초안이 나타납니다. 조용할 때는 기다립니다.'; $('timer').textContent = '00:00'; $('count').textContent = '0개 구간'; $('summary-status').textContent = '1분 15초마다 주제별로 정리합니다';
}
function canReset() { return !queue?.items.length || window.confirm('아직 처리하지 못한 음성은 이 탭에만 있습니다. 이동하면 재시도할 수 없습니다. 계속할까요?'); }
function clock() { started = Date.now(); lastSummaryAt = started; timer = setInterval(() => { elapsed = (Date.now() - started) / 1000; $('timer').textContent = timestamp(elapsed); }, 500); }
function pendingSummary() {
  return transcript.ordered().filter(x => x.final && x.text && !x.failed).flatMap(item => {
    const parts = [];
    for (let offset = 0; offset < item.text.length; offset += 3000) parts.push({ ...item, id: `${item.id}:${offset}`, text: item.text.slice(offset, offset + 3000) });
    return parts;
  }).filter(item => !summarized.has(item.id));
}
async function summarize() {
  if (summarizing || mode === 'demo' || !summaryReady()) return false;
  let block = summaryBlocks.find(item => item.state === 'failed');
  if (!block) {
    const batch = sourceBatch(pendingSummary());
    if (!batch.length) return false;
    block = { id: summaryBlocks.length + 1, items: batch, source: batch.map(x => `[${timestamp(x.seconds)}] ${x.text}`).join('\n'), state: 'pending' };
    summaryBlocks.push(block);
  }
  block.attempts ??= block.method === 'chatgpt' ? [null] : [];
  const attempt = block.attempts.push(null) - 1;
  lastSummaryAt = Date.now();
  block.state = 'pending'; block.warning = ''; renderSummary(block);
  summarizing = true; controls(); $('summary-status').textContent = `블록 ${block.id} · 원문을 다듬고 요점을 정리하는 중…`;
  try {
    const result = await api('/api/summary', { previous: '', block: true, paragraph: true, live: mode === 'live' || mode === 'stopping', transcript: block.source, ...summaryOptions() });
    block.attempts[attempt] = reportedUsage(result.usage); block.sections = result.sections;
    block.title = result.title; block.cleaned = result.cleaned; block.text = result.summary; block.method = result.method; block.warning = result.warning || ''; block.state = 'done';
    block.items.forEach(x => summarized.add(x.id)); renderSummary(block);
    $('summary-status').textContent = `블록 ${block.id} 완료 · ${summaryLabels[result.method]}`;
    return true;
  } catch (e) { block.attempts[attempt] = reportedUsage(e.usage); pauseChatGPT(e.message); block.state = 'failed'; block.warning = e.message; renderSummary(block); $('summary-status').textContent = '블록 처리 실패 · 지금 요약으로 다시 시도'; return false; }
  finally { summarizing = false; controls(); }
}

async function finalSummary() {
  while (summarizing) await new Promise(resolve => setTimeout(resolve, 100));
  while (pendingSummary().length) { if (!await summarize()) break; }
}
function queueChanged() {
  if (mode === 'live') $('status').textContent = queue.items.length > 2 ? `듣는 중 · ${queue.items.length}개 구간 처리 대기` : '강의를 듣고 있어요';
  if (qwen && mode === 'live' && queue.items.filter(item => item.final !== false).length >= 2) {
    error('인식이 강의 속도를 따라가지 못해 녹음을 멈춥니다. 받은 음성은 처리합니다. 0.6B 모델을 사용하거나 다른 무거운 앱을 종료하세요.');
    void stopLecture();
  }
  if (streaming && mode === 'live' && queue.items.length >= 10) {
    error('인식 지연이 2초 이상 쌓여 녹음을 멈춥니다. 다른 무거운 앱을 종료하세요. 받은 음성은 모두 처리합니다.');
    void stopLecture();
  }
  if (queue.failed) {
    error(`${queue.failed.message} 음성 구간은 이 탭에 보관됩니다. 처리 재시도를 눌러 주세요.`);
    if (mode === 'live') void stopLecture();
  }
  controls();
}
function enqueue(chunk) {
  const id = chunk.id || `packet-${chunk.sequence}`;
  if (!queue.enqueue({ ...chunk, id })) {
    transcript.set(id, { seconds: chunk.seconds, text: '[처리 지연으로 이 구간을 저장하지 못했습니다]', final: true, failed: true });
    error('컴퓨터의 처리 속도가 강의를 따라가지 못해 녹음을 멈춥니다. 누락 구간이 표시됩니다. 더 작은 모델을 선택하세요.');
    render();
    if (mode === 'live') void stopLecture();
  }
}
$('start').onclick = async () => {
  if (!canReset()) return;
  if (!await createRecord()) return;
  reset(); mode = 'connecting'; changed(); controls(); $('status').textContent = '로컬 음성 모델 준비 중';
  try {
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) throw new Error('Chrome 또는 Edge에서 localhost로 접속해 주세요.');
    // Load the model before recording so initial model load cannot lose lecture audio.
    const session = await api('/api/session', { glossary: $('glossary').value });
    streamSession = session.session;
    const glossary = $('glossary').value;
    if (streaming) queue = new StreamingQueue(async chunk => {
      const result = await api('/api/transcribe', { audio: chunk.audio, glossary, final: chunk.final, session: streamSession, sequence: chunk.sequence });
      renderStreaming(result.events || []);
      const packetDelay = (performance.now() - chunk.capturedAt) / 1000;
      $('latency').textContent = `처리 ${Math.round(result.inferenceMs || 0)} ms · 전송/대기 ${packetDelay.toFixed(1)}초`;
      $('latency').title = '최근 200ms 음성 패킷의 처리/대기 시간입니다. 단어 인식 지연에는 모델의 문맥 대기 시간도 포함됩니다.';
    }, queueChanged);
    else queue = new AudioQueue(async chunk => {
      let result; const requestStarted = performance.now();
      try { result = await api('/api/transcribe', { audio: chunk.audio, glossary, final: chunk.final }); }
      catch (error) {
        if (chunk.final) {
          transcript.set(chunk.id, { seconds: chunk.seconds, text: transcript.items.get(chunk.id)?.text || '[이 구간은 인식 재시도가 필요합니다]', final: false }); render();
        }
        throw error;
      }
      if (qwen) $('latency').textContent = `인식 요청 ${((performance.now() - requestStarted) / 1000).toFixed(1)}초 · 대기 ${Math.max(0, queue.items.length - 1)}개`;
      if (transcript.items.get(chunk.id)?.failed) return;
      const previous = transcript.ordered().filter(x => x.final && !x.failed && x.seconds < chunk.seconds).at(-1)?.text || '';
      const text = chunk.overlap ? removeOverlap(previous, result.text) : result.text;
      if (text.trim()) transcript.set(chunk.id, { seconds: chunk.seconds, text, final: chunk.final });
      else if (chunk.final) transcript.items.delete(chunk.id);
      render();
    }, queueChanged);
    microphone = new Microphone();
    await microphone.start(enqueue, message => { error(message); if (mode === 'live') void stopLecture(); }, streaming ? (rate, callback) => new StreamingFrames(rate, callback) : qwen ? (rate, callback) => new Segmenter(rate, callback, { previewSeconds: 0.8, pauseSeconds: 0.5, maxSeconds: 6 }) : undefined);
    mode = 'live'; clock(); $('status').textContent = '강의를 듣고 있어요'; $('notice').textContent = '로컬 처리 중 · 음성은 이 컴퓨터에서만 처리됩니다. 말하는 동안 초안이 갱신되고, 잠시 멈추면 확정됩니다. 회색 글씨는 수정될 수 있는 초안입니다.';
  } catch (e) { microphone?.dispose(); mode = 'idle'; $('status').textContent = '시작 실패'; error(e.name === 'NotAllowedError' ? '마이크 권한을 허용한 후 다시 시도해 주세요.' : e.message); }
  changed(); controls(); await saveRecord();
};
async function stopLecture() {
  if (!['live', 'demo'].includes(mode)) return;
  const wasDemo = mode === 'demo'; mode = 'stopping'; controls(); $('status').textContent = '남은 구간 처리 중';
  clearInterval(timer); clearInterval(demoTimer);
  if (!wasDemo) { await microphone?.stop(); await queue?.settle(); await finalSummary(); }
  mode = 'idle'; $('status').textContent = queue?.failed ? '처리 중단 · 재시도 가능' : '강의 종료'; $('notice').textContent = '전사와 문단 노트가 이 컴퓨터에 저장됩니다. 처리하지 못한 음성은 탭을 닫으면 사라집니다.'; changed(); controls(); await flushRecord();
}
$('stop').onclick = () => stopLecture();
$('retry').onclick = async () => {
  error(''); mode = 'stopping'; controls(); $('status').textContent = '남은 구간 다시 처리 중';
  await queue.retry(); await finalSummary(); mode = 'idle'; $('status').textContent = queue.failed ? '처리 실패 · 재시도 가능' : '처리 완료'; changed(); controls(); await flushRecord();
};
$('demo').onclick = async () => {
  if (!await newLecture()) return;
  if (!await createRecord()) return;
  mode = 'demo'; $('title').value = '경제학개론 · 기회비용'; $('status').textContent = '샘플 강의 재생'; $('notice').textContent = '데모 · 미리 작성된 전사와 요약입니다. 마이크나 AI 모델을 사용하지 않습니다.'; clock(); controls(); let i = 0;
  const next = () => {
    transcript.set(`demo-${i}`, { text: demoLines[i], seconds: elapsed, final: true }); summarized.add(`demo-${i}:0`); render();
    const block = { id: i + 1, items: [{ seconds: elapsed }], source: `[${timestamp(elapsed)}] ${demoLines[i]}`, title: ['기회비용의 의미', '공부와 아르바이트 사이의 선택', '최선의 대안 하나를 기준으로', '매몰비용과 합리적인 선택'][i], cleaned: demoLines[i], text: '• ' + demoParaphrases[i], method: 'demo', state: 'done' };
    summaryBlocks.push(block); renderSummary(block); i++;
    if (i === demoLines.length) {  $('summary-status').textContent = '샘플 요약 · 미리 작성된 예시'; clearInterval(timer); clearInterval(demoTimer); mode = 'idle'; $('status').textContent = '샘플 강의 종료'; changed(); controls(); void flushRecord(); }
  }; demoTimer = setInterval(next, 2200); next();
};
$('summarize').onclick = () => void summarize();
$('export').onclick = () => {
  const blocks = summaryBlocks.map(block => `### 블록 ${block.id} · ${timestamp(block.items[0].seconds)}\n\n원문\n${block.source}\n\n${block.cleaned ? '다듬은 문장 (AI 편집)\n' + block.cleaned + '\n\n' : ''}${summaryLabels[block.method] || '처리 중 / 재시도 필요'}\n${blockMarkdown(block)}${block.warning ? '\n' + block.warning : ''}`).join('\n\n');
  const content = `# ${$('title').value || '강의 노트'}\n\n## 토큰 사용량\n${$('token-total').textContent} · ${$('token-detail').textContent}\n\n## 블록별 노트\n${blocks || '(요약 없음)'}\n\n## 전사\n${transcript.ordered().map(x => `[${timestamp(x.seconds)}] ${x.text || '(음성 없음)'}${x.final ? '' : ' [미확정 초안 / 재처리 필요]'}`).join('\n\n')}\n`;
  const url = URL.createObjectURL(new Blob([content], { type: 'text/markdown;charset=utf-8' })); const a = document.createElement('a'); a.href = url; a.download = 'lecture-notes.md'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
setInterval(() => { if (mode === 'live' && summaryDue(Date.now(), lastSummaryAt)) void summarize(); }, 1000);
window.addEventListener('beforeunload', event => { if (changeVersion !== savedVersion || mode !== 'idle' || queue?.items.length) { event.preventDefault(); event.returnValue = ''; } });
window.addEventListener('pagehide', () => { microphone?.dispose(); clearInterval(timer); clearInterval(demoTimer); });
async function runAction(action) {
  if (uiBusy) return;
  uiBusy = true; controls();
  try { return await action(); } finally { uiBusy = false; controls(); }
}
for (const id of ['start', 'stop', 'new-lecture', 'new-from-note', 'home', 'demo', 'retry']) {
  const action = $(id).onclick; $(id).onclick = () => void runAction(action);
}
$('settings-toggle').onclick = () => { const panel = $('settings-panel'); panel.hidden = !panel.hidden; $('settings-toggle').setAttribute('aria-expanded', String(!panel.hidden)); };
initChatGPT(controls);
try { const response = await fetch('/api/config'); const config = await response.json(); configured = config.configured; streaming = Boolean(config.streaming); qwen = config.backend === 'qwen-mlx'; $('glossary').placeholder = streaming ? '스트리밍 모드에서는 전공 용어 힌트를 지원하지 않습니다' : $('glossary').placeholder; $('model').textContent = config.model; $('status').textContent = configured ? '시작할 준비가 됐어요' : '로컬 음성 모델 설치 필요'; $('notice').textContent = configured ? 'API 키 없이 이 컴퓨터에서 강의를 받아씁니다. 녹음 전 강의 정책과 동의를 확인하세요.' : qwen ? 'Qwen3-ASR 설치가 필요합니다. Apple Silicon Mac에서 npm run setup:qwen을 실행하고 서버를 다시 시작하세요.' : config.reason === 'platform' ? 'MLX는 Apple Silicon Mac과 ARM Python이 필요합니다. 다른 컴퓨터에서는 WHISPER_BACKEND=faster-whisper를 사용하세요.' : config.reason === 'model' ? streaming ? '스트리밍 모델 설치가 필요합니다: npm run setup:realtime. 설치 후 서버를 재시작하세요.' : '음성 모델 파일이 없습니다. README의 모델 다운로드 단계를 완료한 뒤 새로고침하세요. 샘플 강의는 바로 체험할 수 있습니다.' : '음성 엔진을 설치하세요. 실시간 모드: npm run setup:realtime. 샘플 강의는 바로 체험할 수 있습니다.'; }
catch { error('서버에 연결할 수 없습니다. 새로고침해 주세요.'); $('status').textContent = '서버 연결 실패'; }
controls();

await refreshLibrary();
