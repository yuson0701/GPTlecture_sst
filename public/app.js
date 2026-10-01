import { Transcript, timestamp } from './transcript.js';
import { Microphone, AudioQueue, removeOverlap } from './audio.js';
import { StreamingFrames, StreamingQueue } from './streaming.js';
const $ = id => document.getElementById(id);
let transcript = new Transcript(), microphone, queue, timer, demoTimer, mode = 'idle', started = 0, elapsed = 0, summary = '', summaryMethod = '', configured = false, summarizing = false;
const summarized = new Set();
let streaming = false, streamingRows = new Map(), finalRows = 0, streamSession = null;
const demoLines = ['오늘은 경제학의 기본 개념인 기회비용에 대해 알아보겠습니다. 기회비용은 어떤 선택을 했을 때 포기한 대안 중 가장 가치 있는 것의 가치입니다.', '예를 들어 두 시간 동안 아르바이트를 하면 2만 원을 벌 수 있지만, 그 시간에 시험 공부를 선택했다면 포기한 2만 원이 기회비용에 포함됩니다.', '여기서 중요한 것은 모든 대안의 가치를 더하는 것이 아니라, 포기한 대안 중 가장 좋은 하나만 고려한다는 점입니다.', '이미 지출해서 회수할 수 없는 비용은 매몰비용이라고 합니다. 합리적인 의사결정에서는 매몰비용보다 앞으로 발생할 비용과 편익을 비교해야 합니다.'];
const demoSummary = '핵심 요약\n선택에는 기회비용이 따릅니다. 포기한 대안 중 가장 가치 있는 하나를 기준으로 생각합니다.\n\n주요 개념\n• 기회비용: 선택으로 포기한 최선의 대안의 가치\n• 매몰비용: 이미 지출해서 회수할 수 없는 비용\n\n쉽게 이해하기\n공부를 선택해 아르바이트를 못 했다면, 포기한 임금이 기회비용에 포함됩니다.\n\n확인할 점\n의사결정에서는 앞으로의 비용과 편익을 비교합니다.';
function error(message) { $('error').textContent = message; $('error').hidden = !message; }
function controls() {
  const active = mode !== 'idle';
  $('start').disabled = active || !configured || summarizing;
  $('demo').disabled = active || summarizing;
  $('stop').disabled = !['live', 'demo'].includes(mode);
  $('retry').hidden = !queue?.failed || mode !== 'idle';
  $('title').disabled = active; $('glossary').disabled = active || streaming;
  $('export').disabled = !transcript.items.size;
  $('summarize').disabled = summarizing || mode === 'demo' || !pendingSummary().length;
  $('indicator').className = ['live', 'demo'].includes(mode) ? 'live' : '';
}
async function api(path, data) {
  const response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data), signal: AbortSignal.timeout(130000) });
  const result = await response.json(); if (!response.ok) throw new Error(result.error || '요청 실패'); return result;
}
function render() {
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
function renderSummary() { const p = document.createElement('p'); p.className = 'summary-copy'; p.textContent = summary; $('summary').replaceChildren(p); }
function reset() {
  transcript = new Transcript(); streamingRows = new Map(); finalRows = 0; streamSession = null; summarized.clear(); summary = ''; summaryMethod = ''; elapsed = 0; queue = null; $('latency').textContent = ''; error('');
  $('summary').textContent = '강의 내용을 기다리고 있습니다.'; $('transcript').textContent = '말씀하시면 여기에 실시간 초안이 나타납니다. 조용할 때는 기다립니다.'; $('timer').textContent = '00:00'; $('count').textContent = '0개 구간'; $('summary-status').textContent = streaming ? '실시간 핵심 문장 추출 · 20초마다 갱신' : '새 내용이 쌓이면 20초마다 갱신';
}
function canReset() { return !transcript.items.size || window.confirm('현재 노트를 내보내셨나요? 새 강의를 시작하면 기존 노트와 미처리 음성이 지워집니다.'); }
function clock() { started = Date.now(); timer = setInterval(() => { elapsed = (Date.now() - started) / 1000; $('timer').textContent = timestamp(elapsed); }, 500); }
function pendingSummary() { return transcript.ordered().filter(x => x.final && x.text && !x.failed && !summarized.has(x.id)); }
async function summarize() {
  if (summarizing || mode === 'demo') return false;
  const batch = []; let length = 0;
  for (const item of pendingSummary()) { if (length + item.text.length > 6400) break; batch.push(item); length += item.text.length + 20; }
  if (!batch.length) return false;
  summarizing = true; controls(); $('summary-status').textContent = '이 컴퓨터에서 정리하고 있어요…';
  try {
    const result = await api('/api/summary', { previous: summary, live: mode === 'live' || mode === 'stopping', transcript: batch.map(x => `[${timestamp(x.seconds)}] ${x.text}`).join('\n') });
    summary = result.summary; summaryMethod = result.method; batch.forEach(x => summarized.add(x.id)); renderSummary();
    $('summary-status').textContent = result.method === 'ollama' ? '로컬 AI 요약 · Qwen' : '핵심 문장 추출 · AI 설명 아님';
    if (result.warning) $('notice').textContent = result.warning;
    return true;
  } catch (e) { error(e.message); $('summary-status').textContent = '요약 실패 · 지금 요약으로 다시 시도'; return false; }
  finally { summarizing = false; controls(); }
}
async function finalSummary() {
  while (summarizing) await new Promise(resolve => setTimeout(resolve, 100));
  while (pendingSummary().length) { if (!await summarize()) break; }
}
function queueChanged() {
  if (mode === 'live') $('status').textContent = queue.items.length > 2 ? `듣는 중 · ${queue.items.length}개 구간 처리 대기` : '강의를 듣고 있어요';
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
  reset(); mode = 'connecting'; controls(); $('status').textContent = '로컬 음성 모델 준비 중';
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
      let result;
      try { result = await api('/api/transcribe', { audio: chunk.audio, glossary, final: chunk.final }); }
      catch (error) {
        if (chunk.final) {
          transcript.set(chunk.id, { seconds: chunk.seconds, text: transcript.items.get(chunk.id)?.text || '[이 구간은 인식 재시도가 필요합니다]', final: false }); render();
        }
        throw error;
      }
      if (transcript.items.get(chunk.id)?.failed) return;
      const previous = transcript.ordered().filter(x => x.final && !x.failed && x.seconds < chunk.seconds).at(-1)?.text || '';
      const text = chunk.overlap ? removeOverlap(previous, result.text) : result.text;
      if (text.trim()) transcript.set(chunk.id, { seconds: chunk.seconds, text, final: chunk.final });
      else if (chunk.final) transcript.items.delete(chunk.id);
      render();
    }, queueChanged);
    microphone = new Microphone();
    await microphone.start(enqueue, message => { error(message); if (mode === 'live') void stopLecture(); }, streaming ? (rate, callback) => new StreamingFrames(rate, callback) : undefined);
    mode = 'live'; clock(); $('status').textContent = '강의를 듣고 있어요'; $('notice').textContent = '로컬 처리 중 · 음성은 이 컴퓨터에서만 처리됩니다. 말하는 동안 초안이 갱신되고, 잠시 멈추면 확정됩니다. 회색 글씨는 수정될 수 있는 초안입니다.';
  } catch (e) { microphone?.dispose(); mode = 'idle'; $('status').textContent = '시작 실패'; error(e.name === 'NotAllowedError' ? '마이크 권한을 허용한 후 다시 시도해 주세요.' : e.message); }
  controls();
};
async function stopLecture() {
  if (!['live', 'demo'].includes(mode)) return;
  const wasDemo = mode === 'demo'; mode = 'stopping'; controls(); $('status').textContent = '남은 구간 처리 중';
  clearInterval(timer); clearInterval(demoTimer);
  if (!wasDemo) { await microphone?.stop(); await queue?.settle(); await finalSummary(); }
  mode = 'idle'; $('status').textContent = queue?.failed ? '처리 중단 · 재시도 가능' : '강의 종료'; $('notice').textContent = '노트를 내보내 저장하세요. 이 탭을 닫으면 노트와 미처리 음성이 사라집니다.'; controls();
}
$('stop').onclick = () => void stopLecture();
$('retry').onclick = async () => {
  error(''); mode = 'stopping'; controls(); $('status').textContent = '남은 구간 다시 처리 중';
  await queue.retry(); await finalSummary(); mode = 'idle'; $('status').textContent = queue.failed ? '처리 실패 · 재시도 가능' : '처리 완료'; controls();
};
$('demo').onclick = () => {
  if (!canReset()) return; reset(); mode = 'demo'; $('title').value = '경제학개론 · 기회비용'; $('status').textContent = '샘플 강의 재생'; $('notice').textContent = '데모 · 미리 작성된 전사와 요약입니다. 마이크나 AI 모델을 사용하지 않습니다.'; clock(); controls(); let i = 0;
  const next = () => {
    transcript.set(`demo-${i}`, { text: demoLines[i], seconds: elapsed, final: true }); summarized.add(`demo-${i}`); render(); i++;
    if (i === demoLines.length) { summary = demoSummary; summaryMethod = 'demo'; renderSummary(); $('summary-status').textContent = '샘플 요약 · 미리 작성된 예시'; clearInterval(timer); clearInterval(demoTimer); mode = 'idle'; $('status').textContent = '샘플 강의 종료'; controls(); }
  }; demoTimer = setInterval(next, 2200); next();
};
$('summarize').onclick = () => void summarize();
$('export').onclick = () => {
  const labels = { ollama: '로컬 AI 요약', extractive: '핵심 문장 추출 (AI 설명 아님)', demo: '미리 작성된 샘플' };
  const content = `# ${$('title').value || '강의 노트'}\n\n## ${labels[summaryMethod] || '요약'}\n${summary || '(요약 없음)'}\n\n## 전사\n${transcript.ordered().map(x => `[${timestamp(x.seconds)}] ${x.text || '(음성 없음)'}${x.final ? '' : ' [미확정 초안 / 재처리 필요]'}`).join('\n\n')}\n`;
  const url = URL.createObjectURL(new Blob([content], { type: 'text/markdown;charset=utf-8' })); const a = document.createElement('a'); a.href = url; a.download = 'lecture-notes.md'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
setInterval(() => { if (mode === 'live') void summarize(); }, 20000);
window.addEventListener('beforeunload', event => { if (transcript.items.size || mode !== 'idle') { event.preventDefault(); event.returnValue = ''; } });
window.addEventListener('pagehide', () => { microphone?.dispose(); clearInterval(timer); clearInterval(demoTimer); });
try { const response = await fetch('/api/config'); const config = await response.json(); configured = config.configured; streaming = Boolean(config.streaming); $('glossary').placeholder = streaming ? '스트리밍 모드에서는 전공 용어 힌트를 지원하지 않습니다' : $('glossary').placeholder; $('model').textContent = config.model; $('status').textContent = configured ? '시작할 준비가 됐어요' : '로컬 음성 모델 설치 필요'; $('notice').textContent = configured ? 'API 키 없이 이 컴퓨터에서 강의를 받아씁니다. 녹음 전 강의 정책과 동의를 확인하세요.' : config.reason === 'platform' ? 'MLX는 Apple Silicon Mac과 ARM Python이 필요합니다. 다른 컴퓨터에서는 WHISPER_BACKEND=faster-whisper를 사용하세요.' : config.reason === 'model' ? streaming ? '스트리밍 모델 설치가 필요합니다: npm run setup:realtime. 설치 후 서버를 재시작하세요.' : '음성 모델 파일이 없습니다. README의 모델 다운로드 단계를 완료한 뒤 새로고침하세요. 샘플 강의는 바로 체험할 수 있습니다.' : '음성 엔진을 설치하세요. 실시간 모드: npm run setup:realtime. 샘플 강의는 바로 체험할 수 있습니다.'; }
catch { error('서버에 연결할 수 없습니다. 새로고침해 주세요.'); $('status').textContent = '서버 연결 실패'; }
controls();
