const $ = id => document.getElementById(id);
let session = { status: 'disconnected', sharing: false }, polling = false, busy = false, notify = () => {};

async function request(path, body) {
  const response = await fetch('/api/chatgpt/' + path, { method: body === undefined ? 'GET' : 'POST',
    headers: { 'X-Lecture-Client': '1', ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }), signal: AbortSignal.timeout(25000) });
  const data = await response.json();
  if (!response.ok) throw Error(data.error || 'ChatGPT 연결 실패');
  return data;
}
export function summaryOptions() {
  return { consent: $('chatgpt-consent').checked, model: $('chatgpt-model').value };
}
export function summaryReady() {
  return session.status === 'connected' && session.sharing && summaryOptions().consent && !!summaryOptions().model;
}
export function pauseChatGPT(message) {
  $('chatgpt-consent').checked = false;
  $('chatgpt-message').textContent = message + ' 원문은 보관됩니다. 연결을 확인한 뒤 자동 정리를 다시 허용해 주세요.';
  notify();
}
function render() {
  const connected = session.status === 'connected', connecting = session.status === 'connecting';
  $('chatgpt-account').textContent = connecting ? '브라우저에서 로그인을 완료해 주세요.' : connected
    ? `${session.identity?.email || session.identity?.name || 'ChatGPT 연결됨'} · ${session.sharing ? '구독 사용 가능' : '구독 사용 권한 필요'}` : 'ChatGPT 연결 안 됨';
  $('chatgpt-login').hidden = connecting || connected;
  $('chatgpt-login').disabled = busy;
  $('chatgpt-reconnect').hidden = !connected;
  $('chatgpt-reconnect').disabled = busy;
  $('chatgpt-cancel').hidden = !connecting;
  $('chatgpt-disconnect').hidden = !connected;
  $('chatgpt-disconnect').disabled = busy;
  $('chatgpt-refresh').disabled = busy || !connected || !session.sharing;
  $('chatgpt-model').disabled = busy || !connected || !session.sharing;
  $('chatgpt-consent').disabled = !connected || !session.sharing || !$('chatgpt-model').value;
  if (!connected || !session.sharing) $('chatgpt-consent').checked = false;
  notify();
}
async function models() {
  const selected = $('chatgpt-model').value;
  const data = await request('models');
  const placeholder = document.createElement('option'); placeholder.value = ''; placeholder.textContent = data.models.length ? '요약 모델을 선택하세요' : '사용 가능한 모델이 없습니다';
  $('chatgpt-model').replaceChildren(placeholder);
  for (const model of data.models) {
    const option = document.createElement('option'); option.value = model.slug; option.textContent = model.displayName;
    $('chatgpt-model').append(option);
  }
  if (data.models.some(x => x.slug === selected)) $('chatgpt-model').value = selected;
  if (!$('chatgpt-model').value) $('chatgpt-consent').checked = false;
}
async function refresh() {
  if (polling || busy) return;
  polling = true;
  try {
    const previous = session;
    session = await request('status');
    if (session.error) $('chatgpt-message').textContent = session.error;
    if (session.status === 'connected' && session.sharing && (previous.status !== 'connected' || !previous.sharing)) await models();
  } catch (e) { session = { status: 'disconnected', sharing: false }; $('chatgpt-message').textContent = e.message; }
  finally { polling = false; render(); }
}
export function initChatGPT(onChange) {
  notify = onChange;
  for (const [id, path, body] of [['login', 'sign-in', {}], ['reconnect', 'sign-in', { reconsent: true }], ['cancel', 'cancel', {}], ['disconnect', 'disconnect', {}]]) {
    $('chatgpt-' + id).onclick = async () => {
      busy = true; $('chatgpt-consent').checked = false; $('chatgpt-message').textContent = ''; render();
      try { session = await request(path, body); }
      catch (e) { $('chatgpt-message').textContent = e.message; }
      finally { busy = false; render(); }
      await refresh();
    };
  }
  $('chatgpt-refresh').onclick = async () => {
    busy = true; render();
    try { await models(); $('chatgpt-message').textContent = ''; }
    catch (e) { pauseChatGPT(e.message); }
    finally { busy = false; render(); }
  };
  $('chatgpt-model').onchange = () => { $('chatgpt-consent').checked = false; render(); };
  $('chatgpt-consent').onchange = () => { $('chatgpt-message').textContent = $('chatgpt-consent').checked ? '새 요약 요청에 강의 텍스트를 전송합니다. ChatGPT 구독 사용량에 반영됩니다.' : '자동 정리를 멈췄습니다. 이미 전송한 요청은 완료될 수 있습니다.'; render(); };
  void refresh();
  setInterval(() => { void refresh(); }, 3000);
}
