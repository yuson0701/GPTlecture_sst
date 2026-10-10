export function initSpeechChoice(onChange) {
  const $ = id => document.getElementById(id);
  let models = [], fallback, selected = '', initialized = false;
  function preferences() { try { return JSON.parse(localStorage.getItem('lecture-speech-choice') || '{}'); } catch { return {}; } }
  function publish() {
    const model = models.find(x => x.id === selected);
    $('stt-model').value = selected;
    $('stt-readiness').hidden = !!model?.available;
    $('stt-readiness').textContent = model ? model.available ? '준비 완료 · 이 컴퓨터에서 처리합니다. 녹음 중에는 모델을 바꿀 수 없습니다.' : `설치 또는 설정 확인 필요 · ${model.setup}${model.reason === 'platform' ? ' · Apple Silicon Mac이 필요합니다.' : ''}` : '기존 기록에 모델 정보가 없습니다. 미처리 음성을 재시도하려면 사용할 모델을 선택하세요.';
    if (models.length) onChange({ configured: !!model?.available, backend: model?.backend, streaming: model?.backend === 'sherpa', model: model?.label || '음성 모델 선택 필요' });
  }
  $('stt-model').onchange = () => {
    selected = $('stt-model').value;
    try { localStorage.setItem('lecture-speech-choice', JSON.stringify({ model: selected, language: $('stt-language').value })); } catch {}
    publish();
  };
  $('stt-language').onchange = () => {
    try { localStorage.setItem('lecture-speech-choice', JSON.stringify({ model: selected, language: $('stt-language').value })); } catch {}
    publish();
  };
  const service = {
    load(config) {
      fallback = config.defaultModel; models = config.models || [];
      $('speech-choice').hidden = !models.length;
      $('stt-model').replaceChildren(new Option('음성 모델 선택', ''), ...models.map(model => new Option(`${model.label}${model.available ? '' : ' · 설치 필요'}`, model.id)));
      if (!initialized) { selected = models.some(x => x.id === preferences().model) ? preferences().model : fallback; initialized = true; }
      publish();
    },
    newLecture() {
      const saved = preferences(); selected = models.some(x => x.id === saved.model) ? saved.model : fallback;
      $('stt-language').value = ['auto', 'ko', 'en'].includes(saved.language) ? saved.language : 'auto'; publish();
    },
    restore(record) { if (!models.length) return; selected = record.sttModel || ''; $('stt-language').value = record.recognitionLanguage || 'auto'; publish(); },
    needsSelection: () => models.length > 0 && !selected,
    request: () => models.length ? { sttModel: selected || undefined, language: $('stt-language').value } : {},
    snapshot: () => models.length ? { sttModel: selected || null, recognitionLanguage: $('stt-language').value } : {},
    lock(locked, active = locked) { for (const id of ['stt-model', 'stt-language']) $(id).disabled = locked; $('stt-refresh').disabled = active; },
  };
  $('stt-refresh').onclick = async () => {
    $('stt-refresh').disabled = true;
    try { const response = await fetch('/api/config'); if (!response.ok) throw Error(); service.load(await response.json()); }
    catch { $('stt-readiness').hidden = false; $('stt-readiness').textContent = '모델 상태를 확인하지 못했습니다. 서버 연결을 확인하고 다시 시도하세요.'; }
    finally { $('stt-refresh').disabled = false; }
  };
  return service;
}
