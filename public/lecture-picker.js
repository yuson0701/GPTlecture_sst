import { suggestVocabulary } from './lecture-vocabulary.js';
export function initLecturePicker({ onApply, canEdit }) {
  const $ = id => document.getElementById(id);
  let data = { lectures: [] }, active = null, busy = false;
  const fields = ['schedule-course', 'schedule-date', 'schedule-search', 'schedule-lecture', 'schedule-file', 'schedule-apply'];
  const label = row => `${row.date} · ${row.period}교시 · ${row.course} · ${row.title}${row.professor ? ' · ' + row.professor : ''}`;
  const option = (text, value) => { const item = document.createElement('option'); item.textContent = text; item.value = value; return item; };
  function locked(value) {
    for (const id of fields) $(id).disabled = value || busy;
    if (!value) preview();
  }
  function preview() {
    const row = data.lectures.find(row => row.id === $('schedule-lecture').value);
    $('schedule-preview').textContent = row ? `${label(row)}\n추천 용어: ${suggestVocabulary(row).join(', ')}` : '강의를 고르면 추천 용어를 미리 볼 수 있습니다.';
    $('schedule-apply').disabled = busy || !canEdit() || !row;
  }
  function filter() {
    const selected = $('schedule-lecture').value;
    const query = $('schedule-search').value.normalize('NFC').trim().toLowerCase();
    const rows = data.lectures.filter(row => (!$('schedule-course').value || row.course === $('schedule-course').value) &&
      (!$('schedule-date').value || row.date === $('schedule-date').value) && (!query || label(row).toLowerCase().includes(query)));
    $('schedule-lecture').replaceChildren(option(rows.length ? `강의 선택 (${rows.length}개)` : '조건에 맞는 강의가 없습니다', ''), ...rows.map(row => option(label(row), row.id)));
    if (rows.some(row => row.id === selected)) $('schedule-lecture').value = selected;
    preview();
  }
  function restore(value) {
    active = value || null;
    $('schedule-active').textContent = active ? `선택한 강의: ${label(active)}` : '시간표 선택 없이 직접 제목과 용어를 입력해도 됩니다.';
    if (active) {
      $('schedule-course').value = data.lectures.some(row => row.course === active.course) ? active.course : '';
      $('schedule-date').value = active.date; $('schedule-search').value = ''; filter();
      $('schedule-lecture').value = active.id; preview();
    } else { $('schedule-lecture').value = ''; preview(); }
  }
  function render() {
    const course = $('schedule-course').value;
    $('schedule-course').replaceChildren(option('모든 과목', ''), ...[...new Set(data.lectures.map(row => row.course))].sort().map(c => option(c, c)));
    $('schedule-course').value = course;
    $('schedule-source').textContent = data.lectures.length ? `${data.sourceName || '시간표'} · ${data.lectures.length}개 일정${data.missingCourse ? ` · 과목 미기재 ${data.missingCourse}개` : ''}` : 'Excel 시간표를 불러오세요. 이 컴퓨터에만 저장됩니다.';
    filter(); if (active) restore(active);
  }
  async function load() {
    try { const response = await fetch('/api/schedule'); if (!response.ok) throw Error('시간표를 불러오지 못했습니다.'); data = await response.json(); render(); }
    catch (error) { $('schedule-source').textContent = error.message; }
  }
  for (const id of ['schedule-course', 'schedule-date', 'schedule-search']) $(id).addEventListener('input', filter);
  $('schedule-lecture').addEventListener('change', preview);
  $('schedule-apply').onclick = () => {
    if (!canEdit()) return;
    const row = data.lectures.find(row => row.id === $('schedule-lecture').value); if (!row) return;
    const { id, date, period, course, title, professor } = row;
    const chosen = { id, date, period, course, title, professor };
    restore(chosen); onApply(chosen, suggestVocabulary(row).join(', '));
  };
  $('schedule-file').onchange = async () => {
    const file = $('schedule-file').files[0]; if (!file || !canEdit()) return;
    busy = true; locked(true); $('schedule-source').textContent = '시간표를 읽는 중…';
    try {
      if (file.size > 2 * 1024 * 1024 || !file.name.toLowerCase().endsWith('.xlsx')) throw Error('2 MB 이하의 .xlsx 파일을 선택하세요.');
      const xlsx = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result).split(',')[1]); reader.onerror = () => reject(Error('파일을 읽지 못했습니다.')); reader.readAsDataURL(file); });
      const response = await fetch('/api/schedule/import', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ filename: file.name, xlsx }), signal: AbortSignal.timeout(25000) });
      const result = await response.json(); if (!response.ok) throw Error(result.error || '시간표를 읽지 못했습니다.');
      data = result; render();
    } catch (error) { $('schedule-source').textContent = `${error.message} 기존 시간표는 유지됩니다.`; }
    finally { busy = false; $('schedule-file').value = ''; locked(!canEdit()); preview(); }
  };
  function wordCount() { const count = $('glossary').value.split(/[,\n]/).filter(word => word.trim()).length; $('glossary-count').textContent = `${count}개 용어 · 모델은 앞에서 최대 50개 사용`; }
  $('glossary').addEventListener('input', wordCount);
  void load(); wordCount();
  return { restore, snapshot: () => active, locked, wordCount };
}
