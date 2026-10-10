import { suggestVocabulary } from './lecture-vocabulary.js';

const START_MINUTES = [510, 570, 640, 700, 810, 870, 930, 990];
function seoulClock(now) {
  const parts = Object.fromEntries(new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Asia/Seoul', year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23',
  }).formatToParts(now).map(part => [part.type, part.value]));
  return { date: `${parts.year}-${parts.month}-${parts.day}`, minutes: Number(parts.hour) * 60 + Number(parts.minute) + Number(parts.second) / 60 };
}
function periodStarts(value) {
  const starts = [];
  for (const part of String(value).replace(/제|교시|\s/g, '').split(/[,/·]/)) {
    const match = /^([1-8])(?:[-~–]([1-8]))?$/.exec(part);
    if (!match) return [];
    const first = Number(match[1]), last = Number(match[2] || match[1]);
    if (last < first) return [];
    for (let period = first; period <= last; period++) starts.push(START_MINUTES[period - 1]);
  }
  return starts;
}
export function nearestLecture(lectures, now = new Date()) {
  const clock = seoulClock(now);
  // A multi-period lecture can match any of its periods. Equal distances prefer
  // the earlier start; equal starts retain timetable order.
  let chosen = null, distance = Infinity, start = Infinity;
  for (const lecture of lectures) {
    if (lecture.date !== clock.date) continue;
    for (const candidate of periodStarts(lecture.period)) {
      const delta = Math.abs(candidate - clock.minutes);
      if (delta < distance || (delta === distance && candidate < start)) {
        chosen = lecture; distance = delta; start = candidate;
      }
    }
  }
  return chosen;
}

export function initLecturePicker({ onApply, canEdit, canAutoApply = canEdit }) {
  const $ = id => document.getElementById(id);
  let data = { lectures: [] }, active = null, generation = 0, pristine = false;
  const fields = ['schedule-course', 'schedule-date', 'schedule-search', 'schedule-lecture', 'schedule-apply'];
  const label = row => `${row.date} · ${row.period}교시 · ${row.course} · ${row.title}${row.professor ? ' · ' + row.professor : ''}`;
  const option = (text, value) => { const item = document.createElement('option'); item.textContent = text; item.value = value; return item; };
  function locked(value) {
    for (const id of fields) $(id).disabled = value;
    if (!value) preview();
  }
  function preview() {
    const row = data.lectures.find(row => row.id === $('schedule-lecture').value);
    $('schedule-preview').textContent = row ? `${label(row)}\n추천 용어: ${suggestVocabulary(row).join(', ')}` : '';
    $('schedule-apply').disabled = !canEdit() || !row;
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
  function showSelection(value) {
    active = value || null;
    $('schedule-active').textContent = active ? `선택한 강의: ${label(active)}` : '오늘의 등록된 강의가 없습니다. 제목과 용어를 직접 입력하세요.';
    $('schedule-course').value = active && data.lectures.some(row => row.course === active.course) ? active.course : '';
    $('schedule-date').value = active?.date || seoulClock(new Date()).date;
    $('schedule-search').value = ''; filter();
    $('schedule-lecture').value = active?.id || ''; preview();
  }
  function restore(value) { generation++; pristine = false; showSelection(value); }
  function apply(row) {
    const { id, date, period, course, title, professor } = row;
    const chosen = { id, date, period, course, title, professor };
    showSelection(chosen); onApply(chosen, suggestVocabulary(row).join(', '));
  }
  function render() {
    const course = $('schedule-course').value;
    $('schedule-course').replaceChildren(option('모든 과목', ''), ...[...new Set(data.lectures.map(row => row.course))].sort().map(c => option(c, c)));
    $('schedule-course').value = course;
    $('schedule-source').textContent = data.lectures.length ? `${data.lectures.length}개 일정 · 한국 시간 기준 자동 선택` : '저장된 시간표가 없습니다.';
    filter(); if (active) showSelection(active);
  }
  async function load() {
    try { const response = await fetch('/api/schedule', { signal: AbortSignal.timeout(15000) }); if (!response.ok) throw Error('시간표를 불러오지 못했습니다.'); data = await response.json(); render(); }
    catch (error) { $('schedule-source').textContent = error.message; }
  }
  async function autoSelect() {
    if (!pristine || !canAutoApply()) return;
    const version = generation;
    await ready;
    if (version !== generation || !pristine || !canAutoApply()) return;
    const row = nearestLecture(data.lectures);
    if (row && row.id !== active?.id) apply(row);
    else if (!row) { const previous = active; showSelection(null); if (previous) onApply(null, ''); }
  }
  function newLecture() { generation++; pristine = true; showSelection(null); return autoSelect(); }
  // Late timetable responses and the pre-recording time check must never replace
  // deliberate user edits, another lecture, or a recording already in progress.
  for (const id of ['title', 'glossary', 'schedule-course', 'schedule-date', 'schedule-search', 'schedule-lecture']) {
    $(id).addEventListener('input', () => { pristine = false; });
  }
  for (const id of ['schedule-course', 'schedule-date', 'schedule-search']) $(id).addEventListener('input', filter);
  $('schedule-lecture').addEventListener('change', () => { pristine = false; preview(); });
  $('schedule-apply').onclick = () => {
    if (!canEdit()) return;
    const row = data.lectures.find(row => row.id === $('schedule-lecture').value); if (!row) return;
    pristine = false; apply(row);
  };
  function wordCount() { const count = $('glossary').value.split(/[,\n]/).filter(word => word.trim()).length; $('glossary-count').textContent = `${count}개 용어 · 모델은 앞에서 최대 50개 사용`; }
  $('glossary').addEventListener('input', wordCount);
  const ready = load(); wordCount();
  return { restore, newLecture, autoSelect, snapshot: () => active, locked, wordCount };
}
