"""Build an offline, hash-bound review page beside a prepared Qwen3-ASR dataset.

The page never uploads audio, changes labels, approves clips automatically, or
starts training. Decisions are downloaded as JSONL for the reviewed exporter.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SAFE_ID = re.compile(r'[A-Za-z0-9_-]+\Z')
HASH = re.compile(r'[0-9a-f]{64}\Z')
SPLITS = ('train', 'validation', 'test')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def build_payload(dataset):
    segments_path = dataset / 'segments.jsonl'
    digest = file_hash(segments_path)
    lectures_path = dataset / 'lectures.jsonl'
    lectures = {}
    for lecture in read_jsonl(lectures_path):
        lid = lecture['lecture_id']
        require(isinstance(lid, str) and SAFE_ID.fullmatch(lid), 'Unsafe lecture ID')
        require(lid not in lectures, f'Duplicate lecture: {lid}')
        require(lecture['split'] in SPLITS, f'Unknown lecture split: {lid}')
        lectures[lid] = lecture
    require(bool(lectures), 'No lectures to review')
    report_path = dataset / 'alignment_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    if report:
        require(report.get('alignment_complete') is True, 'Alignment is not complete')
        require(report.get('segments_sha256') == digest, 'Alignment report is stale: segments changed')
        require(report.get('lectures_sha256') == file_hash(lectures_path), 'Alignment report is stale: lectures changed')
    rows, seen = [], set()
    for segment in read_jsonl(segments_path):
        cid, lid = segment['clip_id'], segment['lecture_id']
        require(isinstance(cid, str) and SAFE_ID.fullmatch(cid), 'Unsafe clip ID')
        require(cid not in seen, f'Duplicate clip: {cid}')
        seen.add(cid)
        require(lid in lectures, f'Unknown lecture: {lid}')
        lecture = lectures[lid]
        require(segment['split'] == lecture['split'], f'Clip split differs from lecture: {cid}')
        issues = segment.get('issues', [])
        require(isinstance(issues, list) and all(isinstance(issue, str) for issue in issues), f'Invalid issues: {cid}')
        coverage_warnings = segment.get('coverage_warnings', [])
        lecture_warnings = lecture.get('issues', [])
        require(all(isinstance(values, list) and all(isinstance(value, str) for value in values)
                    for values in (coverage_warnings, lecture_warnings)), f'Invalid warnings: {cid}')
        automatic_status = segment.get('automatic_status')
        require(automatic_status in ('pass', 'review_required', 'quarantined'), f'Unknown automatic status: {cid}')
        passing = automatic_status == 'pass' and not issues
        require(automatic_status != 'pass' or passing, f'Passing clip has unresolved issues: {cid}')
        boundary_review = (automatic_status == 'review_required'
                           and issues == ['zero_duration_alignment_token']
                           and segment.get('review_kind') == 'sparse_internal_timestamp_anomalies')
        require(automatic_status != 'review_required' or boundary_review,
                f'Invalid boundary-review eligibility: {cid}')
        reviewable = passing or boundary_review
        require(isinstance(segment.get('reference_text', ''), str) and isinstance(segment.get('text', ''), str),
                f'Invalid transcript: {cid}')
        clip_path = dataset / 'clips' / f'{cid}.wav'
        has_clip = segment.get('audio') == str(clip_path) and clip_path.is_file() and not clip_path.is_symlink()
        if reviewable:
            require(has_clip, f'Reviewable clip is missing or moved: {cid}')
            for field in ('audio_sha256', 'text_sha256'):
                require(isinstance(segment.get(field), str) and HASH.fullmatch(segment[field]), f'Invalid {field}: {cid}')
            require(file_hash(clip_path) == segment['audio_sha256'], f'Clip audio changed: {cid}')
            require(hashlib.sha256(segment['reference_text'].encode('utf-8')).hexdigest() == segment['text_sha256'],
                    f'Clip transcript changed: {cid}')
        source_start = segment.get('audio_start') if has_clip else segment.get('start')
        source_end = segment.get('audio_end') if has_clip else segment.get('end')
        require(number(source_start) and number(source_end) and 0 <= source_start < source_end,
                f'Invalid source interval: {cid}')
        require(number(lecture['duration_seconds']) and source_end <= lecture['duration_seconds'] + 1e-6,
                f'Interval exceeds source audio: {cid}')
        if has_clip:
            audio_url = f'clips/{cid}.wav'
        else:
            lecture_path = dataset / 'lectures' / f'{lid}.wav'
            require(lecture.get('audio') == str(lecture_path) and lecture_path.is_file() and not lecture_path.is_symlink(),
                    f'Missing normalized lecture audio: {lid}')
            audio_url = f'lectures/{lid}.wav#t={source_start:.6f},{source_end:.6f}'
        rows.append({
            'clip_id': cid, 'lecture_id': lid, 'title': lecture['title'], 'split': segment['split'],
            'passing': passing, 'reviewable': reviewable, 'automatic_status': automatic_status,
            'review_kind': segment.get('review_kind'), 'issues': issues, 'coverage_warnings': coverage_warnings,
            'lecture_warnings': lecture_warnings,
            'reference_text': segment.get('reference_text', ''), 'asr_text': segment.get('text', ''),
            'cer': segment.get('cer'), 'match_coverage': segment.get('match_coverage'),
            'duration_seconds': source_end - source_start, 'start': source_start, 'end': source_end,
            'audio_url': audio_url, 'has_clip': has_clip,
            'audio_sha256': segment.get('audio_sha256'), 'text_sha256': segment.get('text_sha256'),
        })
    require(bool(rows), 'No segments to review')
    if report:
        require(report.get('windows') == len(rows), 'Alignment report segment count differs')
    return {'schema': 1, 'segments_sha256': digest, 'rows': rows,
            'speaker_disjoint_verified': False,
            'source_lectures': len(lectures), 'training_ready': False}


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src 'self' file: blob:; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>Qwen3-ASR · Local dataset review</title>
<style>
:root{color-scheme:light;--ink:#172233;--muted:#596778;--line:#dce2e9;--paper:#fff;--blue:#184bb0;--pale:#edf3ff;--amber:#8c5500;--red:#a22a35;--green:#146344}
*{box-sizing:border-box}body{margin:0;background:#f4f6f9;color:var(--ink);font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1180px;margin:auto;padding:34px 22px 70px}h1{font-size:30px;letter-spacing:-.8px;line-height:1.2;margin:7px 0 14px}h2{font-size:18px;margin:0}p{margin:8px 0}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:1.4px;color:var(--blue);font-weight:750}.muted{color:var(--muted)}.instructions{background:var(--paper);border:1px solid var(--line);border-left:4px solid var(--blue);padding:16px 20px;border-radius:10px;margin:22px 0}.instructions ol{padding-left:21px;margin:6px 0}.instructions li{margin:5px 0}.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:20px 0}.stat{padding:14px 16px;background:var(--paper);border:1px solid var(--line);border-radius:10px}.stat strong{display:block;font-size:25px;font-variant-numeric:tabular-nums}.stat span{color:var(--muted);font-size:12px}.toolbar{background:var(--paper);padding:16px;border:1px solid var(--line);border-radius:12px;display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin:16px 0}.field{display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:650}.field.search{flex:1;min-width:200px}select,input,textarea,button{font:inherit;border:1px solid #b9c4d1;border-radius:7px;padding:8px 10px;background:white;color:var(--ink)}select,input{min-height:39px}button{cursor:pointer;font-weight:650;font-size:13px}button:hover{background:#eef2f7}button.primary{background:var(--blue);border-color:var(--blue);color:white}button.primary:hover{background:#113987}button:disabled{opacity:.4;cursor:default}button:focus-visible,select:focus-visible,input:focus-visible,textarea:focus-visible,a:focus-visible{outline:3px solid #9bb8f1;outline-offset:2px}.actions{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 5px}.statusline{min-height:24px;color:var(--muted);font-size:13px}.statusline.warning{color:var(--amber)}.pager{display:flex;justify-content:space-between;gap:10px;align-items:center;margin:15px 0;font-size:13px}.pager-buttons{display:flex;gap:8px}.card{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:21px;margin:15px 0;box-shadow:0 2px 5px #16253604}.card.quarantined{border-left:4px solid #c58b31}.card.approved{border-left:4px solid var(--green)}.card.rejected{border-left:4px solid var(--red)}.card.pending{border-left:4px solid #9ba8b9}.card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:14px}.tags{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}.badge{font-size:11px;line-height:1.4;font-weight:700;padding:4px 8px;border-radius:20px;background:#edf0f5;color:#465364}.badge.pass{background:#e8f6ee;color:var(--green)}.badge.quarantined{background:#fff2d9;color:var(--amber)}.badge.test{background:#f4eefe;color:#7048a1}.badge.validation{background:#e9f2ff;color:var(--blue)}.id{font:12px ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--muted);overflow-wrap:anywhere}.metrics{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:12px 0}.audio-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:12px 0}audio{height:38px;max-width:100%;width:430px}.audio-row a{font-size:12px;color:var(--blue)}.transcripts{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}.transcript{background:#f6f8fb;border-radius:8px;padding:14px;min-width:0}.transcript h3{margin:0 0 8px;font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:var(--muted)}.transcript p{white-space:pre-wrap;overflow-wrap:anywhere;font-size:15px;line-height:1.75;margin:0}.issues{color:var(--amber);font-size:13px;overflow-wrap:anywhere}.review-controls{display:grid;grid-template-columns:180px 1fr;gap:14px;align-items:start;border-top:1px solid var(--line);padding-top:16px;margin-top:16px}.review-controls textarea{width:100%;min-height:65px;resize:vertical;font-weight:400;font-size:13px}details{font-size:12px;color:var(--muted);margin-top:10px}details summary{cursor:pointer}details p{overflow-wrap:anywhere}.empty{text-align:center;padding:45px;color:var(--muted)}footer{font-size:12px;color:var(--muted);margin-top:30px;overflow-wrap:anywhere}.hidden{display:none!important}@media(max-width:720px){main{padding:22px 13px}.stats{grid-template-columns:repeat(2,1fr)}.transcripts{grid-template-columns:1fr}.review-controls{grid-template-columns:1fr}.card{padding:16px}.card-head{display:block}h1{font-size:25px}.toolbar{gap:10px}.field{flex:1}}
.stats{grid-template-columns:repeat(6,1fr)}.badge.review_required{background:#fff0d3;color:#825000}.boundary-note{background:#fff4de;border:1px solid #e5bd6f;border-radius:8px;padding:12px 14px;margin:14px 0;color:#704400;font-size:14px}.boundary-note strong{display:block;margin-bottom:4px}.card.boundary-review{border-top:3px solid #bf852b}@media(max-width:720px){.stats{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body><main>
<header><div class="eyebrow">Local dataset · Review before training</div><h1>Qwen3-ASR audio & transcript review</h1><p class="muted">Listen, check the reference, and record a decision for each candidate. All audio and review state stay on this computer.</p></header>
<section class="instructions" aria-label="Review instructions">
<strong>Approval means you checked the complete audio clip.</strong>
<ol><li>Listen from the first sound through the ending. Confirm the boundaries include all labeled speech and exclude neighboring speech.</li><li>Compare the reference transcript word for word with the audio, including Korean and English terms. ASR text is supporting evidence, not ground truth.</li><li>Approve only a correct pair. Reject errors and describe them in notes; label or boundary corrections require realignment and fresh review.</li><li>Review held-out validation and test labels too. Keep every lecture in its frozen split; reserve test data for final evaluation.</li></ol>
<p><strong>Boundary review required</strong> is a separate queue: internal word timestamps are uncertain. These clips have not passed every automatic check. Listen to the complete clip and verify both boundaries and the verbatim transcript before any approval.</p>
<p class="muted">Quarantined windows are available for diagnosis and cannot be approved here. Lecturer names do not establish verified speaker separation. No training starts from this page.</p>
</section>
<section class="stats" id="stats" aria-label="Review totals"></section>
<div class="toolbar">
<label class="field">Automatic checks<select id="automatic"><option value="reviewable">Reviewable candidates</option><option value="pass">Automatic pass</option><option value="review_required">Boundary review required</option><option value="quarantined">Quarantined</option><option value="all">All windows</option></select></label>
<label class="field">Split<select id="split"><option value="all">All splits</option><option value="train">Train</option><option value="validation">Validation</option><option value="test">Test · final evaluation</option></select></label>
<label class="field">Review status<select id="decision"><option value="all">All statuses</option><option value="pending">Pending</option><option value="approved">Approved</option><option value="rejected">Rejected</option></select></label>
<label class="field search">Search lecture, clip, or transcript<input id="search" type="search" placeholder="Search locally…" autocomplete="off"></label>
</div>
<div class="actions"><button class="primary" id="download">Download review</button><button id="download-decisions">Download decisions for exporter</button></div>
<p class="muted" style="font-size:12px">Review download contains every automatic-pass and eligible boundary-review candidate, including pending records. The exporter download contains only approved and rejected decisions.</p>
<p id="storage-status" class="statusline" role="status" aria-live="polite"></p>
<div class="pager"><span id="page-info"></span><div class="pager-buttons"><button id="previous">Previous</button><button id="next">Next</button></div></div>
<section id="cards" aria-label="Audio and transcript pairs"></section>
<footer><p id="fingerprint"></p><p>Review files bind each decision to its audio and transcript SHA-256 hashes. The export utility independently verifies provenance before creating training manifests. Browser storage is a convenience: download review records to keep a durable copy.</p></footer>
</main>
<script type="application/json" id="dataset">__DATASET_JSON__</script>
<script>
'use strict';
const data = JSON.parse(document.getElementById('dataset').textContent);
const rows = data.rows;
const candidates = rows.filter(row => row.reviewable);
const byId = new Map(candidates.map(row => [row.clip_id, row]));
const storageKey = 'qwen3-asr-review:v1:' + data.segments_sha256;
const statuses = new Set(['pending', 'approved', 'rejected']);
const state = new Map(candidates.map(row => [row.clip_id, {status:'pending', notes:''}]));
const statusLine = document.getElementById('storage-status');
let storageOkay = true;
let page = 0;
const perPage = 24;
let pendingSave;
function announce(message, warning=false) { statusLine.textContent = message; statusLine.classList.toggle('warning', warning); }
function reviewRecord(row) { const decision = state.get(row.clip_id); return {clip_id:row.clip_id, audio_sha256:row.audio_sha256, text_sha256:row.text_sha256, status:decision.status, notes:decision.notes}; }
function save() {
  try { localStorage.setItem(storageKey, JSON.stringify({schema:1, segments_sha256:data.segments_sha256, records:candidates.map(reviewRecord)})); storageOkay=true; announce('Saved in this browser. Download a copy to preserve your review.'); }
  catch (_) { storageOkay=false; announce('Browser storage is unavailable or full. Decisions remain in this open page; download them before closing.', true); }
}
try {
  const raw = localStorage.getItem(storageKey);
  if (raw !== null) {
    const saved = JSON.parse(raw);
    if (saved.schema !== 1 || saved.segments_sha256 !== data.segments_sha256 || !Array.isArray(saved.records)) throw new Error('Stale review');
    const restored = new Map();
    for (const record of saved.records) {
      const row = byId.get(record.clip_id);
      if (!row || restored.has(record.clip_id) || record.audio_sha256 !== row.audio_sha256 || record.text_sha256 !== row.text_sha256 || !statuses.has(record.status) || typeof record.notes !== 'string') throw new Error('Invalid saved review');
      restored.set(record.clip_id, {status:record.status, notes:record.notes});
    }
    for (const [id, value] of restored) state.set(id, value);
    announce('Restored matching review decisions from this browser.');
  } else announce('No saved decisions for this dataset. Every candidate starts pending.');
} catch (_) { storageOkay=false; announce('Saved browser state could not be loaded safely. All decisions start pending; download your work before closing.', true); }
function el(tag, className, text) { const element=document.createElement(tag); if(className) element.className=className; if(text !== undefined) element.textContent=text; return element; }
function time(seconds) { const value=Math.max(0, seconds); const minutes=Math.floor(value/60); return minutes + ':' + (value%60).toFixed(2).padStart(5,'0'); }
function percent(value) { return typeof value === 'number' && Number.isFinite(value) ? (value*100).toFixed(1)+'%' : 'unmapped'; }
function updateStats() {
  const count = key => candidates.filter(row => state.get(row.clip_id).status === key).length;
  const values = [['Automatic pass', rows.filter(row=>row.passing).length], ['Boundary review', rows.filter(row=>row.automatic_status==='review_required').length], ['Quarantined', rows.filter(row=>row.automatic_status==='quarantined').length], ['Pending', count('pending')], ['Approved', count('approved')], ['Rejected', count('rejected')]];
  const stats=document.getElementById('stats'); stats.replaceChildren();
  for(const [label, value] of values) { const box=el('div','stat'); box.append(el('strong','',value), el('span','',label)); stats.append(box); }
}
function filtered() {
  const automatic=document.getElementById('automatic').value;
  const split=document.getElementById('split').value;
  const decision=document.getElementById('decision').value;
  const search=document.getElementById('search').value.toLocaleLowerCase().trim();
  return rows.filter(row => {
    if(automatic === 'reviewable' && !row.reviewable) return false;
    if(!['all','reviewable'].includes(automatic) && row.automatic_status !== automatic) return false;
    if(split !== 'all' && row.split !== split) return false;
    if(decision !== 'all' && (!row.reviewable || state.get(row.clip_id).status !== decision)) return false;
    return !search || [row.title, row.clip_id, row.reference_text, row.asr_text].some(value => value.toLocaleLowerCase().includes(search));
  });
}
function makeCard(row) {
  const decision=row.reviewable ? state.get(row.clip_id) : null;
  const boundaryReview=row.automatic_status==='review_required';
  const card=el('article','card '+(decision ? decision.status : 'quarantined')+(boundaryReview?' boundary-review':''));
  const head=el('div','card-head');
  const heading=el('div'); heading.append(el('h2','',row.title),el('div','id',row.clip_id));
  const statusLabelText=row.passing?'Automatic pass':boundaryReview?'Boundary review required':'Quarantined';
  const tags=el('div','tags'); tags.append(el('span','badge '+row.automatic_status,statusLabelText),el('span','badge '+row.split,row.split));
  if(decision) tags.append(el('span','badge',decision.status));
  head.append(heading,tags); card.append(head);
  if(boundaryReview) { const notice=el('div','boundary-note'); notice.append(el('strong','','Boundary review required — internal word timestamps uncertain'),el('span','','This clip has not passed every automatic check. Listen to the full clip, confirm both audio boundaries, and verify every reference word before approving.')); card.append(notice); }
  const metrics=el('div','metrics');
  for(const text of [row.duration_seconds.toFixed(2)+' s', 'Source '+time(row.start)+'–'+time(row.end), 'Normalized CER '+percent(row.cer), 'Match coverage '+percent(row.match_coverage)]) metrics.append(el('span','',text));
  card.append(metrics);
  const audioRow=el('div','audio-row'); const audio=el('audio'); audio.controls=true; audio.preload='none'; audio.src=row.audio_url; audio.setAttribute('aria-label','Audio for '+row.clip_id);
  audio.addEventListener('play',()=> { document.querySelectorAll('audio').forEach(other=> { if(other!==audio) other.pause(); }); });
  if(!row.has_clip) {
    audio.addEventListener('loadedmetadata',()=> { if(audio.currentTime < row.start || audio.currentTime >= row.end) audio.currentTime=row.start; });
    audio.addEventListener('play',()=> { if(audio.currentTime < row.start || audio.currentTime >= row.end) audio.currentTime=row.start; });
    audio.addEventListener('timeupdate',()=> { if(audio.currentTime >= row.end && !audio.paused) audio.pause(); });
  }
  const link=el('a','',row.has_clip?'Open clip':'Open lecture interval'); link.href=row.audio_url; link.target='_blank'; link.rel='noopener';
  audioRow.append(audio,link); card.append(audioRow);
  if(!row.has_clip) card.append(el('p','muted','This quarantined window plays its interval in the full lecture; no approved clip exists.'));
  const transcripts=el('div','transcripts');
  for(const [label,text] of [['Reference · verify verbatim',row.reference_text],['Independent ASR · evidence only',row.asr_text]]) { const box=el('div','transcript'); box.append(el('h3','',label),el('p','',text || '(No mapped text)')); transcripts.append(box); }
  card.append(transcripts);
  if(row.issues.length) card.append(el('p','issues','Issues: '+row.issues.join(' · ')));
  if(row.coverage_warnings.length) card.append(el('p','issues','Coverage warnings: '+row.coverage_warnings.join(' · ')));
  const details=el('details'); details.append(el('summary','','Lecture warnings and provenance'));
  details.append(el('p','','Lecture: '+row.lecture_id),el('p','','Warnings: '+(row.lecture_warnings.join(' · ') || 'None recorded')));
  if(row.audio_sha256) details.append(el('p','id','Audio SHA-256: '+row.audio_sha256));
  if(row.text_sha256) details.append(el('p','id','Text SHA-256: '+row.text_sha256));
  card.append(details);
  if(decision) {
    const controls=el('div','review-controls');
    const statusLabel=el('label','field','Review decision'); const select=el('select'); select.setAttribute('aria-label','Decision for '+row.clip_id);
    for(const [value,label] of [['pending','Pending'],['approved','Approve'],['rejected','Reject']]) { const option=el('option','',label); option.value=value; select.append(option); }
    select.value=decision.status;
    select.addEventListener('change',()=> { decision.status=select.value; clearTimeout(pendingSave); save(); updateStats(); card.className='card '+decision.status+(boundaryReview?' boundary-review':''); tags.lastChild.textContent=decision.status; if(document.getElementById('decision').value!=='all') render(); });
    statusLabel.append(select);
    const notesLabel=el('label','field','Notes'); const notes=el('textarea'); notes.value=decision.notes; notes.placeholder='Corrections needed, uncertain words, or boundary issues'; notes.setAttribute('aria-label','Notes for '+row.clip_id);
    notes.addEventListener('input',()=> { decision.notes=notes.value; clearTimeout(pendingSave); pendingSave=setTimeout(save,250); }); notes.addEventListener('change',()=> { clearTimeout(pendingSave); save(); });
    notesLabel.append(notes); controls.append(statusLabel,notesLabel); card.append(controls);
  } else card.append(el('p','issues','Quarantined windows cannot be approved here. Correction and realignment are required before review.'));
  return card;
}
function render() {
  const visible=filtered(); const pages=Math.max(1,Math.ceil(visible.length/perPage)); page=Math.min(page,pages-1);
  const cards=document.getElementById('cards'); cards.querySelectorAll('audio').forEach(audio=>audio.pause()); cards.replaceChildren();
  if(!visible.length) cards.append(el('div','empty','No windows match these filters.'));
  else for(const row of visible.slice(page*perPage,(page+1)*perPage)) cards.append(makeCard(row));
  document.getElementById('page-info').textContent=visible.length+' matching windows · Page '+(page+1)+' of '+pages;
  document.getElementById('previous').disabled=page===0; document.getElementById('next').disabled=page>=pages-1;
}
for(const id of ['automatic','split','decision']) document.getElementById(id).addEventListener('change',()=> {page=0;render();});
document.getElementById('search').addEventListener('input',()=> {page=0;render();});
document.getElementById('previous').addEventListener('click',()=> {page--;render();});
document.getElementById('next').addEventListener('click',()=> {page++;render();});
function download(decisionsOnly) {
  clearTimeout(pendingSave); save();
  const records=candidates.map(reviewRecord).filter(record=>!decisionsOnly || record.status!=='pending');
  if(decisionsOnly && !records.length) {announce('No approved or rejected decisions yet. Listen and review a candidate first.',true);return;}
  const text=records.map(record=>JSON.stringify(record)).join('\n')+(records.length?'\n':'');
  const url=URL.createObjectURL(new Blob([text],{type:'application/x-ndjson;charset=utf-8'}));
  const anchor=el('a'); anchor.href=url; anchor.download=(decisionsOnly?'review-decisions-':'review-')+data.segments_sha256.slice(0,12)+'.jsonl';
  document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(()=>URL.revokeObjectURL(url),1000);
  announce('Downloaded '+records.length+' hash-bound '+(decisionsOnly?'decisions':'review records')+'.'+(storageOkay?'':' Browser storage remains unavailable.'),!storageOkay);
}
document.getElementById('download').addEventListener('click',()=>download(false));
document.getElementById('download-decisions').addEventListener('click',()=>download(true));
window.addEventListener('pagehide',()=> {if(pendingSave){clearTimeout(pendingSave);save();}});
document.getElementById('fingerprint').textContent=data.source_lectures+' source lectures · Dataset SHA-256: '+data.segments_sha256;
updateStats(); render();
</script></body></html>
'''


def build_review(dataset):
    dataset = Path(dataset).resolve()
    payload = build_payload(dataset)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    # Embedded application/json is still parsed by HTML before JSON.parse.
    encoded = (encoded.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
               .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))
    destination = dataset / 'review.html'
    temporary = dataset / '.review.html.tmp'
    require(not destination.is_symlink() and not temporary.is_symlink(), 'Refusing symlinked review output')
    temporary.write_text(HTML.replace('__DATASET_JSON__', encoded), encoding='utf-8')
    temporary.replace(destination)
    return {'review_page': str(destination), 'segments_sha256': payload['segments_sha256'],
            'pass_candidates': sum(row['passing'] for row in payload['rows']),
            'boundary_review_candidates': sum(row['automatic_status'] == 'review_required' for row in payload['rows']),
            'quarantined_windows': sum(row['automatic_status'] == 'quarantined' for row in payload['rows']),
            'training_ready': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/qwen3_asr')
    args = parser.parse_args()
    try:
        result = build_review(args.dataset)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f'Review page refused: {error}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
