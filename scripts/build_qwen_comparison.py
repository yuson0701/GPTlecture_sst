"""Build a self-contained, offline Qwen transcript comparison from comparison.json.

Usage: python scripts/build_qwen_comparison.py --dataset data/qwen_transcripts
Audio and downloadable transcripts are local files relative to the generated page.
"""

import argparse
import json
import math
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def local_path(value):
    """Allow sibling dataset paths, while excluding network and executable URLs."""
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        return False
    parsed = urlsplit(value)
    return not (parsed.scheme or parsed.netloc or parsed.query or parsed.fragment
                or value.startswith(('/', '\\')) or '\\' in value)


def validate_payload(data):
    require(isinstance(data, dict), 'comparison.json must contain an object')
    require(isinstance(data.get('model'), str), 'Missing model name')
    require(isinstance(data.get('method_note'), str), 'Missing method note')
    lectures = data.get('lectures')
    require(isinstance(lectures, list) and lectures, 'No lectures to compare')
    seen = set()
    for row in lectures:
        lid = row.get('lecture_id')
        require(isinstance(lid, str) and lid and lid not in seen, 'Missing or duplicate lecture ID')
        seen.add(lid)
        for key in ('title', 'reference_text', 'qwen_text'):
            require(isinstance(row.get(key), str), f'Invalid {key}: {lid}')
        require(row.get('source_split') in ('train', 'validation', 'test'), f'Invalid split: {lid}')
        require(number(row.get('duration_seconds')) and row['duration_seconds'] > 0,
                f'Invalid duration: {lid}')
        require(row.get('cer') is None or (number(row['cer']) and row['cer'] >= 0), f'Invalid CER: {lid}')
        for key in ('reference_characters', 'qwen_characters'):
            require(type(row.get(key)) is int and row[key] >= 0, f'Invalid {key}: {lid}')
        require(local_path(row.get('audio')), f'Audio must be a local relative path: {lid}')
        require(isinstance(row.get('files'), dict), f'Missing transcript files: {lid}')
        for key in ('original_txt', 'qwen_txt', 'qwen_timed_txt'):
            require(local_path(row['files'].get(key)), f'Invalid {key} download path: {lid}')
        warnings = row.get('warnings', [])
        require(isinstance(warnings, list) and all(isinstance(w, str) for w in warnings),
                f'Invalid warnings: {lid}')
        diffs = row.get('diff')
        require(isinstance(diffs, list), f'Missing diff: {lid}')
        for chunk in diffs:
            require(isinstance(chunk, dict) and chunk.get('tag') in ('equal', 'replace', 'insert', 'delete'),
                    f'Invalid diff tag: {lid}')
            require(isinstance(chunk.get('original'), str) and isinstance(chunk.get('qwen'), str),
                    f'Invalid diff text: {lid}')
            require(chunk['tag'] != 'equal' or chunk['original'] == chunk['qwen'],
                    f'Equal diff text differs: {lid}')
            require(chunk['tag'] != 'insert' or not chunk['original'], f'Invalid insertion: {lid}')
            require(chunk['tag'] != 'delete' or not chunk['qwen'], f'Invalid deletion: {lid}')
        require(''.join(d['original'] for d in diffs) == row['reference_text'],
                f'Diff does not preserve original text: {lid}')
        require(''.join(d['qwen'] for d in diffs) == row['qwen_text'],
                f'Diff does not preserve Qwen text: {lid}')
        segments = row.get('segments')
        require(isinstance(segments, list), f'Missing transcript windows: {lid}')
        last_start = -1
        for segment in segments:
            require(isinstance(segment, dict) and isinstance(segment.get('text'), str),
                    f'Invalid transcript window: {lid}')
            start, end = segment.get('start'), segment.get('end')
            require(number(start) and number(end) and 0 <= start < end <= row['duration_seconds'] + 0.1,
                    f'Invalid transcript interval: {lid}')
            require(start >= last_start, f'Transcript windows are not chronological: {lid}')
            last_start = start
    examples = data.get('examples', [])
    require(isinstance(examples, list), 'Invalid examples')
    for example in examples:
        require(isinstance(example, dict) and example.get('lecture_id') in seen, 'Invalid example lecture')
        require(all(isinstance(example.get(key), str) for key in ('original', 'qwen', 'note')),
                'Invalid example text')
        require(number(example.get('window_start')) and example['window_start'] >= 0,
                'Invalid example timestamp')
    return data


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src 'self' file:; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>Qwen · Lecture transcript comparison</title>
<style>
:root{color-scheme:light;--ink:#15293c;--muted:#5c6b78;--line:#dce3e8;--blue:#225a84;--bluepale:#eaf3f9;--paper:#fff;--bg:#f3f6f7;--old:#fff0e4;--new:#e2f3ec;--accent:#a15b21}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--bg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}button,input,select{font:inherit}button,a,input,summary{touch-action:manipulation}button{cursor:pointer;border:1px solid var(--line);border-radius:8px;padding:8px 13px;background:white;color:var(--ink)}button:hover{background:var(--bluepale)}button:disabled{opacity:.4;cursor:default}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #8bc0e3;outline-offset:3px}a{color:var(--blue);text-underline-offset:3px}h1,h2,h3,p{margin-top:0}h1{font-size:31px;line-height:1.25;letter-spacing:-.6px;margin-bottom:10px}h2{font-size:22px;line-height:1.35;margin-bottom:8px}h3{font-size:16px;margin-bottom:12px}.top{background:#15344c;color:#fff;padding:34px max(24px,calc((100vw - 1480px)/2)) 28px}.top p{color:#c4d7e4;margin:0;max-width:880px}.eyebrow{font-size:11px;letter-spacing:1.8px;text-transform:uppercase;font-weight:750;margin-bottom:9px;color:#a5cbdf}.shell{max-width:1480px;margin:auto;padding:24px;display:grid;grid-template-columns:248px minmax(0,1fr);gap:26px}.sidebar{position:sticky;top:20px;align-self:start}.sidebar h2{font-size:11px;text-transform:uppercase;letter-spacing:1.3px;color:var(--muted);margin:0 0 10px}.lecture-list{display:flex;flex-direction:column;gap:7px}.lecture-option{padding:12px;text-align:left;line-height:1.45;background:#ffffff70;display:block;width:100%}.lecture-option strong{display:block;font-size:13px;font-weight:650}.lecture-option small{display:block;color:var(--muted);margin-top:5px;font-size:11px}.lecture-option[aria-current=true]{background:white;border-color:var(--blue);box-shadow:inset 3px 0 var(--blue)}.sidebar-note{font-size:11px;color:var(--muted);margin-top:17px}.tabs{display:flex;gap:4px;border-bottom:1px solid var(--line);margin-bottom:22px}.tabs button{border:0;border-radius:5px 5px 0 0;background:none;padding:11px 16px;font-weight:600;color:var(--muted)}.tabs button[aria-selected=true]{color:var(--blue);box-shadow:inset 0 -3px var(--blue);background:#fff9}.content{min-width:0}.muted{color:var(--muted)}.small{font-size:12px}.panel{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:22px;margin-bottom:18px}.note{font-size:13px;color:var(--muted);margin-bottom:0}.metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:20px 0}.metric{padding:20px;border:1px solid var(--line);background:white;border-radius:10px}.metric strong{display:block;font-size:29px;line-height:1.25;font-weight:650;letter-spacing:-.7px}.metric span{display:block;margin-top:7px;font-size:12px;color:var(--muted)}.metric em{display:block;font-style:normal;font-size:11px;color:var(--muted);margin-top:2px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th{text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.65px;color:var(--muted);font-weight:650;white-space:nowrap}td,th{border-bottom:1px solid var(--line);padding:14px 9px;vertical-align:top}td:first-child,th:first-child{padding-left:0}td:last-child,th:last-child{padding-right:0}tr:last-child td{border-bottom:0}.number{text-align:right;font-variant-numeric:tabular-nums}.row-link{padding:0;border:0;border-radius:0;background:none;text-align:left;color:var(--blue);font-weight:650}.row-link:hover{text-decoration:underline;background:none}.split{font-size:10px;display:inline-block;border-radius:20px;padding:3px 8px;background:#edf1f4;white-space:nowrap;color:var(--muted)}.split.validation{background:#e9f0fc;color:#3a5f9b}.split.test{background:#f0eaf7;color:#765397}.lecture-head{display:flex;gap:18px;align-items:start;justify-content:space-between;margin-bottom:17px}.lecture-head h2{overflow-wrap:anywhere}.score{text-align:right;min-width:150px}.score strong{font-size:30px;line-height:1.2;display:block;color:var(--accent);font-weight:650}.score span{display:block;font-size:11px;color:var(--muted)}.downloads{display:flex;flex-wrap:wrap;gap:9px 18px;font-size:12px;margin-top:14px}.audio{display:grid;grid-template-columns:minmax(200px,1fr) auto;gap:12px;align-items:center}.audio audio{width:100%;height:40px}.audio-label{font-size:12px;color:var(--muted);min-height:20px;margin:8px 0 0}.warnings{margin:15px 0 0;padding:12px 16px;background:#fff7e7;border:1px solid #eddfbb;border-radius:8px;font-size:12px;color:#735224}.warnings ul{margin:0;padding-left:18px}.toolbar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:18px 0 12px}.search{flex:1;min-width:210px}.search input{width:100%;border:1px solid var(--line);border-radius:8px;padding:9px 12px;background:white}.checkbox{display:flex;gap:7px;align-items:center;font-size:13px;white-space:nowrap;cursor:pointer}.legend{font-size:12px;display:flex;gap:15px;flex-wrap:wrap;margin-bottom:12px;color:var(--muted)}.swatch{display:inline-block;width:12px;height:12px;vertical-align:-1px;margin-right:5px;border-radius:2px}.swatch.original{background:#f6cfad}.swatch.qwen{background:#a8d9c2}.pager{display:flex;justify-content:space-between;align-items:center;gap:12px;font-size:12px;color:var(--muted);margin:12px 0}.pager-buttons{display:flex;gap:7px}.pager button{padding:6px 11px;font-size:12px}.diff-head,.diff-columns{display:grid;grid-template-columns:1fr 1fr;gap:0}.diff-head{border:1px solid var(--line);border-radius:8px;padding:10px 17px;background:#eaf0f3;font-size:11px;letter-spacing:.5px;font-weight:650}.diff-head span:last-child{padding-left:21px}.diff-block{margin-top:11px;background:white;border:1px solid var(--line);border-radius:10px;overflow:hidden}.block-label{font-size:10px;color:var(--muted);padding:7px 17px;background:#f9fbfc;border-bottom:1px solid #edf1f4;text-transform:uppercase;letter-spacing:.65px}.diff-text{padding:17px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.85;font-size:14px;min-width:0}.diff-text+.diff-text{border-left:1px solid var(--line)}.changed-original{background:var(--old);color:#804315;border-radius:2px}.changed-qwen{background:var(--new);color:#176148;border-radius:2px}.empty-text{color:#8b959d;font-style:italic;font-size:12px}.no-results{padding:40px 20px;text-align:center;border:1px dashed var(--line);border-radius:10px;color:var(--muted);background:white}.whole-text{margin-top:20px}.whole-text details{background:white;border:1px solid var(--line);border-radius:9px;margin-top:10px}.whole-text summary{cursor:pointer;padding:12px 16px;font-size:13px;font-weight:650}.whole-text pre{font:14px/1.85 inherit;white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:4px 16px 20px;max-height:600px;overflow:auto}.window{background:white;border:1px solid var(--line);border-radius:10px;padding:17px;margin-top:11px;display:grid;grid-template-columns:130px minmax(0,1fr);gap:18px}.window-time{font-size:12px;color:var(--blue);font-variant-numeric:tabular-nums}.window-time button{display:block;font-size:11px;margin-top:7px;padding:5px 10px;color:var(--blue)}.window-text{font-size:14px;line-height:1.85;white-space:pre-wrap;overflow-wrap:anywhere}.window.playing{border-color:#5e9ebc;background:#fbfdff;box-shadow:inset 3px 0 #5e9ebc}.flags{font-size:10px;display:block;margin-top:8px;color:#935b18}.footer{font-size:11px;color:var(--muted);margin-top:25px}.hidden{display:none!important}mark{background:#ffe081;color:#3b3100;padding:0;border-radius:2px}@media(max-width:950px){.shell{grid-template-columns:200px minmax(0,1fr);gap:17px;padding:18px}.metric{padding:15px}.metric strong{font-size:25px}.panel{padding:17px}.diff-text{padding:13px}.window{grid-template-columns:110px minmax(0,1fr);gap:13px}}@media(max-width:720px){.top{padding:25px 17px}.top h1{font-size:26px}.shell{display:block;padding:16px}.sidebar{position:static;margin-bottom:17px}.lecture-list{display:flex;flex-direction:row;overflow:auto;padding-bottom:7px}.lecture-option{min-width:195px;width:195px;flex-shrink:0}.sidebar-note{display:none}.tabs button{font-size:13px;padding:10px 11px}.metric-grid{gap:8px}.metric{padding:12px 9px}.metric strong{font-size:22px}.metric span{font-size:11px}.lecture-head{display:block}.score{text-align:left;margin-top:10px}.score strong{display:inline;font-size:25px}.score span{display:inline;margin-left:8px}.audio{grid-template-columns:1fr}.diff-head{display:none}.diff-columns{grid-template-columns:1fr}.diff-text+.diff-text{border-left:0;border-top:1px dashed var(--line)}.diff-text::before{content:attr(data-label);display:block;font-size:10px;font-weight:700;letter-spacing:.7px;text-transform:uppercase;color:var(--muted);margin-bottom:5px}.window{grid-template-columns:1fr;gap:9px}.window-time button{display:inline-block;margin:0 0 0 9px}.flags{display:inline;margin-left:8px}.pager{align-items:start}.legend{gap:8px}.toolbar{gap:9px}}
.examples-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:0 0 20px}.example{padding:16px;background:white;border:1px solid var(--line);border-radius:10px}.example h3{font-size:12px;margin-bottom:12px}.example-label{font-size:9px;font-weight:700;letter-spacing:.9px;text-transform:uppercase;color:var(--muted);margin-bottom:4px}.example-text{font-size:12px;line-height:1.8;padding:7px 9px;border-radius:5px;margin-bottom:10px;overflow-wrap:anywhere}.example-text.original{background:var(--old)}.example-text.qwen{background:var(--new)}.example p{font-size:11px;color:var(--muted);margin-bottom:10px}.example button{font-size:11px;padding:5px 9px}.whole-text pre{font-family:inherit;font-size:14px;line-height:1.85}@media(max-width:950px){.examples-grid{grid-template-columns:1fr}.example{display:block}}
</style>
</head>
<body>
<header class="top"><div class="eyebrow">Local audio · Qwen3-ASR</div><h1>What did Qwen hear?</h1><p>Qwen audio transcripts, alongside the original lecture scripts. Explore wording changes and listen to the source audio.</p></header>
<div class="shell">
<aside class="sidebar"><h2>Lectures</h2><nav id="lecture-list" class="lecture-list" aria-label="Choose a lecture"></nav><p class="sidebar-note">All transcripts and audio stay local. Evaluation lectures retain their reserved splits.</p></aside>
<main class="content">
<nav class="tabs" role="tablist" aria-label="Comparison views"><button id="tab-overview" role="tab" aria-selected="true" aria-controls="overview">Overview</button><button id="tab-difference" role="tab" aria-selected="false" aria-controls="difference">Differences</button><button id="tab-transcript" role="tab" aria-selected="false" aria-controls="transcript">Qwen transcript</button></nav>
<section id="overview" role="tabpanel" aria-labelledby="tab-overview">
<h2>Every lecture, compared</h2><p class="muted small">The percentage measures textual difference from the supplied script. It is <strong>not transcription accuracy</strong>; listening is needed to establish which wording is correct.</p>
<div id="metrics" class="metric-grid"></div>
<div id="examples" class="examples-grid" aria-label="Examples of transcript differences"></div>
<div class="panel"><div class="table-wrap"><table><thead><tr><th>Lecture</th><th>Reserved split</th><th class="number">Duration</th><th class="number">Character difference</th><th class="number">Original → Qwen<br>normalized characters</th></tr></thead><tbody id="overview-rows"></tbody></table></div></div>
<div class="panel"><h3>How to read this comparison</h3><p id="method-note" class="note"></p><p id="source-note" class="note" style="margin-top:10px"></p><p class="note" style="margin-top:10px">Character difference is the normalized character edit distance (CER): substitutions + deletions + insertions, divided by the reference character count. It can exceed 100%. The highlights use word, punctuation, and whitespace differences, so their boundaries do not represent the CER calculation.</p><p class="note" style="margin-top:10px">Script metadata, speaker labels, and non-speech annotations are omitted from the comparison. Korean and English wording is kept without paraphrasing.</p></div>
</section>
<div id="selected-lecture" class="hidden">
<section class="panel"><div class="lecture-head"><div><h2 id="lecture-title"></h2><div id="lecture-meta" class="muted small"></div></div><div class="score"><strong id="lecture-score"></strong><span>character difference · not accuracy</span></div></div><div class="audio"><audio id="audio" controls preload="none"></audio><button id="play-full">Play full lecture</button></div><p id="audio-label" class="audio-label" aria-live="polite"></p><div id="downloads" class="downloads"></div><div id="warnings" class="warnings hidden"></div></section>
<section id="difference" role="tabpanel" aria-labelledby="tab-difference" class="hidden"><p class="note">Highlighted wording differs between the two texts. Nearby edits are grouped into readable passages. These passages have no verified audio alignment.</p><div class="toolbar"><label class="search"><span class="hidden">Search differences</span><input id="diff-search" type="search" placeholder="Search either text…" aria-label="Search either comparison text"></label><label class="checkbox"><input id="changes-only" type="checkbox" checked>Only passages with changes</label></div><div class="legend"><span><i class="swatch original"></i>Original wording removed or replaced</span><span><i class="swatch qwen"></i>Qwen wording added or replaced</span></div><div class="pager"><span id="diff-page-info" role="status"></span><div class="pager-buttons"><button id="diff-prev">Previous</button><button id="diff-next">Next</button></div></div><div class="diff-head"><span>ORIGINAL SCRIPT</span><span>QWEN TRANSCRIPT</span></div><div id="diff-blocks"></div><div class="whole-text"><details id="whole-original"><summary>Read the complete original comparison text</summary><pre></pre></details><details id="whole-qwen"><summary>Read the complete Qwen transcript</summary><pre></pre></details></div></section>
<section id="transcript" role="tabpanel" aria-labelledby="tab-transcript" class="hidden"><h3>Listen alongside Qwen’s transcript</h3><p class="note">Times identify the audio windows supplied to Qwen, usually 30 seconds. They are coarse window boundaries, <strong>not aligned word or sentence timestamps</strong>. Shorter windows may appear after a retry.</p><div class="toolbar"><label class="search"><input id="transcript-search" type="search" placeholder="Search Qwen’s transcript…" aria-label="Search Qwen transcript"></label></div><div class="pager"><span id="transcript-page-info" role="status"></span><div class="pager-buttons"><button id="transcript-prev">Previous</button><button id="transcript-next">Next</button></div></div><div id="transcript-windows"></div></section>
</div>
<footer class="footer"><span id="model"></span> · Local, offline comparison</footer>
</main></div>
<script id="dataset" type="application/json">__DATASET_JSON__</script>
<script>
'use strict';
const data = JSON.parse(document.getElementById('dataset').textContent);
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => {const node = document.createElement(tag); if(cls) node.className=cls; if(text !== undefined) node.textContent=text; return node;};
const percent = value => value === null ? '—' : (value*100).toFixed(1)+'%';
const count = value => value.toLocaleString();
const clock = seconds => {const s=Math.floor(seconds); return (s>=3600?Math.floor(s/3600)+':':'')+String(Math.floor(s/60)%60).padStart(2,'0')+':'+String(s%60).padStart(2,'0');};
const duration = seconds => {const m=Math.round(seconds/60); return m>=60?Math.floor(m/60)+'h '+m%60+'m':m+'m';};
let selected=0, view='overview', diffPage=0, transcriptPage=0, activeWindow=null;
const blockCache=new Map(), DIFF_PAGE_SIZE=8, TRANSCRIPT_PAGE_SIZE=18;
const audio=$('audio');
function splitText(text, maximum){if(text.length<=maximum)return [text,''];const prefix=text.slice(0,maximum);const found=Math.max(prefix.lastIndexOf('\n'),prefix.lastIndexOf('. ')+1,prefix.lastIndexOf(' '));const cut=found>maximum*.55?found+1:maximum;return [text.slice(0,cut),text.slice(cut)];}
function buildBlocks(parts){
  // Pack lexical edits into passages. Text order and whitespace remain lossless.
  const blocks=[];let current=[],size=0;
  const flush=()=>{if(current.length){blocks.push({parts:current,changed:current.some(p=>p.tag!=='equal'),index:blocks.length});current=[];size=0;}};
  for(const originalPart of parts){let a=originalPart.original,b=originalPart.qwen;
    while(a.length||b.length){const room=1400-size;if(room<150)flush();const available=1400-size;
      let headA,tailA,headB,tailB;
      if(originalPart.tag==='equal'){[headA,tailA]=splitText(a,available);headB=headA;tailB=tailA;}
      else{[headA,tailA]=splitText(a,available);[headB,tailB]=splitText(b,available);}
      current.push({tag:originalPart.tag,original:headA,qwen:headB});size+=Math.max(headA.length,headB.length);a=tailA;b=tailB;
      if(a.length||b.length||size>=1300||(size>=750&&/[.!?。？！]\s*$|\n\s*$/.test(headA+headB)))flush();
    }
  }
  flush();return blocks;
}
function highlighted(parent,text,query){
  if(!query){parent.append(document.createTextNode(text));return;}
  const lower=text.toLocaleLowerCase();let offset=0,index;
  while((index=lower.indexOf(query,offset))!==-1){parent.append(document.createTextNode(text.slice(offset,index)));parent.append(el('mark','',text.slice(index,index+query.length)));offset=index+query.length;}
  parent.append(document.createTextNode(text.slice(offset)));
}
function pager(prefix,page,total,size,noun){const pages=Math.ceil(total/size);$(prefix+'-page-info').textContent=total?`${page*size+1}–${Math.min((page+1)*size,total)} of ${count(total)} ${noun}`:`No matching ${noun}`;$(prefix+'-prev').disabled=page===0;$(prefix+'-next').disabled=page+1>=pages;}
function clearAudio(){audio.pause();activeWindow=null;$('audio-label').textContent='Full lecture audio';}
function selectLecture(index,nextView='difference'){
  if(selected!==index||!audio.getAttribute('src')){clearAudio();selected=index;const row=data.lectures[selected];audio.src=row.audio;diffPage=0;transcriptPage=0;$('diff-search').value='';$('transcript-search').value='';
    $('lecture-title').textContent=row.title;$('lecture-meta').replaceChildren(el('span','split '+row.source_split,row.source_split),document.createTextNode(' · '+duration(row.duration_seconds)+' · '+count(row.segments.length)+' audio windows'));
    $('lecture-score').textContent=percent(row.cer);$('downloads').replaceChildren();for(const [key,label] of [['original_txt','Original text'],['qwen_txt','Qwen text'],['qwen_timed_txt','Qwen text with window times']]){const a=el('a','',label+' ↓');a.href=row.files[key];a.download='';$('downloads').append(a);}
    const warnings=row.warnings||[];$('warnings').classList.toggle('hidden',!warnings.length);$('warnings').replaceChildren();if(warnings.length){const list=el('ul');warnings.forEach(w=>list.append(el('li','',w)));$('warnings').append(list);}
    for(const key of ['whole-original','whole-qwen']){$(key).open=false;$(key).querySelector('pre').textContent='';}
  }
  document.querySelectorAll('.lecture-option').forEach((node,i)=>node.setAttribute('aria-current',String(i===selected)));setView(nextView);
}
function setView(next){view=next;for(const name of ['overview','difference','transcript']){$('tab-'+name).setAttribute('aria-selected',String(name===view));$(name).classList.toggle('hidden',name!==view);}$('selected-lecture').classList.toggle('hidden',view==='overview');if(view==='difference')renderDiff();if(view==='transcript')renderTranscript();}
function renderDiff(){
  const row=data.lectures[selected];if(!blockCache.has(row.lecture_id))blockCache.set(row.lecture_id,buildBlocks(row.diff));const query=$('diff-search').value.trim().toLocaleLowerCase();
  const blocks=blockCache.get(row.lecture_id).filter(block=>(!$('changes-only').checked||block.changed)&&(!query||['original','qwen'].some(key=>block.parts.map(p=>p[key]).join('').toLocaleLowerCase().includes(query))));
  diffPage=Math.min(diffPage,Math.max(0,Math.ceil(blocks.length/DIFF_PAGE_SIZE)-1));pager('diff',diffPage,blocks.length,DIFF_PAGE_SIZE,'passages');const target=$('diff-blocks');target.replaceChildren();
  for(const block of blocks.slice(diffPage*DIFF_PAGE_SIZE,(diffPage+1)*DIFF_PAGE_SIZE)){const card=el('article','diff-block');card.append(el('div','block-label','Passage '+(block.index+1)+(block.changed?' · wording differs':' · unchanged')));const columns=el('div','diff-columns');
    for(const [key,label] of [['original','Original script'],['qwen','Qwen transcript']]){const text=el('div','diff-text');text.dataset.label=label;let hasText=false;for(const part of block.parts){if(!part[key])continue;hasText=true;const span=el('span',part.tag==='equal'?'':'changed-'+key);highlighted(span,part[key],query);text.append(span);}if(!hasText)text.append(el('span','empty-text','No text on this side'));columns.append(text);}
    card.append(columns);target.append(card);
  }
  if(!blocks.length)target.append(el('p','no-results',query?'No passages match this search.':$('changes-only').checked?'The comparison contains no differing passages.':'No text to compare.'));
}
function playWindow(index){const segment=data.lectures[selected].segments[index];activeWindow={index,start:segment.start,end:segment.end};audio.currentTime=segment.start;$('audio-label').textContent='Playing Qwen input window '+clock(segment.start)+'–'+clock(segment.end)+' · coarse boundaries';audio.play().catch(()=>{$('audio-label').textContent='Press play in the audio controls to listen to this window.';});updatePlaying();}
function updatePlaying(){document.querySelectorAll('.window').forEach(node=>node.classList.toggle('playing',!!activeWindow&&Number(node.dataset.index)===activeWindow.index));}
function renderTranscript(){
  const query=$('transcript-search').value.trim().toLocaleLowerCase();const segments=data.lectures[selected].segments.map((segment,index)=>({segment,index})).filter(({segment})=>!query||segment.text.toLocaleLowerCase().includes(query));
  transcriptPage=Math.min(transcriptPage,Math.max(0,Math.ceil(segments.length/TRANSCRIPT_PAGE_SIZE)-1));pager('transcript',transcriptPage,segments.length,TRANSCRIPT_PAGE_SIZE,'audio windows');const target=$('transcript-windows');target.replaceChildren();
  for(const {segment,index} of segments.slice(transcriptPage*TRANSCRIPT_PAGE_SIZE,(transcriptPage+1)*TRANSCRIPT_PAGE_SIZE)){const card=el('article','window');card.dataset.index=index;const time=el('div','window-time',clock(segment.start)+'–'+clock(segment.end));const play=el('button','','Play window');play.addEventListener('click',()=>playWindow(index));time.append(play);const flags=[];if(segment.truncated)flags.push('Output may be truncated');if(segment.retried)flags.push('Retried window');if(flags.length)time.append(el('span','flags',flags.join(' · ')));const text=el('div','window-text');if(segment.text)highlighted(text,segment.text,query);else text.append(el('span','empty-text','Qwen returned no text for this window.'));card.append(time,text);target.append(card);}
  if(!segments.length)target.append(el('p','no-results','No transcript windows match this search.'));updatePlaying();
}
for(const [index,row] of data.lectures.entries()){
  const option=el('button','lecture-option');option.append(el('strong','',row.title),el('small','',duration(row.duration_seconds)+' · '+row.source_split+' · '+percent(row.cer)+' difference'));option.setAttribute('aria-current',String(index===selected));option.addEventListener('click',()=>selectLecture(index,view==='overview'?'difference':view));$('lecture-list').append(option);
  const tr=el('tr'),title=el('td'),link=el('button','row-link',row.title);link.addEventListener('click',()=>selectLecture(index));title.append(link);tr.append(title);const split=el('td');split.append(el('span','split '+row.source_split,row.source_split));tr.append(split,el('td','number',duration(row.duration_seconds)),el('td','number',percent(row.cer)),el('td','number',count(row.reference_characters)+' → '+count(row.qwen_characters)));$('overview-rows').append(tr);
}
const totalSeconds=data.lectures.reduce((sum,row)=>sum+row.duration_seconds,0),scored=data.lectures.filter(row=>row.cer!==null),totalChars=scored.reduce((sum,row)=>sum+row.reference_characters,0),weighted=totalChars?scored.reduce((sum,row)=>sum+row.cer*row.reference_characters,0)/totalChars:null;
for(const [value,label,note] of [[String(data.lectures.length),'lectures transcribed','Separate evaluation lectures preserved'],[duration(totalSeconds),'source audio','All audio windows included'],[percent(weighted),'weighted character difference','Difference from script, not accuracy']]){const metric=el('div','metric');metric.append(el('strong','',value),el('span','',label),el('em','',note));$('metrics').append(metric);}
for(const example of data.examples||[]){const index=data.lectures.findIndex(row=>row.lecture_id===example.lecture_id),row=data.lectures[index],card=el('article','example');card.append(el('h3','',row.title));for(const [key,label] of [['original','Original'],['qwen','Qwen']])card.append(el('div','example-label',label),el('div','example-text '+key,example[key]));card.append(el('p','',example.note));const button=el('button','','Listen near '+clock(example.window_start));button.addEventListener('click',()=>{selectLecture(index,'transcript');const segmentIndex=row.segments.findIndex(segment=>segment.start<=example.window_start&&segment.end>example.window_start);if(segmentIndex>=0){transcriptPage=Math.floor(segmentIndex/TRANSCRIPT_PAGE_SIZE);renderTranscript();playWindow(segmentIndex);$('audio').scrollIntoView({block:'center'});}});card.append(button);$('examples').append(card);}
$('method-note').textContent=data.method_note;$('source-note').textContent=data.source_note||'';$('model').textContent=data.model;
for(const name of ['overview','difference','transcript'])$('tab-'+name).addEventListener('click',()=>selectLecture(selected,name));
for(const [prefix,render] of [['diff',renderDiff],['transcript',renderTranscript]]){
  $(prefix+'-search').addEventListener('input',()=>{if(prefix==='diff')diffPage=0;else transcriptPage=0;render();});
  for(const [direction,delta] of [['prev',-1],['next',1]])$(prefix+'-'+direction).addEventListener('click',()=>{if(prefix==='diff')diffPage+=delta;else transcriptPage+=delta;render();$(prefix+'-page-info').scrollIntoView({block:'nearest'});});
}
$('changes-only').addEventListener('change',()=>{diffPage=0;renderDiff();});
for(const [key,field] of [['whole-original','reference_text'],['whole-qwen','qwen_text']])$(key).addEventListener('toggle',()=>{if($(key).open)$(key).querySelector('pre').textContent=data.lectures[selected][field];});
$('play-full').addEventListener('click',()=>{activeWindow=null;audio.currentTime=0;$('audio-label').textContent='Full lecture audio';updatePlaying();audio.play().catch(()=>{$('audio-label').textContent='Press play in the audio controls to listen.';});});
audio.addEventListener('timeupdate',()=>{if(activeWindow&&audio.currentTime>=activeWindow.end){audio.pause();activeWindow=null;updatePlaying();$('audio-label').textContent='Window finished. Select another window or continue with the audio controls.';}});
audio.addEventListener('seeked',()=>{if(activeWindow&&(audio.currentTime<activeWindow.start-.1||audio.currentTime>activeWindow.end+.1)){activeWindow=null;updatePlaying();$('audio-label').textContent='Full lecture audio';}});
audio.addEventListener('error',()=>{$('audio-label').textContent='The local audio could not be opened. Keep this page beside its transcript folder and source audio.';});
selectLecture(0,'overview');
</script>
</body></html>
'''


def build_page(dataset):
    dataset = Path(dataset).resolve()
    data = validate_payload(json.loads((dataset / 'comparison.json').read_text(encoding='utf-8')))
    encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    # Do not allow transcript text to close the JSON script element.
    encoded = encoded.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    encoded = encoded.replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    output = dataset / 'comparison.html'
    output.write_text(HTML.replace('__DATASET_JSON__', encoded), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data' / 'qwen_transcripts')
    args = parser.parse_args()
    print(build_page(args.dataset))


if __name__ == '__main__':
    main()
