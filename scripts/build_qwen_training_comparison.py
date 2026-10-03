"""Build an offline original/base/fine-tuned viewer from a completed evaluation.

No model is loaded. Prediction files and local audio remain unchanged.
"""
import argparse
from difflib import SequenceMatcher
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import unicodedata
from urllib.parse import quote, urlsplit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def audio_path(value, directory):
    require(isinstance(value, str) and value and not any(ord(c) < 32 for c in value), 'Invalid audio path')
    parsed = urlsplit(value)
    require(not parsed.scheme and not parsed.netloc and not value.startswith('//') and '\\' not in value,
            'Audio must be a local filesystem path, not a URL')
    path = Path(value).expanduser()
    path = (directory / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_file() and path.suffix.lower() == '.wav', f'Missing local WAV: {path}')
    return path


def diff_parts(original, prediction):
    """Lossless lexical diff, retaining whitespace, punctuation, and code switching."""
    left, right = (re.findall(r'\w+|\s+|[^\w\s]', text, flags=re.UNICODE) for text in (original, prediction))
    return [{'tag': tag, 'original': ''.join(left[a:b]), 'prediction': ''.join(right[c:d])}
            for tag, a, b, c, d in SequenceMatcher(None, left, right, autojunk=False).get_opcodes()]


def load_predictions(path, directory):
    data = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(data.get('rows'), list) and data['rows'], 'Empty prediction rows')
    rows = {}
    for row in data['rows']:
        require(isinstance(row, dict), 'Invalid prediction row')
        audio = audio_path(row.get('audio'), directory)
        require(str(audio) not in rows, 'Duplicate prediction audio')
        for key in ('transcript', 'prediction', 'lecture_id'):
            require(isinstance(row.get(key), str), f'Invalid {key}')
        require(bool(row['lecture_id']) and bool(normalized(row['transcript'])), 'Missing lecture or original label')
        require(number(row.get('duration_seconds')) and row['duration_seconds'] > 0, 'Invalid duration')
        require(type(row.get('edits')) is int and row['edits'] >= 0, 'Invalid edit count')
        require(type(row.get('reference_characters')) is int and
                row['reference_characters'] == len(normalized(row['transcript'])), 'Reference count differs from original label')
        require(number(row.get('normalized_cer')) and math.isclose(row['normalized_cer'],
                row['edits'] / row['reference_characters'], rel_tol=1e-10, abs_tol=1e-12), 'Invalid row CER')
        require(type(row.get('truncated')) is bool, 'Missing truncation evidence')
        if 'initial_prediction' in row:
            require(isinstance(row['initial_prediction'], str) and type(row.get('initial_edits')) is int
                    and row['initial_edits'] >= 0, 'Invalid initial prediction evidence')
            require(number(row.get('initial_normalized_cer')) and math.isclose(row['initial_normalized_cer'],
                    row['initial_edits'] / row['reference_characters'], abs_tol=1e-12), 'Invalid initial CER')
            require(all(type(row.get(k)) is bool for k in ('initial_token_limit', 'initial_repetition', 'fallback_used', 'unresolved')),
                    'Invalid decoding flags')
            require(type(row.get('retry_attempts')) is int and row['retry_attempts'] >= 0
                    and isinstance(row.get('decode_evidence'), dict), 'Missing retry evidence')
        require(sha256(audio) == row.get('file_sha256'), 'Audio changed since evaluation')
        rows[str(audio)] = row
    edits, chars = sum(r['edits'] for r in rows.values()), sum(r['reference_characters'] for r in rows.values())
    require(data.get('edits') == edits and data.get('reference_characters') == chars, 'Aggregate counts mismatch')
    require(number(data.get('normalized_cer')) and math.isclose(data['normalized_cer'], edits / chars,
            rel_tol=1e-10, abs_tol=1e-12), 'Aggregate CER mismatch')
    require(data.get('truncated') == sum(r['truncated'] for r in rows.values()), 'Aggregate truncation mismatch')
    has_initial = ['initial_prediction' in r for r in rows.values()]
    require(all(has_initial) or not any(has_initial), 'Incomplete raw initial predictions')
    if all(has_initial):
        initial_cer = sum(r['initial_edits'] for r in rows.values()) / chars
        require(number(data.get('initial_normalized_cer')) and
                math.isclose(data['initial_normalized_cer'], initial_cer, abs_tol=1e-12), 'Aggregate initial CER mismatch')
        for total, field in [('initial_token_limit', 'initial_token_limit'), ('initial_repetition', 'initial_repetition'),
                             ('retry_attempts', 'retry_attempts'), ('fallback_predictions', 'fallback_used'), ('unresolved', 'unresolved')]:
            require(data.get(total) == sum(r[field] for r in rows.values()), f'Aggregate {total} mismatch')
    return data, rows


def payload(directory, output):
    report = json.loads((directory / 'evaluation_report.json').read_text(encoding='utf-8'))
    require(report.get('evaluation_completed') is True and report.get('inputs_unchanged') is True,
            'A completed evaluation with unchanged inputs is required')
    baseline, left = load_predictions(directory / 'baseline_predictions.json', directory)
    selected, right = load_predictions(directory / 'selected_predictions.json', directory)
    require(set(left) == set(right) and report.get('test_examples') == len(left), 'Evaluation clip inventories differ')
    for key, source in [('baseline_normalized_cer', baseline), ('selected_normalized_cer', selected)]:
        require(number(report.get(key)) and math.isclose(report[key], source['normalized_cer'], abs_tol=1e-12),
                'Report disagrees with prediction CER')
    has_initial = 'initial_normalized_cer' in baseline
    require(has_initial == ('initial_normalized_cer' in selected), 'Raw initial predictions are missing for one model')
    totals = {}
    for label, source in [('base', baseline), ('tuned', selected)]:
        prefix = 'baseline' if label == 'base' else 'selected'
        totals[label + '_initial_cer'] = source.get('initial_normalized_cer')
        totals[label + '_decode'] = {key: source.get(key) for key in
                                    ('initial_token_limit', 'initial_repetition', 'retry_attempts', 'fallback_predictions', 'unresolved')}
        for report_key, source_key in [('raw_initial_cer', 'initial_normalized_cer'),
                                       *[(k, k) for k in totals[label + '_decode']]]:
            if prefix + '_' + report_key in report:
                require(report[prefix + '_' + report_key] == source.get(source_key), 'Report disagrees with decoding evidence')
    clips = []
    for audio, original in left.items():
        adapted = right[audio]
        for key in ('transcript', 'lecture_id', 'duration_seconds', 'file_sha256', 'pcm_sha256', 'text_sha256'):
            require(original.get(key) == adapted.get(key), f'Baseline/selected input mismatch: {key}')
        clip = {'clip_id': Path(audio).stem, 'lecture_id': original['lecture_id'],
                      'audio': quote(os.path.relpath(audio, output.parent), safe='/'),
                      'duration_seconds': original['duration_seconds'], 'original': original['transcript'],
                      'base': original['prediction'], 'tuned': adapted['prediction'],
                      'base_cer': original['normalized_cer'], 'tuned_cer': adapted['normalized_cer'],
                      'base_truncated': original['truncated'], 'tuned_truncated': adapted['truncated'],
                      'base_diff': diff_parts(original['transcript'], original['prediction']),
                      'tuned_diff': diff_parts(original['transcript'], adapted['prediction'])}
        for label, source in [('base', original), ('tuned', adapted)]:
            if has_initial:
                clip[label + '_initial'] = source['initial_prediction']
                clip[label + '_initial_cer'] = source['initial_normalized_cer']
                clip[label + '_initial_diff'] = diff_parts(source['transcript'], source['initial_prediction'])
                clip[label + '_decode'] = {key: source[key] for key in ('initial_token_limit', 'initial_repetition',
                                                                      'fallback_used', 'retry_attempts', 'unresolved', 'decode_evidence')}
        clips.append(clip)
    return {'clips': clips, 'base_cer': baseline['normalized_cer'], 'tuned_cer': selected['normalized_cer'],
            **totals, 'has_initial': has_initial,
            'base_truncated': baseline['truncated'], 'tuned_truncated': selected['truncated'],
            'partition': report.get('partition', 'evaluation'), 'reference_characters': baseline['reference_characters']}


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src 'self' file:; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>Qwen · Training comparison</title><style>
:root{font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172a3b;background:#f2f5f7}*{box-sizing:border-box}body{margin:0}header{background:#17374b;color:white;padding:30px max(24px,calc((100vw - 1480px)/2))}h1{margin:0 0 8px;font-size:30px}header p{margin:0;color:#d6e1e9}main{max-width:1480px;margin:auto;padding:24px}.cards,.columns{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}.card,.panel,.column{background:white;border:1px solid #d9e2e8;border-radius:10px;padding:20px}.card strong{display:block;font-size:30px;font-variant-numeric:tabular-nums}.card span,.note,.meta{color:#536878;font-size:13px}.panel{margin:20px 0}.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center}select,button{font:inherit;padding:8px;border:1px solid #bccbd5;border-radius:5px;background:white;color:inherit}select{max-width:100%}button{cursor:pointer}button:disabled{opacity:.4}button:focus-visible,select:focus-visible{outline:3px solid #3b89b7;outline-offset:3px}audio{width:100%;margin-top:16px}h2{margin:0 0 6px;font-size:19px}.words{white-space:pre-wrap;overflow-wrap:anywhere;font-size:16px;line-height:1.9;margin-top:16px}.old{background:#ffead1;color:#773d0b}.new{background:#d9f0e6;color:#155b3b}.warning{color:#995500}.legend{margin:12px 0}.legend span{padding:2px 5px;border-radius:3px}.clip-title{overflow-wrap:anywhere}.column{min-width:0}.note{margin:12px 0}.empty{font-style:italic;color:#677b8a}@media(max-width:850px){.columns{grid-template-columns:1fr}.cards{gap:8px}.card{padding:12px}.card strong{font-size:24px}main{padding:14px}header{padding:22px 16px}h1{font-size:25px}}
</style></head><body><header><h1>Original, base model, fine-tuned model</h1><p>Listen to the same clip and compare both predictions against its original transcript.</p></header>
<main><section id="initial-metrics"><h2>Raw initial decoding</h2><div class="cards"><div class="card"><span>Base · initial corpus CER</span><strong id="base-initial-score"></strong></div><div class="card"><span>Fine-tuned · initial corpus CER</span><strong id="tuned-initial-score"></strong></div><div class="card"><span>Initial fine-tuned minus base</span><strong id="initial-change"></strong></div></div><p id="initial-summary" class="note warning"></p></section>
<h2 id="final-heading">Final predictions</h2><div class="cards"><div class="card"><span>Base · final corpus normalized CER</span><strong id="base-score"></strong></div><div class="card"><span>Fine-tuned · final corpus normalized CER</span><strong id="tuned-score"></strong></div><div class="card"><span>Final fine-tuned minus base · percentage points</span><strong id="change"></strong></div></div>
<p id="summary" class="note"></p><p class="note">CER is character edit distance against these original labels after NFKC, case folding, and removal of punctuation and spacing. Lower is better; it can exceed 100%. It is not general transcription accuracy. Source labels may contain errors. Corpus CER weights reference characters, not clips.</p>
<section class="panel"><div class="controls"><label for="clip">Audio clip </label><select id="clip"></select><button id="previous">Previous</button><button id="next">Next</button><label for="decode-view">Prediction version </label><select id="decode-view"><option value="final">Final output</option><option id="initial-option" value="initial">Raw initial output</option></select><label for="original-against">Original highlights versus </label><select id="original-against"><option value="tuned">Fine-tuned</option><option value="base">Base</option></select></div><p id="clip-info" class="meta clip-title" aria-live="polite"></p><audio id="audio" controls preload="none"></audio><p id="audio-error" class="note"></p></section>
<p class="note legend"><span class="old">Original words removed/replaced</span> · <span class="new">Prediction words added/replaced</span> · Highlights retain word, punctuation, and spacing differences; they are not the CER calculation or word timestamps.</p>
<div class="columns"><section class="column"><h2>Original transcript</h2><div id="original-meta" class="meta"></div><div id="original" class="words"></div></section><section class="column"><h2>Base prediction</h2><div id="base-meta" class="meta"></div><div id="base" class="words"></div><details id="base-evidence-box"><summary>Decode evidence</summary><pre id="base-evidence" class="words"></pre></details></section><section class="column"><h2>Fine-tuned prediction</h2><div id="tuned-meta" class="meta"></div><div id="tuned" class="words"></div><details id="tuned-evidence-box"><summary>Decode evidence</summary><pre id="tuned-evidence" class="words"></pre></details></section></div>
<p class="note">All text is shown in full, including initial repetition failures and unresolved outputs. Bounded retries change inference behavior, not the model weights, and do not use reference labels. Retry audio intervals are not word alignments. This offline page makes no network requests. Audio paths are relative to this page; keep its dataset folders together.</p></main>
<script id="payload" type="application/json">__PAYLOAD_JSON__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('payload').textContent), $=id=>document.getElementById(id);
const percent=n=>(100*n).toFixed(2)+'%', signed=n=>(n>0?'+':'')+n.toFixed(2);
let selected=0;
function textDiff(target,parts,key){const node=$(target);node.replaceChildren();for(const part of parts){if(!part[key])continue;const span=document.createElement('span');span.textContent=part[key];if(part.tag!=='equal')span.className=key==='original'?'old':'new';node.append(span);}if(!node.textContent){const blank=document.createElement('span');blank.className='empty';blank.textContent='No prediction text';node.append(blank);}}
function render(changeAudio=true){const row=data.clips[selected],against=$('original-against').value,suffix=$('decode-view').value==='initial'?'_initial':'',version=suffix?'Raw initial':'Final';$('clip').value=String(selected);$('clip-info').textContent=(selected+1)+' / '+data.clips.length+' · '+row.clip_id+' · '+row.duration_seconds.toFixed(2)+' seconds · '+version;$('original-meta').textContent='Highlights versus '+version.toLowerCase()+' '+(against==='base'?'base':'fine-tuned')+' prediction';textDiff('original',row[against+suffix+'_diff'],'original');for(const key of ['base','tuned']){textDiff(key,row[key+suffix+'_diff'],'prediction');const evidence=row[key+'_decode'];let note=version+' normalized CER '+percent(row[key+suffix+'_cer']);if(evidence){note+=' · initial flags: '+([evidence.initial_token_limit?'token limit':'',evidence.initial_repetition?'repetition':''].filter(Boolean).join(', ')||'none')+' · fallback '+(evidence.fallback_used?'used':'not used')+' · retries '+evidence.retry_attempts+' · final unresolved '+(evidence.unresolved?'yes':'no');$(key+'-evidence').textContent=JSON.stringify(evidence.decode_evidence,null,2);}else if(row[key+'_truncated'])note+=' · Reached generation limit';$(key+'-meta').textContent=note;$(key+'-meta').classList.toggle('warning',evidence?evidence.initial_token_limit||evidence.initial_repetition||evidence.unresolved:row[key+'_truncated']);$(key+'-evidence-box').hidden=!evidence;}for(const [i,clip] of data.clips.entries())$('clip').options[i].textContent=(i+1)+'. '+clip.clip_id+' · '+version+' '+percent(clip['base'+suffix+'_cer'])+' → '+percent(clip['tuned'+suffix+'_cer']);if(changeAudio){$('audio').pause();$('audio').src=row.audio;$('audio-error').textContent='';}$('previous').disabled=selected===0;$('next').disabled=selected===data.clips.length-1;}
for(const [i,row] of data.clips.entries()){const option=document.createElement('option');option.value=String(i);$('clip').append(option);}
$('base-score').textContent=percent(data.base_cer);$('tuned-score').textContent=percent(data.tuned_cer);$('change').textContent=signed(100*(data.tuned_cer-data.base_cer))+' pp';$('summary').textContent=data.clips.length+' clips · '+new Set(data.clips.map(r=>r.lecture_id)).size+' lecture(s) · '+data.partition.replaceAll('_',' ')+' · '+data.reference_characters.toLocaleString()+' reference characters';$('initial-metrics').hidden=!data.has_initial;$('initial-option').disabled=!data.has_initial;
if(data.has_initial){$('final-heading').textContent='Final bounded retry decoding';$('base-initial-score').textContent=percent(data.base_initial_cer);$('tuned-initial-score').textContent=percent(data.tuned_initial_cer);$('initial-change').textContent=signed(100*(data.tuned_initial_cer-data.base_initial_cer))+' pp';$('initial-summary').textContent=['base','tuned'].map(key=>{const d=data[key+'_decode'];return (key==='base'?'Base':'Fine-tuned')+': initial token limits '+d.initial_token_limit+', initial repetition '+d.initial_repetition+', fallback clips '+d.fallback_predictions+', retry attempts '+d.retry_attempts+', final unresolved clips '+d.unresolved;}).join(' · ');}else{$('summary').textContent+=' · generation-limit outputs: '+data.base_truncated+' base / '+data.tuned_truncated+' fine-tuned · raw initial evidence not recorded';}
$('clip').addEventListener('change',()=>{selected=Number($('clip').value);render();});$('previous').addEventListener('click',()=>{if(selected>0){selected--;render();}});$('next').addEventListener('click',()=>{if(selected+1<data.clips.length){selected++;render();}});$('original-against').addEventListener('change',()=>render(false));$('decode-view').addEventListener('change',()=>render(false));$('audio').addEventListener('error',()=>{$('audio-error').textContent='Audio could not be opened. Keep the local WAV files in their original relative locations and open this page from disk.';});render();
</script></body></html>'''


def build(directory, output=None):
    directory = Path(directory).expanduser().resolve()
    output = Path(output).expanduser().resolve() if output else directory / 'comparison.html'
    require(output.suffix.lower() == '.html' and output.parent.is_dir(), 'Output must be an HTML file in an existing directory')
    git = subprocess.run(['git', '-C', str(output.parent), 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
    if git.returncode == 0:
        require(subprocess.run(['git', '-C', git.stdout.strip(), 'check-ignore', '--quiet', str(output)]).returncode == 0,
                'Comparison containing private transcripts must be Git-ignored')
    data = payload(directory, output)
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False)
    for character in ('<', '>', '&', '\u2028', '\u2029'):
        encoded = encoded.replace(character, '\\u%04x' % ord(character))
    page = HTML.replace('__PAYLOAD_JSON__', encoded)
    if output.exists():
        require(output.read_text(encoding='utf-8') == page, 'Output already exists and differs; choose another path')
    else:
        with output.open('x', encoding='utf-8') as handle:
            handle.write(page)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        print(build(args.evaluation_dir, args.output))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f'Comparison refused: {error}\n')


if __name__ == '__main__':
    main()
