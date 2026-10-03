"""Download explicitly listed Tiro shares into a private local timestamp cache.

The source list contains {"sources": [{"lecture_id": "...", "url": "..."}]}.
Only the supplied public share pages are fetched; no account/API discovery occurs.
HTML is parsed as data, never executed. Existing cached pages are reused.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, build_opener


def share_url(url):
    parsed = urlparse(url)
    if (parsed.scheme != 'https' or parsed.netloc != 'tiro.ooo'
            or not re.fullmatch(r'/s/[A-Za-z0-9]+', parsed.path)
            or parsed.query or parsed.fragment):
        raise ValueError('Expected an explicit https://tiro.ooo/s/... share URL')
    return url


class SameShareRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        share_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class PageData(HTMLParser):
    def __init__(self):
        super().__init__()
        self.capture = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == 'script' and dict(attrs).get('id') == '__NEXT_DATA__':
            self.capture = True

    def handle_endtag(self, tag):
        if tag == 'script':
            self.capture = False

    def handle_data(self, data):
        if self.capture:
            self.parts.append(data)


def extract_page(data, source):
    parser = PageData()
    parser.feed(data.decode('utf-8'))
    props = json.loads(''.join(parser.parts))['props']['pageProps']
    if props.get('needPassword') or not props.get('sharedNote'):
        raise ValueError('Share requires access or contains no shared transcript')
    note = props['sharedNote']
    if not isinstance(note.get('paragraphs'), list) or not note['paragraphs']:
        raise ValueError('Share contains no timestamped paragraphs')
    # Account details and unrelated page configuration stay out of the parsed cache.
    keep = {key: note.get(key) for key in
            ('title', 'sourceType', 'totalRecordingDurationInMillis', 'paragraphs')}
    return {**source, 'page_sha256': hashlib.sha256(data).hexdigest(), 'source': keep}


def fetch_one(source, output):
    ident = source['lecture_id']
    if not re.fullmatch(r'lecture_[a-f0-9]{12}', ident):
        raise ValueError('Invalid lecture ID')
    url = share_url(source['url'])
    raw = output / f'{ident}.html'
    if raw.exists():
        data = raw.read_bytes()
    else:
        with build_opener(SameShareRedirect()).open(url, timeout=45) as response:
            data = response.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024:
            raise ValueError('Unexpectedly large share page')
        raw.write_bytes(data)
    result = extract_page(data, source)
    destination = output / f'{ident}.json'
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != result:
            raise ValueError('Parsed cache differs; use a new output directory')
    else:
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'lecture_id': ident, 'paragraphs': len(result['source']['paragraphs']),
            'page_sha256': result['page_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text(encoding='utf-8'))['sources']
    if not sources or len({row['lecture_id'] for row in sources}) != len(sources):
        parser.error('Source list must contain unique lecture IDs')
    # Use the same Git privacy check as the preparation tools before downloading.
    import subprocess
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = subprocess.run(['git', '-C', str(output), 'rev-parse', '--show-toplevel'],
                          capture_output=True, text=True)
    if repo.returncode == 0:
        check = subprocess.run(['git', '-C', repo.stdout.strip(), 'check-ignore', '--quiet', str(output / 'source.json')])
        if check.returncode:
            parser.error('Share-page output inside Git must be ignored')
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(lambda source: fetch_one(source, output), sources):
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
