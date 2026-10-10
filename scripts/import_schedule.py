"""Read a flat university XLSX timetable without executing formulas or macros.

Only date, period, course, lecture title and instructor are imported. The original
workbook and unrelated student assignment columns are never copied to the app.
"""
import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import posixpath
import re
import sys
import unicodedata
import zipfile
import xml.etree.ElementTree as ET

NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
HEADERS = {'날짜': 'date', '교시': 'period', '과목명': 'course', '강의명': 'title', '교수님': 'professor'}
MAX_BYTES = 2 * 1024 * 1024


def clean(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', str(value or ''))).strip()


def parse_date(value, epoch):
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}(?:T.*)?', value):
        return dt.date.fromisoformat(value[:10]).isoformat()
    serial = float(value)
    if not 1 <= serial <= 100000:
        raise ValueError('날짜 범위가 올바르지 않습니다.')
    # Excel's fictitious 1900 leap day shifts serials starting with day 60.
    if epoch == dt.datetime(1899, 12, 30) and serial < 60:
        serial += 1
    return (epoch + dt.timedelta(days=serial)).date().isoformat()


def parse_schedule(payload):
    if not payload or len(payload) > MAX_BYTES:
        raise ValueError('2 MB 이하의 .xlsx 시간표를 선택하세요.')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = archive.infolist()
        if len(infos) > 3000 or sum(i.file_size for i in infos) > 64 * 1024 * 1024:
            raise ValueError('시간표 파일의 압축 해제 크기가 너무 큽니다.')
        def xml(name):
            return ET.fromstring(archive.read(name))
        workbook = xml('xl/workbook.xml')
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(t.text or '' for t in item.findall('s:t', NS) + item.findall('s:r/s:t', NS))
                      for item in xml('xl/sharedStrings.xml').findall('s:si', NS)]
        relations = {r.attrib['Id']: r.attrib['Target'] for r in xml('xl/_rels/workbook.xml.rels')
                     if r.attrib.get('TargetMode') != 'External'}
        sheets = workbook.findall('s:sheets/s:sheet', NS)
        target = next((s for s in sheets if clean(s.attrib['name']) == '전체'), None)
        if target is None:
            if len(sheets) != 1:
                raise ValueError('전체 시트가 필요합니다. 날짜·교시·과목명·강의명·교수님 열을 확인하세요.')
            target = sheets[0]
        name = relations.get(target.attrib[REL], '')
        name = posixpath.normpath(name.lstrip('/') if name.startswith('/') else posixpath.join('xl', name))
        if not name.startswith('xl/worksheets/'):
            raise ValueError('시간표 시트 경로가 올바르지 않습니다.')
        sheet = xml(name)
        props = workbook.find('s:workbookPr', NS)
        epoch = dt.datetime(1904, 1, 1) if props is not None and props.attrib.get('date1904') in ('1', 'true') else dt.datetime(1899, 12, 30)
        formats = []
        if 'xl/styles.xml' in archive.namelist():
            styles = xml('xl/styles.xml')
            custom = {f.attrib['numFmtId']: f.attrib['formatCode'] for f in styles.findall('s:numFmts/s:numFmt', NS)}
            formats = [custom.get(f.attrib.get('numFmtId'), '') for f in styles.findall('s:cellXfs/s:xf', NS)]
        columns, lectures, seen = None, [], set()
        duplicates, missing_course = 0, 0
        for row in sheet.findall('s:sheetData/s:row', NS):
            cells = {}
            for cell in row.findall('s:c', NS):
                col = re.sub(r'\d', '', cell.attrib['r'])
                kind = cell.attrib.get('t')
                value = cell.findtext('s:v', default='', namespaces=NS)
                if kind == 's':
                    value = shared[int(value)]
                elif kind == 'inlineStr':
                    value = ''.join(t.text or '' for t in cell.findall('.//s:t', NS))
                # Some period ranges were saved by Excel as dates. Preserve the
                # workbook's displayed m-d value (e.g. 1-3), not serial 46025.
                style = int(cell.attrib.get('s', '0'))
                if columns and col == columns['period'] and kind in (None, 'n') and value and style < len(formats):
                    if formats[style] in ('m\\-d', 'm-d', 'm"-"d'):
                        displayed = dt.date.fromisoformat(parse_date(value, epoch))
                        value = f'{displayed.month}-{displayed.day}'
                cells[col] = clean(value)
            if columns is None:
                found = {HEADERS[value]: col for col, value in cells.items() if value in HEADERS}
                if len(found) == len(HEADERS):
                    columns = found
                continue
            values = {key: cells.get(col, '') for key, col in columns.items()}
            if not values['title']:
                continue
            if not values['date'] or not values['period']:
                raise ValueError(f'{row.attrib["r"]}행의 날짜 또는 교시가 비어 있습니다. 원본을 확인하세요.')
            try:
                values['date'] = parse_date(values['date'], epoch)
            except (ValueError, OverflowError) as exc:
                raise ValueError(f'{row.attrib["r"]}행의 날짜를 읽지 못했습니다.') from exc
            if any(len(values[k]) > limit for k, limit in {'title': 300, 'course': 80, 'period': 40, 'professor': 120}.items()):
                raise ValueError(f'{row.attrib["r"]}행의 강의 정보가 너무 깁니다.')
            if not values['course']:
                missing_course += 1
                values['course'] = '미분류'
            key = json.dumps(values, sort_keys=True, ensure_ascii=False)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            lectures.append({'id': hashlib.sha256(key.encode()).hexdigest()[:24], **values, 'sourceRow': int(row.attrib['r'])})
            if len(lectures) > 5000:
                raise ValueError('시간표는 최대 5,000개 강의까지 지원합니다.')
        if not lectures:
            raise ValueError('강의를 찾지 못했습니다. 날짜·교시·과목명·강의명·교수님 열을 확인하세요.')
        lectures.sort(key=lambda x: (x['date'], int(re.match(r'\d+', x['period'])[0]) if re.match(r'\d+', x['period']) else 999, x['period'], x['course']))
        return {'schema': 1, 'sourceSha256': hashlib.sha256(payload).hexdigest(), 'sheet': target.attrib['name'],
                'lectures': lectures, 'duplicatesRemoved': duplicates, 'missingCourse': missing_course}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workbook', nargs='?', type=Path)
    parser.add_argument('--stdin', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        payload = sys.stdin.buffer.read(MAX_BYTES + 1) if args.stdin else args.workbook.read_bytes()
        result = parse_schedule(payload)
        result['sourceName'] = args.workbook.name if args.workbook else '시간표.xlsx'
        output = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix('.tmp')
            temporary.write_text(output, encoding='utf-8')
            temporary.replace(args.output)
        else:
            print(output, end='')
    except Exception as exc:
        print(str(exc) if isinstance(exc, ValueError) else '시간표를 읽지 못했습니다. 올바른 .xlsx 파일을 선택하세요.', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
