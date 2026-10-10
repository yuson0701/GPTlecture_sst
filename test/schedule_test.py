import io
import sys
import unittest
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from import_schedule import parse_schedule


def workbook(rows=None, *, epoch1904=False, shared=False, duplicate_sheet=False, period_date=False):
    rows = rows or [['날짜', '교시', '과목명', '강의명', '교수님', '학생'],
                    [46200, 2, '심순', '심전도', '교수', 'DO NOT IMPORT'],
                    [46200, 1, '', '심장근', '', 'PRIVATE']]
    strings, sheet = [], []
    for n, row in enumerate(rows, 1):
        cells = []
        for i, value in enumerate(row):
            ref = f'{chr(65+i)}{n}'
            if isinstance(value, (int, float)):
                style = ' s="1"' if period_date and i == 1 else ''
                cells.append(f'<c r="{ref}"{style}><v>{value}</v></c>')
            elif shared:
                strings.append(value)
                cells.append(f'<c r="{ref}" t="s"><v>{len(strings)-1}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>')
        sheet.append(f'<row r="{n}">{"".join(cells)}</row>')
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        extra = '<sheet name="subject duplicate" sheetId="2" r:id="r2"/>' if duplicate_sheet else ''
        z.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><workbookPr date1904="{int(epoch1904)}"/><sheets><sheet name="전체" sheetId="1" r:id="r1"/>{extra}</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr('xl/worksheets/sheet1.xml', f'<worksheet xmlns="{ns}"><sheetData>{"".join(sheet)}</sheetData></worksheet>')
        z.writestr('xl/sharedStrings.xml', f'<sst xmlns="{ns}">' + ''.join(f'<si><t>{escape(s)}</t></si>' for s in strings) + '</sst>')
        if period_date:
            z.writestr('xl/styles.xml', f'<styleSheet xmlns="{ns}"><numFmts><numFmt numFmtId="164" formatCode="m\\-d"/></numFmts><cellXfs><xf numFmtId="0"/><xf numFmtId="164"/></cellXfs></styleSheet>')
    return output.getvalue()


class ScheduleTests(unittest.TestCase):
    def test_only_canonical_fields_and_sheet_with_stable_ids(self):
        result = parse_schedule(workbook(duplicate_sheet=True))
        other = parse_schedule(workbook(shared=True))
        self.assertEqual(result['lectures'], other['lectures'])
        self.assertEqual(len(result['lectures']), 2)
        self.assertEqual(result['lectures'][0]['period'], '1')
        self.assertEqual(result['lectures'][0]['course'], '미분류')
        self.assertEqual(result['missingCourse'], 1)
        self.assertNotIn('PRIVATE', str(result))
        self.assertNotIn('DO NOT IMPORT', str(result))
        self.assertEqual(set(result['lectures'][0]), {'id', 'date', 'period', 'course', 'title', 'professor', 'sourceRow'})

    def test_dates_and_duplicates(self):
        header = ['날짜', '교시', '과목명', '강의명', '교수님']
        row = [1, 1, '해부', '흉부', '']
        result = parse_schedule(workbook([header, row, row]))
        self.assertEqual(result['duplicatesRemoved'], 1)
        self.assertEqual(result['lectures'][0]['date'], '1900-01-01')
        self.assertEqual(parse_schedule(workbook([header, row], epoch1904=True))['lectures'][0]['date'], '1904-01-02')
        row[0] = '2026-10-12'
        self.assertEqual(parse_schedule(workbook([header, row]))['lectures'][0]['date'], '2026-10-12')
        row[0] = '2026-02-30'
        with self.assertRaisesRegex(ValueError, '날짜'):
            parse_schedule(workbook([header, row]))
        row[0] = ''
        with self.assertRaisesRegex(ValueError, '비어'):
            parse_schedule(workbook([header, row]))

    def test_invalid_workbooks(self):
        with self.assertRaises(zipfile.BadZipFile):
            parse_schedule(b'not a workbook')
        with self.assertRaisesRegex(ValueError, '2 MB'):
            parse_schedule(b'x' * (2 * 1024 * 1024 + 1))
        with self.assertRaisesRegex(ValueError, '강의를 찾지'):
            parse_schedule(workbook([['wrong headers']]))

    def test_period_range_preserves_excel_display_instead_of_serial(self):
        rows = [['날짜', '교시', '과목명', '강의명', '교수님'], [46300, 46025, '근골', '강화학습', '']]
        result = parse_schedule(workbook(rows, period_date=True))
        self.assertEqual(result['lectures'][0]['period'], '1-3')


if __name__ == '__main__':
    if '--fixture' in sys.argv:
        sys.stdout.buffer.write(workbook())
    else:
        unittest.main()
