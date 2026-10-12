"""Automatic timetable selection, guarded against user edits and saved records."""
import os
import sys
import uuid
from datetime import datetime, timezone
from playwright.sync_api import sync_playwright, expect

rows = [
    {'id': '1' * 24, 'date': '2026-10-12', 'period': '1', 'course': '심순', 'title': '심장근', 'professor': ''},
    {'id': '2' * 24, 'date': '2026-10-12', 'period': '2', 'course': '심순', 'title': '심전도', 'professor': ''},
    {'id': '3' * 24, 'date': '2026-10-12', 'period': '3', 'course': '호흡', 'title': '천식', 'professor': ''},
]
base = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:3010'
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get('CHROMIUM_BIN'))
    page = browser.new_page()
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.clock.set_fixed_time(datetime(2026, 10, 12, 0, 25, tzinfo=timezone.utc))
    page.route('**/api/config', lambda r: r.fulfill(json={'configured': False}))
    page.route('**/api/chatgpt/status', lambda r: r.fulfill(json={'status': 'disconnected'}))
    page.route('**/api/schedule', lambda r: r.fulfill(json={'lectures': rows}))
    page.goto(base)
    page.locator('#new-lecture').click()
    expect(page.locator('#title')).to_have_value('심순 · 심전도')
    assert 'ECG' in page.locator('#glossary').input_value()
    expect(page.locator('#stt-language')).to_have_count(0)
    expect(page.locator('#schedule-file')).to_have_count(0)
    page.locator('#lecture-options > summary').click()
    page.locator('#title').fill('직접 입력한 강의')
    page.locator('#glossary').fill('my own terms')
    page.clock.set_fixed_time(datetime(2026, 10, 12, 1, 40, tzinfo=timezone.utc))
    page.wait_for_timeout(100)
    expect(page.locator('#title')).to_have_value('직접 입력한 강의')
    expect(page.locator('#glossary')).to_have_value('my own terms')
    page.locator('#new-from-note').click()
    expect(page.locator('#title')).to_have_value('호흡 · 천식')
    assert 'asthma' in page.locator('#glossary').input_value()
    page.clock.set_fixed_time(datetime(2026, 10, 13, 1, 40, tzinfo=timezone.utc))
    page.locator('#new-from-note').click()
    expect(page.locator('#title')).to_have_value('호흡 · 천식')
    assert 'asthma' in page.locator('#glossary').input_value()
    expect(page.locator('.usage-strip')).to_have_count(0)
    expect(page.locator('#summary-status')).to_have_text('요약하려면 설정에서 ChatGPT에 연결하세요.')
    expect(page.locator('#summary-status')).not_to_contain_text('1분 15초마다')
    # Opening a saved record must not replace its lecture or edited terminology.
    saved_id = str(uuid.uuid4())
    saved = {'revision': 0, 'title': 'Saved first period', 'glossary': 'saved custom terms',
             'elapsed': 60, 'state': 'idle', 'transcript': [], 'blocks': [], 'summarized': [],
             'scheduleLecture': rows[0]}
    response = page.request.post(base + '/api/records/' + saved_id, data=saved, headers={'Origin': base})
    assert response.ok
    page.clock.set_fixed_time(datetime(2026, 10, 12, 1, 40, tzinfo=timezone.utc))
    page.locator('#home').click()
    page.locator('.record-card').filter(has_text='Saved first period').first.click()
    expect(page.locator('#title')).to_have_value('Saved first period')
    expect(page.locator('#glossary')).to_have_value('saved custom terms')
    expect(page.locator('#schedule-active')).to_contain_text('심장근')
    # Late network response must not overwrite a title entered while loading.
    page.unroute('**/api/schedule')
    pending = []
    page.route('**/api/schedule', lambda r: pending.append(r))
    page.clock.set_fixed_time(datetime(2026, 10, 12, 0, 25, tzinfo=timezone.utc))
    page.goto(base, wait_until='domcontentloaded')
    page.locator('#new-lecture').click()
    page.locator('#title').fill('수동 제목 유지')
    assert pending
    pending.pop().fulfill(json={'lectures': rows})
    expect(page.locator('#schedule-source')).to_contain_text('3개 일정')
    expect(page.locator('#title')).to_have_value('수동 제목 유지')
    expect(page.locator('#glossary')).to_have_value('')
    assert not errors, errors
    browser.close()
    print('PASS: nearest Korean class and vocabulary, new-lecture clock refresh, no language/import menus, no-class handling, delayed-response edit protection')
