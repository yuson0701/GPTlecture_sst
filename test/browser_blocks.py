"""Block lifecycle QA with mocked ASR/Ollama and real browser capture."""
import os
import sys
from playwright.sync_api import sync_playwright, expect
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get('CHROMIUM_BIN'), args=['--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'])
    page = browser.new_page(permissions=['microphone'])
    errors, calls, held = [], [], []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.route('**/api/config', lambda r: r.fulfill(json={'configured': True, 'streaming': True, 'model': 'Test'}))
    page.route('**/api/session', lambda r: r.fulfill(json={'session': 'test'}))
    def transcribe(r):
        n = r.request.post_data_json['sequence']
        r.fulfill(json={'events': [{'id': str(n), 'seconds': n / 5, 'text': f'원문 번호 {n}의 강의 내용입니다.', 'final': True}]})
    def summary(r):
        calls.append(r.request.post_data_json)
        if len(calls) == 1:
            held.append(r)
        elif len(calls) == 2:
            r.fulfill(status=503, json={'error': 'Retry test'})
        else:
            r.fulfill(json={'summary': '현재 블록의 쉬운 설명입니다.', 'method': 'ollama'})
    page.route('**/api/transcribe', transcribe)
    page.route('**/api/summary', summary)
    page.goto(sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:3010')
    page.locator('#start').click()
    expect(page.locator('#summarize')).to_be_enabled()
    # The first block must be created by the live 20-second timer.
    expect(page.locator('.summary-block[aria-busy="true"]')).to_have_count(1, timeout=25000)
    page.locator('.summary-block details').first.evaluate('(el) => el.open = true')
    source = page.locator('.block-source').inner_text()
    assert source == calls[0]['transcript']
    page.wait_for_timeout(700)
    assert page.locator('.block-source').inner_text() == source
    held[0].fulfill(json={'summary': '첫 블록의 쉬운 설명입니다.', 'method': 'ollama'})
    expect(page.locator('.summary-copy')).to_have_text('첫 블록의 쉬운 설명입니다.')
    page.locator('#summarize').click()
    expect(page.locator('.summary-block')).to_have_count(2)
    expect(page.locator('.block-status').nth(1)).to_contain_text('처리 실패')
    page.locator('#summarize').click()
    expect(page.locator('.summary-copy').nth(1)).to_have_text('현재 블록의 쉬운 설명입니다.')
    assert calls[1] == calls[2], 'Retry must retain exactly the original block'
    assert not set(calls[0]['transcript'].splitlines()) & set(calls[1]['transcript'].splitlines())
    assert all(c['previous'] == '' and c['block'] for c in calls)
    page.locator('#stop').click()
    expect(page.locator('#status')).to_have_text('강의 종료')
    assert page.locator('.block-source').first.inner_text() == source
    with page.expect_download() as download:
        page.locator('#export').click()
    text = open(download.value.path()).read()
    assert source in text and '첫 블록의 쉬운 설명입니다.' in text
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.locator('#script-tab').click()
    expect(page.locator('#transcript')).to_be_visible()
    expect(page.locator('#notes-view')).to_be_hidden()
    page.locator('#notes-tab').click()
    expect(page.locator('#notes-view')).to_be_visible()
    page.set_viewport_size({'width': 1440, 'height': 1000})
    page.screenshot(path='/tmp/lecture-paragraphs.png', full_page=True)
    assert not errors, errors
    print('PASS: automatic block creation, visible source during processing, immutable blocks, isolated requests, retry, stop flush, export, mobile')
    browser.close()
