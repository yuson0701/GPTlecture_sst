"""Optional browser QA: real audio pipeline, synthetic audio, mocked local models.
Run a local server, then: python test/browser_smoke.py http://127.0.0.1:3010
Requires Playwright and Chromium; set CHROMIUM_BIN if using a system browser.
"""
import array
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import wave
from playwright.sync_api import sync_playwright, expect

base = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:3000'
with tempfile.TemporaryDirectory() as temp, sync_playwright() as p:
    wav_path = Path(temp) / 'speech.wav'
    samples = array.array('h', (int(9000 * math.sin(2 * math.pi * 440 * n / 48000)) if 3 <= n / 48000 < 6 else 0 for n in range(48000 * 12)))
    if sys.byteorder != 'little':
        samples.byteswap()
    with wave.open(str(wav_path), 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(48000); wav.writeframes(samples.tobytes())
    browser = p.chromium.launch(executable_path=os.environ.get('CHROMIUM_BIN'), headless=True, args=[
        '--no-sandbox', '--disable-dev-shm-usage', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
        f'--use-file-for-fake-audio-capture={wav_path}%noloop',
    ])
    page = browser.new_page(viewport={'width': 1440, 'height': 1000}, permissions=['microphone'])
    errors, requests, summaries = [], [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('dialog', lambda dialog: dialog.accept())
    page.route('**/api/config', lambda route: route.fulfill(json={'configured': True, 'model': 'Test fixture'}))
    page.route('**/api/session', lambda route: route.fulfill(json={'loaded': True}))
    fail_final = False
    def transcribe(route):
        global fail_final
        request = route.request.post_data_json
        requests.append(request)
        if request['final'] and fail_final:
            fail_final = False
            route.fulfill(status=503, json={'error': 'Simulated final recognition failure'})
        else:
            route.fulfill(json={'text': '경제학 수업입니다.' if request['final'] else '경제학 수업'})
    def summarize(route):
        summaries.append(route.request.post_data_json)
        route.fulfill(json={'summary': '• 경제학 수업입니다.', 'method': 'extractive'})
    page.route('**/api/transcribe', transcribe)
    page.route('**/api/summary', summarize)
    page.goto(base, wait_until='networkidle')
    page.locator('#start').click()
    expect(page.locator('#stop')).to_be_enabled()
    page.wait_for_timeout(1200)
    assert not requests, 'Silence must not call the recognizer'
    expect(page.locator('.segment.pending')).to_have_count(1, timeout=8000)
    assert not any(request['final'] for request in requests), 'Text should appear before speech ends'
    assert not summaries, 'Drafts must not be summarized'
    page.locator('.segment').evaluate('(row) => { window.originalTranscriptRow = row; }')
    expect(page.locator('.segment:not(.pending)')).to_have_count(1, timeout=8000)
    assert page.locator('.segment').evaluate('(row) => row === window.originalTranscriptRow'), 'Final should update the same row'
    expect(page.locator('.segment p')).to_have_text('경제학 수업입니다.')
    count = len(requests)
    page.wait_for_timeout(1500)
    assert len(requests) == count, 'Silence after finalization must leave the transcript alone'
    page.locator('#stop').click()
    expect(page.locator('#status')).to_have_text('강의 종료', timeout=10000)
    assert len(summaries) == 1 and '경제학 수업입니다.' in summaries[0]['transcript']
    with page.expect_download() as result:
        page.locator('#export').click()
    assert result.value.suggested_filename == 'lecture-notes.md'
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.set_viewport_size({'width': 1440, 'height': 1000})
    # Ensure final audio survives failure even when provisional text already exists.
    fail_final = True
    start_index = len(requests)
    page.locator('#start').click()
    expect(page.locator('#retry')).to_be_visible(timeout=15000)
    expect(page.locator('.segment.pending')).to_have_count(1)
    failed_audio = [r['audio'] for r in requests[start_index:] if r['final']][-1]
    page.locator('#retry').click()
    expect(page.locator('#status')).to_have_text('처리 완료', timeout=10000)
    assert requests[-1]['audio'] == failed_audio
    expect(page.locator('.segment:not(.pending)')).to_have_count(1)
    assert not errors, errors
    print(json.dumps({'partial_before_final': True, 'stable_row': True, 'silent_requests': 0, 'final_audio_retry': True, 'export': True, 'mobile_overflow': False, 'page_errors': errors}))
    browser.close()
