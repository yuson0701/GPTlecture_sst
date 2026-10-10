"""Model selection is pinned during capture and survives saved-audio recovery."""
import math
import struct
import sys
import tempfile
import wave
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
    wav = Path(tmp) / 'speech.wav'
    with wave.open(str(wav), 'wb') as audio:
        audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        for i in range(16000 * 12):
            audio.writeframesraw(struct.pack('<h', int(10000 * math.sin(i * 2 * math.pi * 220 / 16000))))
    browser = p.chromium.launch(args=['--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--use-file-for-fake-audio-capture=' + str(wav)])
    page = browser.new_page(permissions=['microphone'])
    errors, calls, sessions = [], [], []
    unavailable = False
    page.on('pageerror', lambda error: errors.append(str(error)))
    def config(route):
        route.fulfill(json={'defaultModel': 'qwen', 'configured': True, 'backend': 'qwen-mlx', 'models': [
            {'id': 'qwen', 'backend': 'qwen-mlx', 'label': 'Qwen 내 모델', 'available': True, 'setup': 'npm run setup:qwen'},
            {'id': 'whisper', 'backend': 'mlx', 'label': 'Whisper GPU', 'available': not unavailable, 'setup': 'npm run setup:whisper'}]})
    page.route('**/api/config', config)
    page.route('**/api/chatgpt/status', lambda r: r.fulfill(json={'status': 'disconnected'}))
    def session(route):
        sessions.append(route.request.post_data_json)
        route.fulfill(json={'loaded': True})
    page.route('**/api/session', session)
    failed = False
    def transcribe(route):
        global failed
        body = route.request.post_data_json
        calls.append(body)
        if body['sttModel'] == 'whisper' and body['final'] and not failed:
            failed = True
            route.fulfill(status=503, json={'error': 'Test recoverable decode failure'})
        else:
            route.fulfill(json={'text': body['sttModel'] + ' ECG 활동전위입니다.'})
    page.route('**/api/transcribe', transcribe)
    page.goto(sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:3010')
    for choice in ['whisper', 'qwen']:
        page.locator('#new-lecture' if choice == 'whisper' else '#new-from-note').click()
        page.locator('#title').fill('Model regression ' + choice)
        page.locator('#stt-model').select_option(choice)
        page.locator('#stt-language').select_option('ko')
        expect(page.locator('#glossary')).not_to_be_visible()
        page.locator('#lecture-options > summary').click()
        page.locator('#glossary').fill('ECG, 활동전위')
        page.locator('#lecture-options > summary').click()
        page.locator('#start').click()
        expect(page.locator('#stt-model')).to_be_disabled()
        expect(page.locator('#stt-language')).to_be_disabled()
        page.wait_for_timeout(1800)
        page.locator('#stop').click()
        expect(page.locator('#new-from-note')).to_be_enabled(timeout=10000)
        assert sessions[-1]['sttModel'] == choice
        assert sessions[-1]['language'] == 'ko'
    assert failed
    assert {c['sttModel'] for c in calls} == {'whisper', 'qwen'}
    assert all(c['language'] == 'ko' and c['glossary'] == 'ECG, 활동전위' for c in calls)
    page.reload()
    page.locator('.record-card').filter(has_text='Model regression whisper').first.click()
    expect(page.locator('#stt-model')).to_have_value('whisper')
    expect(page.locator('#stt-language')).to_have_value('ko')
    expect(page.locator('#stt-model')).to_be_disabled()
    page.locator('#retry').click()
    expect(page.locator('#status')).to_have_text('처리 완료', timeout=10000)
    assert calls[-1]['sttModel'] == 'whisper'
    page.locator('#new-from-note').click()
    page.locator('#settings-toggle').click()
    page.locator('.settings-details > summary').click()
    unavailable = True
    page.locator('#stt-refresh').click()
    page.locator('#stt-model').select_option('whisper')
    expect(page.locator('#start')).to_be_disabled()
    expect(page.locator('#stt-readiness')).to_contain_text('npm run setup:whisper')
    unavailable = False
    page.locator('#stt-refresh').click()
    expect(page.locator('#start')).to_be_enabled()
    page.locator('#settings-toggle').click()
    page.set_viewport_size({'width': 390, 'height': 844})
    expect(page.locator('#start')).to_be_visible()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Model controls must fit mobile'
    assert not errors, errors
    browser.close()
    print('PASS: model/language choice, capture locking, glossary forwarding, saved-model recovery, setup readiness')
