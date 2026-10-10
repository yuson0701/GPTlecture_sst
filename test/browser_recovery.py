"""Exercise a failed Qwen clip, continued capture, and IndexedDB recovery after reload."""
import math
import os
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
        for i in range(16000 * 30):
            t = i / 16000
            voiced = any(start <= t < start + .4 for start in [.5, 2, 3.5, 5, 6.5])
            audio.writeframesraw(struct.pack('<h', int(10000 * math.sin(2 * math.pi * 220 * t)) if voiced else 0))
    browser = p.chromium.launch(executable_path=os.environ.get('CHROMIUM_BIN'), args=[
        '--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
        '--use-file-for-fake-audio-capture=' + str(wav)])
    page = browser.new_page(permissions=['microphone'])
    errors, finals = [], []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.route('**/api/config', lambda r: r.fulfill(json={'configured': True, 'backend': 'qwen-mlx', 'model': 'Qwen test'}))
    page.route('**/api/chatgpt/status', lambda r: r.fulfill(json={'status': 'disconnected'}))
    page.route('**/api/session', lambda r: r.fulfill(json={'loaded': True}))
    def transcribe(route):
        body = route.request.post_data_json
        if body['final']:
            finals.append(body)
            if len(finals) == 1:
                route.fulfill(status=503, json={'error': '미세조정 음성 인식이 반복 또는 길이 제한에 도달했습니다.'})
                return
            route.fulfill(json={'text': f'ECG 정상 문장 {len(finals)}입니다.'})
        else:
            route.fulfill(json={'text': '실시간 초안'})
    page.route('**/api/transcribe', transcribe)
    page.goto(sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:3010')
    page.locator('#new-lecture').click()
    page.locator('#title').fill('Qwen recovery regression')
    page.locator('#start').click()
    page.wait_for_timeout(9500)
    assert len(finals) >= 3, 'Later speech must be transcribed after the first failure'
    expect(page.locator('#stop')).to_be_enabled()
    before = page.locator('#timer').inner_text()
    page.wait_for_timeout(1200)
    assert page.locator('#timer').inner_text() != before, 'Timer must keep running'
    page.locator('#stop').click()
    expect(page.locator('#retry')).to_be_visible(timeout=10000)
    page.reload()
    page.locator('.record-card').filter(has_text='Qwen recovery regression').first.click()
    expect(page.locator('#retry')).to_be_visible()
    page.locator('#retry').click()
    expect(page.locator('#status')).to_have_text('처리 완료', timeout=10000)
    assert finals[-1]['audio'] == finals[0]['audio'], 'Retry must read the original failed PCM from IndexedDB'
    page.locator('#script-tab').click()
    expect(page.locator('#transcript')).not_to_contain_text('인식 실패')
    page.reload()
    page.locator('.record-card').filter(has_text='Qwen recovery regression').first.click()
    expect(page.locator('#status')).to_have_text('저장된 강의')
    expect(page.locator('#retry')).to_be_hidden()
    assert not errors, errors
    browser.close()
    print('PASS: failed decode keeps capture/timer running; PCM survives reload; successful retry removes recovery audio')
