"""Real microphone segmentation with a synthetic WAV and delayed mock Qwen."""
import math
import os
import struct
import sys
import tempfile
import wave
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
    wav = Path(tmp) / 'speech-and-pauses.wav'
    # Four short phrases, a long pause, then another phrase and another pause.
    with wave.open(str(wav), 'wb') as audio:
        audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        for i in range(16000 * 40):
            t = i / 16000
            voiced = any(start <= t < start + .4 for start in [0.5, 1.6, 2.7, 3.8]) or 18 <= t < 19
            audio.writeframesraw(struct.pack('<h', int(10000 * math.sin(2 * math.pi * 220 * t)) if voiced else 0))
    browser = p.chromium.launch(executable_path=os.environ.get('CHROMIUM_BIN'), args=[
        '--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
        '--use-file-for-fake-audio-capture=' + str(wav),
    ])
    page = browser.new_page(permissions=['microphone'])
    errors, finals, summaries, held = [], [], [], []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.route('**/api/config', lambda r: r.fulfill(json={'configured': True, 'backend': 'qwen-mlx', 'model': 'Qwen test'}))
    page.route('**/api/chatgpt/status', lambda r: r.fulfill(json={'status': 'connected', 'sharing': True}))
    page.route('**/api/chatgpt/models', lambda r: r.fulfill(json={'models': [{'slug': 'test', 'displayName': 'Test'}]}))
    page.route('**/api/session', lambda r: r.fulfill(json={'loaded': True}))
    def transcribe(route):
        body = route.request.post_data_json
        if body['final']:
            finals.append(body)
            if len(finals) == 1:
                held.append(route)
            else:
                route.fulfill(json={'text': f'확정 문장 {len(finals)}입니다.'})
        else:
            route.fulfill(json={'text': '초안입니다.'})
    def summarize(route):
        summaries.append(route.request.post_data_json)
        route.fulfill(json={'summary': '• 강의 요점입니다.', 'method': 'chatgpt'})
    page.route('**/api/transcribe', transcribe)
    page.route('**/api/summary', summarize)
    page.goto(sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:3010')
    page.locator('#settings-toggle').click()
    page.locator('#chatgpt-model').select_option('test')
    page.locator('#chatgpt-consent').check()
    page.locator('#new-lecture').click()
    page.locator('#start').click()
    page.wait_for_timeout(11500)
    assert held, 'Expected a final transcription to be in flight'
    expect(page.locator('#stop')).to_be_enabled()
    expect(page.locator('#status')).to_contain_text('대기')
    assert not summaries, 'Pause must wait for pre-pause audio to finish transcription'
    held[0].fulfill(json={'text': '확정 문장 1입니다.'})
    expect(page.locator('.summary-block')).to_have_count(1, timeout=4000)
    assert len(finals) >= 3, 'Multiple final chunks must survive the held first request'
    for n in range(1, len(finals) + 1):
        assert f'확정 문장 {n}입니다.' in summaries[0]['transcript']
    page.wait_for_timeout(2500)
    assert len(summaries) == 1, 'Ongoing silence must not create repeated summaries'
    expect(page.locator('#stop')).to_be_enabled()
    # Second speech burst re-arms the five-second silence trigger, well before 75s.
    expect(page.locator('.summary-block')).to_have_count(2, timeout=16000)
    assert not set(summaries[0]['transcript'].splitlines()) & set(summaries[1]['transcript'].splitlines())
    page.locator('#stop').click()
    expect(page.locator('#status')).to_have_text('강의 종료')
    assert len(summaries) == 2
    assert not errors, errors
    browser.close()
    print('PASS: transient Qwen backlog stays recording; pause waits for finals; 5s silence fires once and re-arms')
