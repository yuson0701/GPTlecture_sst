"""Whisper capture -> final text -> ChatGPT notes, including backlog and retry."""
import math
import os
import struct
import sys
import tempfile
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
    wav = Path(tmp) / 'lecture.wav'
    with wave.open(str(wav), 'wb') as audio:
        audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        for i in range(16000 * 60):
            # Two final Whisper windows followed by a long pause.
            audio.writeframesraw(struct.pack('<h', int(10000 * math.sin(i * 2 * math.pi * 220 / 16000)) if i < 16000 * 10 else 0))
    browser = p.chromium.launch(executable_path=os.environ.get('CHROMIUM_BIN'), args=[
        '--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
        '--use-file-for-fake-audio-capture=' + str(wav),
    ])
    page = browser.new_page(permissions=['microphone'])
    errors, finals, summaries, held = [], [], [], []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.route('**/api/config', lambda r: r.fulfill(json={
        'configured': True, 'defaultModel': 'qwen', 'backend': 'qwen-mlx', 'models': [
            {'id': 'qwen', 'backend': 'qwen-mlx', 'label': 'Qwen', 'available': True},
            {'id': 'whisper', 'backend': 'mlx', 'label': 'Whisper', 'available': True},
        ],
    }))
    page.route('**/api/chatgpt/status', lambda r: r.fulfill(json={'status': 'connected', 'sharing': True}))
    page.route('**/api/chatgpt/models', lambda r: r.fulfill(json={'models': [{'slug': 'test', 'displayName': 'Test'}]}))
    page.route('**/api/session', lambda r: r.fulfill(json={'loaded': True}))
    text = 'Troponin C에 칼슘이 결합합니다. L-type calcium channel과 SR의 칼슘 방출이 심근 수축을 조절합니다. 이완 과정에는 칼슘의 재흡수가 필요하며 수축과 이완이 함께 영향을 받습니다. 이러한 과정을 사례를 통해 살펴봅니다.'
    def transcribe(route):
        body = route.request.post_data_json
        assert body['sttModel'] == 'whisper' and body['language'] == 'auto'
        if body['final']:
            finals.append(body)
            if len(finals) == 2:
                held.append(route)
                return
        route.fulfill(json={'text': text if body['final'] else '실시간 초안'})
    def summarize(route):
        summaries.append(route.request.post_data_json)
        if len(summaries) == 1:
            route.fulfill(status=503, json={'code': 'invalid_paragraph', 'error': 'Incomplete note',
                                          'usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120}})
        else:
            route.fulfill(json={'method': 'chatgpt', 'summary': '칼슘이 심근 수축을 조절합니다.',
                               'sections': [{'title': '칼슘과 심근 수축', 'bullets': ['칼슘이 심근 수축을 조절합니다.']}]})
    page.route('**/api/transcribe', transcribe)
    page.route('**/api/summary', summarize)
    page.goto(sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:3010')
    page.locator('#new-lecture').click()
    page.locator('#title').fill('Whisper note regression')
    page.locator('#stt-model').select_option('whisper')
    expect(page.locator('#summary-status')).to_contain_text('ChatGPT 요약 모델')
    page.locator('#settings-toggle').click()
    page.locator('#chatgpt-model').select_option('test')
    page.locator('#chatgpt-consent').check()
    page.locator('#settings-toggle').click()
    page.locator('#start').click()
    page.wait_for_timeout(16500)
    assert len(finals) == 2 and held
    assert not summaries, 'five-second pause waits for its in-flight final'
    expect(page.locator('#stop')).to_be_enabled()
    # Wall-clock deadline moves, but microphone capture and the held request do not.
    now = datetime.now(timezone.utc) + timedelta(seconds=100)
    page.clock.set_fixed_time(now)
    expect(page.locator('.summary-block')).to_have_count(1, timeout=4000)
    expect(page.locator('.block-warning')).to_contain_text('한 번 다시 시도')
    expect(page.locator('#chatgpt-consent')).to_be_checked()
    assert len(summaries) == 1
    assert summaries[0]['transcript'].count('Troponin C') == 1, 'only the finished prefix is summarized'
    page.wait_for_timeout(1500)
    assert len(summaries) == 1, 'no rapid retry on queue or timer updates'
    page.clock.set_fixed_time(now + timedelta(seconds=16))
    expect(page.locator('.summary-copy h3')).to_have_text('칼슘과 심근 수축', timeout=4000)
    assert len(summaries) == 2 and summaries[0] == summaries[1], 'retry uses identical source'
    held.pop().fulfill(json={'text': '두 번째 구간입니다. ' + text})
    expect(page.locator('.summary-block')).to_have_count(2, timeout=5000)
    expect(page.locator('.summary-copy h3')).to_have_count(2)
    assert '두 번째 구간입니다.' in summaries[2]['transcript']
    assert not set(summaries[0]['transcript'].splitlines()) & set(summaries[2]['transcript'].splitlines())
    page.locator('#stop').click()
    expect(page.locator('#status')).to_have_text('강의 종료', timeout=10000)
    assert len(summaries) == 3, 'stop does not duplicate completed notes'
    page.reload()
    page.locator('.record-card').filter(has_text='Whisper note regression').first.click()
    expect(page.locator('#stt-model')).to_have_value('whisper')
    expect(page.locator('.summary-copy h3')).to_have_count(2)
    assert not errors, errors
    browser.close()
    print('PASS: Whisper notes, pause/backlog deadline, one delayed retry, consent, source integrity, persistence')
