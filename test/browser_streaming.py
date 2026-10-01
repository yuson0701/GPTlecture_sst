"""Real Korean streaming model + browser audio pipeline; no ASR mocks.
Run STT_BACKEND=sherpa npm start first, then this script with the localhost URL.
"""
import json
import os
from pathlib import Path
import sys
import time
from playwright.sync_api import sync_playwright, expect

base = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:3000'
root = Path(__file__).resolve().parents[1]
audio = root / 'models/sherpa-onnx-streaming-zipformer-korean-2024-06-16/test_wavs/0.wav'
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get('CHROMIUM_BIN'), headless=True, args=[
        '--no-sandbox', '--disable-dev-shm-usage', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
        f'--use-file-for-fake-audio-capture={audio}%noloop',
    ])
    page = browser.new_page(viewport={'width': 1440, 'height': 1000}, permissions=['microphone'])
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(base, wait_until='networkidle')
    expect(page.locator('#model')).to_contain_text('Zipformer')
    page.locator('#start').click()
    expect(page.locator('#stop')).to_be_enabled(timeout=30000)
    started = time.monotonic()
    expect(page.locator('.segment').first).to_be_visible(timeout=10000)
    first_display = time.monotonic() - started
    first_text = page.locator('.segment').first.inner_text()
    page.wait_for_timeout(2500)
    assert not page.locator('#error').is_visible(), page.locator('#error').inner_text()
    diagnostic = page.locator('#latency').inner_text()
    page.locator('#stop').click()
    expect(page.locator('#status')).to_have_text('강의 종료', timeout=15000)
    assert page.locator('.segment').count() > 0
    assert page.locator('.segment.pending').count() == 0
    with page.expect_download() as result:
        page.locator('#export').click()
    assert result.value.suggested_filename == 'lecture-notes.md'
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert not errors, errors
    print(json.dumps({'firstTextAfterListeningSeconds': round(first_display, 2), 'firstText': first_text,
                      'diagnostic': diagnostic, 'finalizedRows': page.locator('.segment').count(), 'pageErrors': errors}, ensure_ascii=False))
    browser.close()
