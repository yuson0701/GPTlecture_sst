"""Real on-disk persistence and server restart; demo content needs no speech model."""
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.request
from playwright.sync_api import sync_playwright, expect
ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as storage, sync_playwright() as p:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    env = {**os.environ, 'PORT': str(port), 'LECTURE_DATA_DIR': storage}
    def start():
        process = subprocess.Popen(['node', 'server.mjs'], cwd=ROOT, env=env, stdout=subprocess.DEVNULL)
        for _ in range(100):
            try:
                urllib.request.urlopen(base + '/api/records', timeout=1).close()
                return process
            except OSError:
                time.sleep(.1)
        process.terminate(); raise RuntimeError('Server did not start')
    server = start()
    browser = p.chromium.launch(executable_path=os.environ.get('CHROMIUM_BIN'), args=['--no-sandbox'])
    try:
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('dialog', lambda d: d.accept())
        page.goto(base)
        expect(page.locator('#library')).to_be_visible()
        expect(page.locator('#library-message')).to_contain_text('아직 저장된')
        page.locator('#demo').click()
        expect(page.locator('#status')).to_have_text('샘플 강의 종료', timeout=15000)
        expect(page.locator('#save-status')).to_have_text('이 컴퓨터에 저장됨')
        page.locator('#title').fill('첫 번째 경제학 강의')
        page.locator('#home').click()
        expect(page.locator('.record-card')).to_have_count(1)
        expect(page.locator('.record-card')).to_contain_text('첫 번째 경제학 강의')
        server.terminate(); server.wait(timeout=10); server = start()
        page.reload()
        expect(page.locator('.record-card')).to_have_count(1)
        page.locator('.record-card').click()
        expect(page.locator('.summary-block')).to_have_count(4)
        page.locator('#script-tab').click()
        expect(page.locator('.segment')).to_have_count(4)
        expect(page.locator('#start')).to_be_disabled()
        # A failed save must block navigation and preserve the on-screen text.
        page.route('**/api/records/*', lambda r: r.fulfill(status=503, json={'error': 'Simulated disk failure'}) if r.request.method == 'POST' else r.continue_())
        page.locator('#title').fill('저장 실패 후 복구한 제목')
        page.locator('#home').click()
        expect(page.locator('#save-retry')).to_be_visible()
        expect(page.locator('#workspace')).to_be_visible()
        page.unroute('**/api/records/*')
        page.locator('#save-retry').click()
        expect(page.locator('#save-status')).to_have_text('이 컴퓨터에 저장됨')
        page.locator('#home').click()
        expect(page.locator('.record-card')).to_contain_text('저장 실패 후 복구한 제목')
        page.locator('#new-lecture').click()
        expect(page.locator('#title')).to_have_value('')
        expect(page.locator('.segment')).to_have_count(0)
        page.locator('#demo').click()
        expect(page.locator('#status')).to_have_text('샘플 강의 종료', timeout=15000)
        page.locator('#home').click()
        expect(page.locator('.record-card')).to_have_count(2)
        page.screenshot(path='/tmp/lecture-library.png')
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        assert not errors, errors
        print('PASS: empty home, auto-save, server restart, transcripts + blocks restored, failed-save recovery, distinct lectures, mobile')
    finally:
        browser.close(); server.terminate(); server.wait(timeout=10)
