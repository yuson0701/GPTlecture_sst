"""Browser regression for document following and reading without scroll jumps."""
import sys
from playwright.sync_api import sync_playwright, expect

base = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:3010'
with sync_playwright() as p:
    for engine in (p.chromium, p.webkit):
        browser = engine.launch()
        for width in (1280, 390):
            page = browser.new_page(viewport={'width': width, 'height': 800})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.route('**/api/config', lambda route: route.fulfill(json={'configured': False}))
            page.route('**/api/chatgpt/status', lambda route: route.fulfill(json={'status': 'disconnected'}))
            page.goto(base)
            page.locator('#new-lecture').click()
            assert page.locator('.toolbar').count() == 0
            controls = page.locator('.record-controls')
            expect(controls).to_be_visible()
            assert controls.evaluate("(el) => getComputedStyle(el).position") == 'static'
            size = controls.bounding_box()
            for status in ['강의를 듣고 있어요', '듣는 중 · 14개 대기 · 확정 전사 우선']:
                page.locator('#status').evaluate('(el, text) => el.textContent = text', status)
                box = controls.bounding_box()
                assert abs(box['width'] - size['width']) < 1
                assert abs(box['height'] - size['height']) < 1
            # Exercise the production scroll controller against the actual document
            # layout, without requiring a microphone or model.
            page.evaluate("""async () => {
                const { followLatest } = await import('/follow-scroll.js');
                window.followTest = followLatest(document.querySelector('#workspace'), () => true);
                window.addParagraphs = (selector, count) => {
                    const target = document.querySelector(selector);
                    for (let i = 0; i < count; i++) {
                        const row = document.createElement('p');
                        row.textContent = '한국어 강의와 English technical terms. '.repeat(12);
                        target.append(row);
                    }
                };
            }""")
            def bottom():
                page.wait_for_function('document.documentElement.scrollHeight - innerHeight - scrollY <= 3')
            def add(selector='#summary', count=8):
                page.evaluate('([selector, count]) => addParagraphs(selector, count)', [selector, count])
            for view, selector in [('notes', '#summary'), ('script', '#transcript')]:
                if view == 'script':
                    page.locator('#script-tab').click()
                    page.evaluate('followTest.reset()')
                add(selector, 24)
                bottom()
                add(selector)
                bottom()
                # Wheel up and a same-frame content change must not snap back.
                page.mouse.move(width / 2, 400)
                page.mouse.wheel(0, -600)
                page.wait_for_timeout(200)
                top = page.evaluate('scrollY')
                assert page.evaluate('document.documentElement.scrollHeight - innerHeight - scrollY') > 100
                add(selector)
                page.wait_for_timeout(200)
                assert abs(page.evaluate('scrollY') - top) < 3
                # Returning to the bottom re-arms following.
                page.evaluate('scrollTo(0, document.documentElement.scrollHeight)')
                page.wait_for_timeout(100)
                add(selector)
                bottom()
                # Keyboard upward navigation also pauses before the next frame.
                page.keyboard.press('PageUp')
                page.wait_for_timeout(250)
                top = page.evaluate('scrollY')
                add(selector)
                page.wait_for_timeout(200)
                assert abs(page.evaluate('scrollY') - top) < 3
                page.evaluate('scrollTo(0, document.documentElement.scrollHeight)')
                page.wait_for_timeout(100)
            # Reflow while following should keep the latest content in view.
            page.set_viewport_size({'width': max(320, width - 60), 'height': 700})
            bottom()
            assert not errors, errors
            page.close()
            print(f'PASS: {engine.name} width={width}: stable controls, follows updates, pauses for reading, resumes at bottom')
        browser.close()
