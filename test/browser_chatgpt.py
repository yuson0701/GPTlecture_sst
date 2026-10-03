"""Account UI regression with synthetic auth; does not verify live OAuth/Keychain."""
import os
import sys
from playwright.sync_api import sync_playwright, expect

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get('CHROMIUM_BIN'), args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    state = {'status': 'disconnected', 'sharing': False}
    calls, errors = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.route('**/api/config', lambda r: r.fulfill(json={'configured': True, 'model': 'Qwen3-ASR · Apple GPU'}))
    def auth(route):
        action = route.request.url.rsplit('/', 1)[-1]
        if action == 'status':
            route.fulfill(json=state)
        elif action == 'models':
            route.fulfill(json={'models': [{'slug': 'model-a', 'displayName': 'Model A'}, {'slug': 'model-b', 'displayName': 'Model B'}]})
        else:
            calls.append((action, route.request.post_data_json))
            state.update(status='connecting' if action == 'sign-in' else 'disconnected', sharing=False)
            route.fulfill(json=state)
    page.route('**/api/chatgpt/*', auth)
    page.goto(sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:3010')
    page.locator('#settings-toggle').click()
    expect(page.locator('#chatgpt-login')).to_be_visible()
    expect(page.locator('#chatgpt-consent')).to_be_disabled()
    page.locator('#chatgpt-login').click()
    expect(page.locator('#chatgpt-cancel')).to_be_visible()
    page.locator('#chatgpt-cancel').click()
    expect(page.locator('#chatgpt-login')).to_be_visible()
    page.locator('#chatgpt-login').click()
    state.update(status='connected', sharing=True, identity={'email': 'synthetic@example.com'})
    expect(page.locator('#chatgpt-account')).to_contain_text('synthetic@example.com')
    page.locator('#chatgpt-model').select_option('model-a')
    page.locator('#chatgpt-consent').check()
    page.locator('#chatgpt-model').select_option('model-b')
    expect(page.locator('#chatgpt-consent')).not_to_be_checked()
    page.locator('#chatgpt-consent').check()
    page.reload()
    page.locator('#settings-toggle').click()
    expect(page.locator('#chatgpt-consent')).not_to_be_checked()
    expect(page.locator('#chatgpt-account')).to_contain_text('synthetic@example.com')
    page.screenshot(path='/tmp/chatgpt-lecture-home.png')
    page.locator('#chatgpt-disconnect').click()
    expect(page.locator('#chatgpt-login')).to_be_visible()
    expect(page.locator('#chatgpt-consent')).to_be_disabled()
    assert [x[0] for x in calls] == ['sign-in', 'cancel', 'sign-in', 'disconnect'], calls
    assert not any(x[1].get('reconsent') for x in calls)
    assert not errors, errors
    browser.close()
    print('PASS: sign-in/cancel, account display, model selection, consent reset on model/reload, disconnect')
