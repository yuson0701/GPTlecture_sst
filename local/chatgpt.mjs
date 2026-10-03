import { reportedUsage } from '../public/notes.js';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { createChatGPT, ChatGPTError } from '../vendor/siwc-local/dist/index.js';
import { keychainEncryption } from './credential-encryption.mjs';

export function publicChatGPTError(error) {
  const messages = {
    sign_in_required: 'ChatGPT로 로그인해 주세요.',
    sharing_not_enabled: 'ChatGPT 구독 사용 권한을 허용한 뒤 다시 연결해 주세요.',
    subscription_sharing_user_not_eligible: '이 계정은 ChatGPT 구독 연동을 사용할 수 없습니다. 계정의 이용 자격을 확인해 주세요.',
    subscription_sharing_usage_limit_exceeded: 'ChatGPT 사용 한도에 도달했습니다. 사용량 관리에서 확인한 뒤 다시 시도해 주세요.',
    consent_required: '강의 텍스트 전송을 허용하고 모델을 선택해 주세요.',
    model_not_found: '사용 가능한 ChatGPT 모델을 새로고침하고 다시 선택해 주세요.',
    invalid_paragraph: '완전한 문단 정리를 받지 못했습니다. 원문은 보관되어 있으니 다시 시도해 주세요.',
    macos_required: 'ChatGPT 로그인은 macOS Keychain을 사용합니다. Mac에서 실행해 주세요.',
  };
  // Only the official SDK's sanitized errors may cross the HTTP boundary.
  return { usage: reportedUsage(error?.usage), error: messages[error?.code] || (error instanceof ChatGPTError ? error.message : 'ChatGPT 연결을 완료하지 못했습니다. 연결 상태를 확인하고 다시 시도해 주세요.'), code: error instanceof ChatGPTError ? error.code : 'connection_error' };
}

export function createChatGPTService({ client, platform = process.platform } = {}) {
  const sdk = client || createChatGPT({
    appName: 'Lecture Note', appId: 'lecture-note', redirectPort: 0,
    storageDir: join(homedir(), 'Library', 'Application Support', 'Lecture Note', 'chatgpt'),
    credentialEncryption: keychainEncryption(),
  });
  let login, loginError, cachedModels = [], modelTime = 0;
  const supported = () => { if (!client && platform !== 'darwin') throw new ChatGPTError('macos_required', 'macOS required'); };
  const service = {
    async status() {
      supported();
      const state = await sdk.getSession();
      return { status: login ? 'connecting' : state.status, sharing: state.sharing,
        identity: state.identity, error: loginError || (state.error ? publicChatGPTError(new ChatGPTError(state.error.code, state.error.message)).error : undefined) };
    },
    startSignIn({ reconsent = false } = {}) {
      supported();
      if (login) return;
      loginError = undefined; cachedModels = []; modelTime = 0;
      login = Promise.resolve().then(() => sdk.signIn({ reconsent }))
        .catch(error => { loginError = publicChatGPTError(error).error; })
        .finally(() => { login = undefined; });
    },
    async cancel() { sdk.cancelSignIn(); await login; loginError = undefined; },
    async disconnect() { await service.cancel(); cachedModels = []; modelTime = 0; await sdk.disconnect(); },
    async models({ refresh = false } = {}) {
      supported();
      if (refresh || Date.now() - modelTime > 60000 || !cachedModels.length) {
        cachedModels = await sdk.listModels({ signal: AbortSignal.timeout(20000) }); modelTime = Date.now();
      }
      return cachedModels;
    },
    async respond(options) {
      supported();
      if (!(await service.models()).some(x => x.slug === options.model)) throw new ChatGPTError('model_not_found', 'Unknown model');
      return sdk.streamResponse(options);
    },
    close() { sdk.cancelSignIn(); },
  };
  return service;
}
