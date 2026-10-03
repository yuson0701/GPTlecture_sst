import { createCipheriv, createDecipheriv, randomBytes } from 'node:crypto';

// The SDK locks its credential store across processes before encrypt/decrypt.
// The key is stored in macOS Keychain, never in the repository or a .env file.
export function keychainEncryption({ platform = process.platform, loadEntry = async () => {
  const { AsyncEntry } = await import('@napi-rs/keyring');
  return new AsyncEntry('lecture-note.chatgpt', 'credential-encryption-v1');
} } = {}) {
  let entry;
  async function key(create) {
    if (platform !== 'darwin') throw Error('ChatGPT 로그인 저장은 macOS Keychain이 필요합니다.');
    entry ??= await loadEntry();
    let secret = await entry.getSecret();
    if (secret == null && create) {
      secret = randomBytes(32);
      await entry.setSecret(secret);
      secret = await entry.getSecret();
    }
    if (!secret || secret.length !== 32) throw Error('Keychain 암호화 키를 읽지 못했습니다.');
    return Buffer.from(secret);
  }
  return {
    id: 'lecture-note.macos-keychain-aes256gcm.v1',
    isAvailable: () => platform === 'darwin',
    async encrypt(plaintext) {
      const secret = await key(true), nonce = randomBytes(12);
      const cipher = createCipheriv('aes-256-gcm', secret, nonce);
      cipher.setAAD(Buffer.from('lecture-note.chatgpt.v1'));
      const data = Buffer.concat([cipher.update(plaintext, 'utf8'), cipher.final()]);
      return Buffer.concat([nonce, cipher.getAuthTag(), data]);
    },
    async decrypt(ciphertext) {
      const secret = await key(false), data = Buffer.from(ciphertext);
      if (data.length < 28) throw Error('Invalid encrypted credentials');
      const cipher = createDecipheriv('aes-256-gcm', secret, data.subarray(0, 12));
      cipher.setAAD(Buffer.from('lecture-note.chatgpt.v1'));
      cipher.setAuthTag(data.subarray(12, 28));
      return Buffer.concat([cipher.update(data.subarray(28)), cipher.final()]).toString('utf8');
    },
  };
}
