import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { LocalTranscriber } from './transcriber.mjs';

export const MODEL_IDS = ['qwen', 'whisper', 'sherpa'];
const invalid = () => Object.assign(new Error('지원하지 않는 음성 모델입니다.'), { status: 400 });

// Public IDs never become executable paths or arbitrary model downloads.
export class ModelTranscriber {
  constructor(env = process.env, { platform = process.platform, factory = env => new LocalTranscriber(env) } = {}) {
    this.env = env; this.factory = factory; this.active = null; this.busy = false;
    const old = env.STT_BACKEND || env.WHISPER_BACKEND || 'faster-whisper';
    this.defaultModel = old === 'qwen-mlx' ? 'qwen' : old === 'sherpa' ? 'sherpa' : 'whisper';
    const whisper = env.WHISPER_CHOICE_BACKEND || (['mlx', 'faster-whisper'].includes(old) ? old : platform === 'darwin' ? 'mlx' : 'faster-whisper');
    if (!['mlx', 'faster-whisper'].includes(whisper)) throw new Error('Invalid WHISPER_CHOICE_BACKEND');
    const python = (id, legacy) => {
      const override = env[id === 'qwen' ? 'QWEN_PYTHON_BIN' : 'WHISPER_PYTHON_BIN'];
      const local = fileURLToPath(new URL(`../.venv-${id}/bin/python`, import.meta.url));
      return override || (legacy ? env.PYTHON_BIN : undefined) || (existsSync(local) ? local : undefined);
    };
    this.models = [
      { id: 'qwen', backend: 'qwen-mlx', label: env.QWEN_ASR_ADAPTER === 'none' ? 'Qwen3-ASR · 기본 모델' : 'Qwen3-ASR · 내 강의 모델', setup: 'npm run setup:qwen', python: python('qwen', old === 'qwen-mlx') },
      { id: 'whisper', backend: whisper, label: `Whisper · ${whisper === 'mlx' ? 'Apple GPU' : 'faster-whisper'}`, setup: 'npm run setup:whisper', python: python('whisper', ['mlx', 'faster-whisper'].includes(old)) },
    ];
    if (old === 'sherpa') this.models.push({ id: 'sherpa', backend: 'sherpa', label: 'Korean Zipformer · 기존 설정', setup: 'npm run setup:realtime', python: env.PYTHON_BIN });
  }
  selection(id = this.defaultModel) { const model = this.models.find(x => x.id === id); if (!model) throw invalid(); return model; }
  worker(model) { return this.factory({ ...this.env, STT_BACKEND: model.backend, PYTHON_BIN: model.python || '' }); }
  async catalog() {
    const results = [];
    for (const model of this.models) {
      // Readiness probes never load weights or disturb the active decoder.
      const probe = this.worker(model);
      let status;
      try { status = await probe.request('status'); }
      catch { status = { available: false, reason: 'dependencies' }; }
      finally { await probe.shutdown(); }
      const { python, ...publicModel } = model;
      results.push({ ...publicModel, ...status, id: model.id, backend: model.backend });
    }
    return results;
  }
  async request(action, data = {}) {
    const model = this.selection(data.sttModel);
    if (this.busy) throw new Error('음성 처리 중입니다. 다른 작업이 끝난 뒤 다시 시도하세요.');
    this.busy = true;
    try {
      if (this.active?.id !== model.id) {
        // Wait for actual process exit so two GPU models never overlap in RAM.
        if (this.active) await this.active.worker.shutdown();
        this.active = { id: model.id, worker: this.worker(model) };
      }
      return await this.active.worker.request(action, data);
    } catch (error) {
      if (this.active && (action === 'load' || this.active.worker.child === null)) {
        await this.active.worker.shutdown(); this.active = null;
      }
      throw error;
    } finally { this.busy = false; }
  }
  close() { this.active?.worker.close(); this.active = null; }
}
