import { existsSync } from 'node:fs';
import { join } from 'node:path';
export function runtimePaths({ packaged, resources, root, userData }) {
  const pythonHome = packaged ? join(resources, 'python') : null;
  const python = packaged ? join(pythonHome, 'bin', 'python3.14') : join(root, '.venv-qwen', 'bin', 'python');
  return { python, pythonHome, config: join(userData, 'model.json'), records: join(userData, 'lectures') };
}
export function runtimeEnv(paths, model = {}, inherited = process.env) {
  return { ...inherited, STT_BACKEND: 'qwen-mlx', PYTHON_BIN: paths.python,
    ...(paths.pythonHome ? { PYTHONHOME: paths.pythonHome } : {}),
    PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1',
    LECTURE_DATA_DIR: paths.records, OLLAMA_MODEL: inherited.OLLAMA_MODEL || 'qwen2.5:3b',
    QWEN_ASR_MODEL: model.base || 'mlx-community/Qwen3-ASR-1.7B-8bit',
    QWEN_ASR_ADAPTER: model.adapter || 'yuson0701/qwen3-asr-1.7b-korean-lecture-lora-mlx' };
}
export function usableModel(config) {
  return config?.schema === 1 && typeof config.base === 'string' && typeof config.adapter === 'string'
    && existsSync(join(config.base, 'config.json')) && existsSync(join(config.adapter, 'selected.safetensors'));
}
export function trustedPage(url, appOrigin, setupUrl) {
  try { return url === setupUrl || (Boolean(appOrigin) && new URL(url).origin === appOrigin); } catch { return false; }
}
