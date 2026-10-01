# Lecture Note — Korean lectures without an OpenAI API key

A local lecture workspace inspired by Tiro: Korean speech recognition beside continually updated study notes. **No OpenAI API, paid cloud service, API key, or ChatGPT subscription is required.**

- **Speech:** faster-whisper with multilingual Whisper `large-v3-turbo`, running on your computer.
- **AI summaries:** Ollama with local `qwen2.5:7b`, refreshed every 20 seconds when new transcript text is ready.
- **Fallback:** if Ollama is unavailable, a deterministic algorithm selects key sentences from the transcript and prior notes. The UI and exports label this as sentence extraction, not AI explanation.
- **Privacy:** the app sends audio only to its local Node/Python backend, and text only to local Ollama. Once models are downloaded, inference works offline. Notes and pending audio stay in tab memory; export before closing.

Whisper is an open-weight model originally released by OpenAI; using its weights locally does **not** call the OpenAI API. ChatGPT Voice Mode is not used.

## Setup on your computer

Install [Node.js 22+](https://nodejs.org/), Python 3.10–3.12 and, for AI summaries, [Ollama](https://ollama.com/). Run the browser and server on the same computer. Initial package/model downloads require internet and several GB of disk space. No separate FFmpeg executable is needed for the PCM capture path.

### macOS / Linux

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_model.py --model large-v3-turbo
cp .env.example .env
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_model.py --model large-v3-turbo
Copy-Item .env.example .env
```

The server finds `.venv` automatically. Use `PYTHON_BIN` in `.env` if your interpreter is somewhere else.

### Local AI summaries (recommended)

With Ollama running:

```sh
ollama pull qwen2.5:7b
```

On Linux, start `ollama serve` in a separate terminal if the service isn't already running. On macOS/Windows, the Ollama app normally starts the service. The app uses only `http://127.0.0.1:11434` and rejects cloud model tags. Use a downloaded local model, not an Ollama cloud model.

For a lighter summary model, run `ollama pull qwen2.5:3b` and set `OLLAMA_MODEL=qwen2.5:3b` in `.env`. Smaller models can produce weaker Korean summaries.

To skip Ollama entirely, set `SUMMARY_MODE=extractive`. You will still get key sentences, but no generative explanations.

### Start

```sh
npm start
```

Open [http://localhost:3000](http://localhost:3000) in Chrome or Edge. No npm install is needed. Click **샘플 강의 체험** for a scripted demo; it works even without Python, model weights, or Ollama.

For a real lecture:

1. Enter a lecture title and Korean/English course vocabulary.
2. Click **강의 시작**. The model loads before microphone capture starts; allow microphone access when prompted.
3. Transcription appears after approximately 3–10 seconds of audio **plus inference time**. New finalized text is summarized periodically.
4. Click **종료** to stop the microphone, process the remaining audio, and complete the final summary.
5. Click **노트 내보내기** to download Markdown. Closing or refreshing the tab discards notes and pending audio.

## Hardware and model choices

Default: `large-v3-turbo`, CPU, `int8`. This prioritizes a capable multilingual model, not guaranteed real-time speed on every laptop. **Korean lecture accuracy and latency have not been benchmarked here.** Far-away microphones, specialist vocabulary, and simultaneous speakers can reduce accuracy.

| Setting | When to consider it |
| --- | --- |
| `WHISPER_MODEL=large-v3-turbo` | Default Korean-capable model; more compute and memory |
| `WHISPER_MODEL=small` | Lower resource use; potentially worse Korean accuracy |
| `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16` | Supported NVIDIA GPU with compatible CUDA/cuDNN |
| `OLLAMA_MODEL=qwen2.5:3b` | Lighter local summaries than 7b |
| `SUMMARY_MODE=extractive` | No language-model download or inference needed |

After changing the speech model, download the matching weights first, e.g. `.venv/bin/python scripts/download_model.py --model small`, update `.env`, and restart. CPU `int8` is the portable starting point. faster-whisper does not use Apple Metal through this implementation. For GPU installation details, see the [faster-whisper requirements](https://github.com/SYSTRAN/faster-whisper#requirements).

Whisper and Ollama running together compete for resources. A modern computer with 16 GB RAM is a reasonable starting point for the defaults, but actual requirements and throughput depend on hardware, context size and model quantization. If the queue grows, select smaller models or extraction mode.

## Implementation and failure behavior

- AudioWorklet captures mono samples and outputs silence to avoid speaker feedback. Browser code resamples to 16 kHz signed PCM and sends it to the same-origin server.
- Chunks end at a quiet interval after 3 seconds or at a 10-second cap. Forced splits overlap by 400 ms; matching boundary text is removed on a best-effort basis. This is **chunked near-real-time transcription**, not token-by-token streaming. Boundary errors can still occur.
- A persistent Python worker keeps the faster-whisper model in memory. It uses Korean language hints, supplied terminology and VAD. Model loading uses `local_files_only=True`; lecture-time processing never silently downloads weights or falls back to a cloud service.
- A serial browser queue keeps chunks ordered and limits backlog to eight chunks. If full, recording stops and unsaved audio is visibly marked as a missing segment. Processing errors retain the failed chunk and later chunks for **처리 재시도**. Audio is not persisted across refreshes; Markdown export includes a marker for unprocessed segments, not the audio itself.
- Stop flushes the last partial chunk and waits for pending transcription and summary work. Slow inference can make stopping take time. Local recognition requests have a two-minute timeout; timed-out workers are terminated and can be retried.
- Summary batches are bounded and combined with the prior summary. This is lossy: the full transcript remains the source of truth. Fallback extraction may select sentences from earlier generated notes as well as the new transcript; it does not create new explanations.
- Loopback serving, same-origin checks, fixed local Ollama access, payload limits, and single-job admission protect this personal-use app. It is not a public multi-user deployment. There are no speech-provider session time limits, but long lectures remain limited by memory and processing speed.

## Troubleshooting

- **로컬 음성 모델 설치 필요:** install `requirements.txt` into `.venv`; refresh the page. Python must be available on the server computer.
- **Model cannot load:** download the exact configured model with the same user account, check the cache and device settings, then retry. The app never downloads automatically during a lecture.
- **Only key sentences appear:** run Ollama, pull the configured local Qwen model, and verify `SUMMARY_MODE=ollama`. Subsequent summary batches use Ollama if it recovers.
- **Slow transcription / queue fills:** use `small`, disable generative summaries, or configure a supported GPU. Check any visible missing-segment markers.
- **No microphone:** use localhost in a supported browser, grant microphone permission, and keep the tab awake. OS suspension, browser interruptions or microphone disconnects stop recording and preserve queued work.

Recording remains subject to your university's rules and the lecturer's consent. No speaker diarization, system/tab audio capture, persistent lecture history, or audio playback is implemented.

## Validation

```sh
npm run check
npm test
python3 -m unittest discover -s test -p '*_test.py'
```

Validation completed: 14 Node tests and 3 Python tests pass. Chromium smoke checks covered the scripted demo, Markdown export, mobile layout, synthetic-microphone capture, final-chunk flushing, and retrying retained audio after a simulated model failure.

Automated tests cover local-only routing, PCM conversion and segmentation, bounded ordered queues, retry retention, offline worker configuration, Korean hints, and Ollama fallback. Model calls in these tests use fixtures; they do not measure live transcription accuracy. faster-whisper is installed in this workspace, but model weights and Ollama are not installed here. The environment allows package downloads but not the model-hosting destinations. Complete the one-time model setup on your computer before a real lecture.

References: [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Ollama](https://ollama.com/), [Qwen2.5](https://github.com/QwenLM/Qwen2.5).
