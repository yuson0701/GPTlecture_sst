# Lecture Note — Korean lecture transcription

Local Korean speech recognition and lecture notes, with no API key or cloud speech service.

## Upgrade for low-latency lectures (recommended for your M5 MacBook Air)

Stop the running app with **Ctrl+C**, then run these commands in the project folder:

```sh
git pull --ff-only origin work
npm run setup:realtime
npm start
```

Open or refresh [http://localhost:3000](http://localhost:3000). The model label must say **Korean Zipformer · 실시간 스트리밍**. If it still says `faster-whisper · large-v3-turbo`, you are using the old CPU batch model or an old server is still running.

`setup:realtime` installs sherpa-onnx, downloads the official Korean streaming model once, and updates `.env` to select it. It preserves other settings and makes an `.env.before-realtime` backup. Existing Whisper weights are not deleted. The new runtime needs about 135 MB of model files; the initial release archive download is larger because it contains alternative weights. This model runs efficiently on CPU and does **not** need MLX, CUDA, a GPU setup, or an HF token.

The **0–3 second speech-to-text target must be checked on your actual lecture audio and hardware**. This version uses a real streaming model; it does not claim a guaranteed deadline for every word. Poor microphones, unfamiliar vocabulary, background noise, CPU contention and recognition context can still delay or distort words.

## Why this changes the delay

The previous code repeatedly decoded overlapping audio windows with a large Whisper model. On a CPU, extra preview requests could increase work rather than produce faster text.

The recommended mode now uses [sherpa-onnx's Korean streaming Zipformer](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html):

- Captures **200 ms of new audio per packet**, with no overlap and no repeated window decoding.
- Keeps encoder state between packets and displays partial results as soon as the model returns them.
- Finalizes text after a pause and periodically during continuous speech. Silence can still be processed, but it adds no blank transcript rows.
- Keeps at most one request in flight; packets have sequence numbers and retries cannot decode the same audio twice.
- Uses two CPU threads. Consumed feature frames are discarded; endpoint resets retain unread lookahead rather than dropping audio at sentence boundaries.
- Updates only changed transcript rows. Earlier lecture text stays in memory for scrolling and export without rebuilding the entire page on every word.
- Uses lightweight sentence extraction during live recording so Ollama does not compete with recognition by default.

The timing display reports the **latest packet's inference time and capture-to-response delay**. It is a diagnostic, not a measurement of when each spoken word became recognizable: model lookahead and linguistic context add latency. If ten packets (about two seconds of audio) accumulate, recording stops visibly and drains already-received audio instead of quietly slipping 10–20 seconds behind. A hard queue overflow is explicitly marked; audio is never silently skipped to make the display look faster.

The streaming model supports **Korean only**. English terms, mixed-language lectures and punctuation may be weaker than Whisper. Course vocabulary hints are disabled for this backend; text is not silently corrected using the glossary. Test accuracy on a short sample from your course before relying on it for a full lecture.

## First-time installation

Install Node.js 22+ and Python 3.10–3.12. Clone the `work` branch, then:

```sh
git clone -b work https://github.com/yuson0701/GPTlecture_sst.git
cd GPTlecture_sst
npm run setup:realtime
npm start
```

Use Chrome or Edge on the same computer as the server and allow microphone access. On Windows PowerShell, use `npm.cmd` if script execution policy blocks `npm`. Keep the terminal and browser tab open; prevent the computer from sleeping during the lecture.

Enter a lecture title, click **강의 시작**, then **종료** when finished. Stop flushes remaining samples and processes pending results. Use **노트 내보내기** to download Markdown before closing or refreshing. There is no cloud history: notes and any pending audio are in tab memory. The scripted **샘플 강의 체험** demo remains available without installed models.

## Summaries

The low-latency setup sets:

```dotenv
LECTURE_SUMMARY_MODE=extractive
```

During recording, this selects existing key sentences from the lecture and prior notes. The UI labels it **핵심 문장 추출 · AI 설명 아님**. It does not invent explanations. This is a deliberate speed/summary-quality tradeoff for live lectures.

If you prefer generative Korean explanations while recording, install and run [Ollama](https://ollama.com/), run `ollama pull qwen2.5:3b`, and set:

```dotenv
LECTURE_SUMMARY_MODE=ollama
OLLAMA_MODEL=qwen2.5:3b
SUMMARY_MODE=ollama
```

This consumes more memory and compute and can increase recognition delay. No external API is used: requests go only to local Ollama. If Ollama is unavailable, notes fall back to labeled sentence extraction. Summaries update every 20 seconds when new finalized text is available; provisional text is not summarized.

## Existing Whisper modes

They remain available for comparison or mixed-language accuracy needs. To switch back, remove `STT_BACKEND=sherpa` from `.env` or set `STT_BACKEND=faster-whisper` / `STT_BACKEND=mlx`. `STT_BACKEND` takes precedence over the older `WHISPER_BACKEND` variable. These modes still re-decode growing windows and are **not the recommended 0–3 second lecture configuration**.

CPU/NVIDIA setup:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_model.py --model large-v3-turbo
```

Apple Silicon MLX setup:

```sh
.venv/bin/python -m pip install -r requirements-mac.txt
.venv/bin/python scripts/download_model.py --backend mlx
```

Then set `STT_BACKEND=mlx` and `MLX_WHISPER_MODEL=mlx-community/whisper-turbo`. Native ARM Python is required for MLX. MLX uses separate model files from faster-whisper. These are local open-weight models, not ChatGPT Voice Mode or the OpenAI API.

## Verification and performance

On the Linux test machine, real Korean audio through Chromium's microphone pipeline produced the first visible text about **1.92 seconds after listening started**. This measures the beginning of one sample, not every word or M5 performance. A separate 60-second decoder run returned its first partial result after 0.8 seconds of input, with 63.4 ms p95 processing time per 200 ms packet.

The one-hour audio continuity run completed in **368.94 seconds** (real-time factor 0.1025), with 578 finalized segments, 58.9 ms p95 packet processing, and 239.4 MB peak resident memory. Peak memory did not grow beyond its post-load high-water mark. This repeated one Korean sample on Linux at accelerated speed; it does not establish lecture accuracy or one-hour thermal performance on an M5.

Run normal tests:

```sh
npm run check
npm test
.venv/bin/python -m unittest discover -s test -p '*_test.py'
```

Measure **your own computer** using real Korean sample audio included in the streaming model release:

```sh
.venv/bin/python scripts/benchmark_realtime.py --seconds 60
```

A `realTimeFactor` below 1 means decoding is faster than incoming audio; below about 0.5 provides useful headroom. It is a throughput measure, not a word-latency guarantee. `firstResult.audioSeconds` tells you how much sample audio had arrived before the first nonempty recognition result.

For long-session processing and memory continuity:

```sh
.venv/bin/python scripts/benchmark_realtime.py --seconds 3600
```

This feeds one hour of audio (repeating the included Korean recording) as fast as possible through the real model. It verifies long-stream processing, not one hour of wall-clock thermal behavior on a fanless Mac. Live browser/microphone latency and Korean accuracy still need evaluation on your actual lecture.

Optional browser checks require Playwright and Chromium. `test/browser_smoke.py` covers the original Whisper-style UI using mocked responses. With the streaming server running, `CHROMIUM_BIN=/path/to/chromium python3 test/browser_streaming.py http://127.0.0.1:3000` exercises the real Korean model through browser microphone capture, finalization, export, and mobile layout. Streaming unit tests additionally cover sequence validation, idempotent retries, decoder failure recovery, and one-hour capture continuity.

## Privacy and limitations

The server binds only to localhost, rejects cross-origin API calls, and processes speech locally. Initial package/model downloads require internet; lecture-time inference does not download weights. The model is from the official [sherpa-onnx Korean streaming release](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-korean-2024-06-16.tar.bz2). Training provenance is linked in the [upstream model documentation](https://github.com/k2-fsa/sherpa/blob/master/docs/source/onnx/pretrained_models/online-transducer/zipformer-transducer-models.rst).

No system/tab audio capture, speaker diarization, permanent audio storage, or cross-device access is implemented. Microphone interruptions, sleep, and tab closure can interrupt a lecture; export notes before leaving. University recording rules and lecturer consent still apply.
