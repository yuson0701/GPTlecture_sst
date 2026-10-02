# Lecture Note — Korean lecture transcription

Local Korean speech recognition and lecture notes, with no API key or cloud speech service.

## Start page and saved lectures

Branch: **feature/lecture-library**, based on the Qwen3-ASR branch. Existing speech-model settings are preserved.

```sh
git fetch origin
git switch feature/lecture-library
npm start
```

Refresh the browser to see **나의 강의 기록**. Click **새 강의 시작**, enter a title, then **녹음 시작**. Recent lectures show their title, last saved time, duration, and a transcript preview. Open any lecture to read its **대화 기록** and **스크립트**, rename it, finish pending summaries, or export Markdown. **새 강의** creates a separate record; archived lectures are not overwritten by a new recording. **샘플 보기** also saves a clearly labeled demo so you can test the library without installing a speech model.

Transcripts (including marked drafts), source paragraphs, AI notes, title, and timing metadata are automatically saved about every two seconds and flushed when stopping or returning to the library. Watch **이 컴퓨터에 저장됨** before closing. Writes use atomic file replacement and revision checks; a failed or conflicting save stays visible with **저장 재시도**, and navigation is blocked until saved. A browser/server crash can lose changes since the last completed save. Interrupted summaries can be retried after reopening; microphone capture itself is not resumed.

Records live in **`data/lectures/` inside the project folder**, one JSON file per lecture. They survive browser and server restarts. The directory is ignored by Git; pushing/switching branches does not upload the records. Back up this directory if moving computers or deleting/recloning the project. Advanced installations can set `LECTURE_DATA_DIR` to an absolute directory. The app stores text notes, **not audio recordings**; unprocessed audio cannot be recovered after closing the tab. Old notes from tabs closed before this feature cannot be recovered.

Persistence QA: `CHROMIUM_BIN=/path/to/chromium python3 test/browser_library.py` runs an isolated server with temporary storage, saves records, restarts the server, reopens the notes, and tests failed-save recovery.

## New branch: Qwen3-ASR for Korean on Apple Silicon

This branch adds **Qwen3-ASR-1.7B (8-bit), running on the Apple GPU through MLX Audio**, as an accuracy-oriented alternative to Korean Zipformer. It also offers **0.6B (8-bit)** for a smaller model. The existing `work` branch is unchanged.

Stop the app with Ctrl+C, then run:

```sh
git fetch origin
git switch feature/qwen3-korean-asr
npm run setup:qwen
npm start
```

Refresh the browser. The label must say **Qwen3-ASR · Apple GPU · 1.7B**. Use native ARM Node.js and Python 3.10+ on your M5; Python 3.12 is recommended. Do not run the terminal under Rosetta. The initial setup downloads packages and model weights, uses a separate `.venv-qwen`, verifies all model files, and loads/warms the GPU **before** changing `.env`. It backs up existing settings to `.env.before-qwen`. No OpenAI API is involved, and speech stays local. GitHub hosts the implementation; Hugging Face hosts the community-converted weights.

For the smaller model:

```sh
npm run setup:qwen -- --size 0.6B
npm start
```

Only one ASR model is active. Old Zipformer/Whisper files remain installed. To restore the prior configuration, stop the app, run `cp .env.before-qwen .env`, and restart. Switching Git branches alone does not restore `.env` because it is ignored by Git.

### Why this candidate

| Candidate | Reason to consider it | Limitation for this app |
| --- | --- | --- |
| Qwen3-ASR-1.7B / MLX Audio | Korean support, vocabulary hints, multilingual benchmark evidence, native Apple GPU execution | Larger autoregressive decoder; M5 latency and lecture accuracy unmeasured |
| Qwen3-ASR-0.6B / MLX Audio | Smaller alternative on the same backend | Upstream accuracy is lower than 1.7B on the cited multilingual sets |
| Whisper large-v3 / faster-whisper | Established multilingual baseline | Its documented GPU path is NVIDIA CUDA; the old CPU setup caused long delays here |
| Existing Korean Zipformer | Lightweight stateful streaming | User-reported Korean recognition quality was inadequate |

Sources reviewed: [Qwen3-ASR repository and evaluation](https://github.com/QwenLM/Qwen3-ASR#evaluation), [MLX Audio Qwen support](https://github.com/Blaizzy/mlx-audio/tree/94c7716212b2228f178d2f9c7619a591fd1b0b78/mlx_audio/stt/models/qwen3_asr), [faster-whisper requirements](https://github.com/SYSTRAN/faster-whisper#requirements). Qwen reports multilingual CommonVoice WER of **9.18 for 1.7B**, **12.75 for 0.6B**, and **10.77 for Whisper large-v3**. Those are aggregated upstream results, include Korean among other languages, and are **not Korean-only scores, Zipformer comparisons, or measurements of these 8-bit MLX conversions**. Qwen's upstream code/model release is Apache-2.0; MLX Audio is MIT. The installer pins the reviewed MLX Audio source revision.

### Latency and honest validation

This implementation uses rolling audio windows, **not persistent Zipformer-style encoder state**. It requests drafts after approximately 0.8 seconds of speech, updates them as new audio arrives, and finalizes on a pause or after about 6 seconds. Old queued drafts are replaced, final audio is retained, and a sustained final-audio backlog stops capture visibly. Silero VAD rejects silent windows. Course vocabulary is sent through Qwen's native hotword prompt. The timing shown is request duration, not total spoken-word latency.

**0–3 seconds is a target, not a verified result.** The development machine is Linux and cannot run the Apple GPU backend. Tests verify integration and failure behavior using mocked inference; real-model speed, Korean accuracy, memory use, and a full-hour thermal test must be checked on the Mac. Earlier Zipformer performance numbers below do not apply to Qwen. Ollama paragraph cleanup is separate from speech recognition and may compete for GPU/memory while recording.

For a reproducible check on your own Korean recording (16 kHz mono PCM16 WAV):

```sh
.venv-qwen/bin/python scripts/benchmark_qwen.py sample.wav --model mlx-community/Qwen3-ASR-1.7B-8bit --reference transcript.txt
```

The UTF-8 reference must contain the actual spoken words. The report includes recognized text, normalized character error rate, and final-window decoding times. It excludes rolling-preview overhead and is not an end-to-end latency benchmark. Compare the output against the same recording in your lecture environment before replacing your usual setup.

## Older Zipformer setup (optional comparison)


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
- Shows each summary block alongside its source; local Ollama paraphrases one block at a time.

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

Enter a lecture title, click **강의 시작**, then **종료** when finished. Stop flushes remaining samples and processes pending results. Use **노트 내보내기** to download Markdown before closing or refreshing. Notes are saved locally in the lecture library; pending audio remains only in tab memory. The scripted **샘플 강의 체험** demo remains available without installed models.

## Block-by-block notes

Every 20 seconds, finalized speech is automatically grouped into a new block. The document-style **대화 기록** view shows a short heading, Korean bullet notes, and a cleaned paragraph. The current speech appears beneath completed paragraphs. **스크립트** keeps the unmodified recognition text; each completed paragraph also has an expandable original-source section. Earlier blocks stay visible. Long backlogs are divided into smaller blocks; **지금 요약** creates the next block immediately, and stopping the lecture processes the remainder. Failed requests retry the same source block. Markdown export preserves every source/paraphrase pair.

Korean paragraph cleanup adjusts spacing, punctuation and clear disfluencies; it is AI editing, not a second audio recognition pass. It cannot reliably fix misheard names or technical terms. The prompt forbids guessing unclear words, but model edits still require comparison with the original. No measured Korean recognition-accuracy improvement is claimed.

Paraphrasing requires local [Ollama](https://ollama.com/). Install and launch Ollama, then download the default model once:

```sh
ollama pull qwen2.5:3b
```

No OpenAI API or other cloud service is used. Block requests use Ollama even if the earlier streaming setup selected extractive summaries. An existing `OLLAMA_MODEL` setting takes precedence; otherwise blocks use `qwen2.5:3b`. If Ollama/model loading fails, that block shows **핵심 문장 추출 · 바꿔쓰기 아님** with a setup message. Extracted sentences are not presented as paraphrases.

Each request contains only its own source block, never a cumulative summary. Paraphrasing runs separately from transcription, with one summary request at a time and a two-thread CPU setting. Local model inference still consumes compute and memory and may affect STT latency on a fanless Mac. Only finalized speech is included; provisional words stay in the transcript until finalized.

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

No system/tab audio capture, speaker diarization, permanent audio storage, or cross-device access is implemented. Microphone interruptions, sleep, and tab closure can interrupt a lecture; check the save indicator before leaving. University recording rules and lecturer consent still apply.
