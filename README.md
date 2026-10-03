# Lecture Note — Korean lecture transcription

Local Korean speech recognition and lecture notes, with no API key or cloud speech service.

## Start page and saved lectures

Branch: **feature/chatgpt-paraphrases**, based on the saved-lecture branch. Existing speech-model settings are preserved.

```sh
git fetch origin
git switch feature/chatgpt-paraphrases
npm ci
npm start
```

Refresh the browser to see **나의 강의 기록**. Click **새 강의 시작**, enter a title, then **녹음 시작**. Recent lectures show their title, last saved time, duration, and a transcript preview. Open any lecture to read its **대화 기록** and **스크립트**, rename it, finish pending summaries, or export Markdown. **새 강의** creates a separate record; archived lectures are not overwritten by a new recording. **샘플 보기** also saves a clearly labeled demo so you can test the library without installing a speech model.

Transcripts (including marked drafts), source paragraphs, AI notes, title, and timing metadata are automatically saved about every two seconds and flushed when stopping or returning to the library. Watch **이 컴퓨터에 저장됨** before closing. Writes use atomic file replacement and revision checks; a failed or conflicting save stays visible with **저장 재시도**, and navigation is blocked until saved. A browser/server crash can lose changes since the last completed save. Interrupted summaries can be retried after reopening; microphone capture itself is not resumed.

Records live in **`data/lectures/` inside the project folder**, one JSON file per lecture. They survive browser and server restarts. The directory is ignored by Git; pushing/switching branches does not upload the records. Back up this directory if moving computers or deleting/recloning the project. Advanced installations can set `LECTURE_DATA_DIR` to an absolute directory. The app stores text notes, **not audio recordings**; unprocessed audio cannot be recovered after closing the tab. Old notes from tabs closed before this feature cannot be recovered.

Persistence QA: `CHROMIUM_BIN=/path/to/chromium python3 test/browser_library.py` runs an isolated server with temporary storage, saves records, restarts the server, reopens the notes, and tests failed-save recovery.

## Fine-tuned Qwen3-ASR for Korean on Apple Silicon

The default Qwen setup uses **your Korean lecture LoRA adapter on Qwen3-ASR-1.7B (8-bit)**, running locally on the Apple GPU through MLX Audio. The adapter is hosted at [yuson0701/qwen3-asr-1.7b-korean-lecture-lora-mlx](https://huggingface.co/yuson0701/qwen3-asr-1.7b-korean-lecture-lora-mlx), a public Hugging Face model repository. The installer downloads the exact base revision and adapter revision together; the runtime validates both and never silently falls back to the base model.

Stop the app with Ctrl+C, then run:

```sh
git fetch origin
git switch feature/chatgpt-paraphrases
npm run setup:qwen
npm start
```

The base model and adapter are public; no Hugging Face account, login, or token is required. Initial setup needs internet access to download dependencies and weights. Speech recognition runs offline after setup.

Refresh the browser. The label must say **Qwen3-ASR · Apple GPU · 1.7B · 강의 미세조정**. Use native ARM Node.js and Python on Apple Silicon, not Rosetta. The tested environment uses Python 3.14; runtime requirements pin MLX 0.32.3, MLX Audio's tested source revision, Transformers 5.18.0, and Hugging Face Hub 1.33.0. Setup uses a separate `.venv-qwen`, verifies the adapter/base hashes, and loads/warms the GPU **before** changing `.env`. It preserves other settings and backs up the original configuration to `.env.before-qwen`.

Setup writes `QWEN_ASR_MODEL` and `QWEN_ASR_ADAPTER` as local snapshot paths. Merely setting `QWEN_ASR_MODEL` to the adapter repository is insufficient: the adapter is not a standalone model. The server loads the base plus adapter once per worker. Model files stay in the Hugging Face cache, and datasets remain Git-ignored.

For an explicit base-only comparison with the smaller original model:

```sh
npm run setup:qwen -- --base-only --size 0.6B
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

This implementation uses rolling audio windows, **not persistent Zipformer-style encoder state**. It requests drafts after approximately 0.8 seconds of speech, updates them as new audio arrives, and finalizes on a pause or after about 6 seconds. Old queued drafts are replaced, final audio is retained, and a sustained final-audio backlog stops capture visibly. Silero VAD rejects silent windows. Course vocabulary is sent through Qwen's native hotword prompt, including retries. Fine-tuned inference uses the evaluated bounded retry policy for repetition/token limits and reports an error if recovery fails. The timing shown is request duration, not total spoken-word latency.

On one reserved lecture, the adapter reduced normalized character error rate from **43.18% to 21.02%** with the same fixed retry policy (raw fine-tuned CER: **30.20%**). These scores use source paragraph clips and supplied transcripts, not the app's shorter rolling windows; they do not establish live lecture accuracy or latency. **0–3 seconds remains a target.** A full-hour thermal test has not been completed. Earlier Zipformer performance numbers below do not apply to Qwen. ChatGPT paragraph cleanup is separate from speech recognition and requires internet; this branch does not load a local summary model.

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
- Shows each summary block alongside its source; ChatGPT paraphrases one source block at a time after you enable text sharing.

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

Every **75 seconds (1 minute 15 seconds)** from recording start or the last summary request, finalized speech is grouped into a new block. The **요약** view is a single reading document with bold topic headings and explanatory Korean bullet sentences, matching the supplied reference layout. Each block can contain 1–5 topics; the model is instructed to preserve facts and avoid repetitive headings. The current speech appears beneath completed paragraphs. **스크립트** keeps the unmodified recognition text; each completed paragraph also has an expandable original-source section. Earlier blocks stay visible. Long backlogs are divided into smaller blocks; **지금 정리** creates the next block immediately, and stopping the lecture processes the remainder. Failed requests retry the same source block. Markdown export preserves every source/paraphrase pair.

Korean paraphrasing adjusts spacing, punctuation and clear disfluencies; it is AI editing, not a second audio recognition pass. Original and previously saved cleaned text stay inside expandable source details, rather than repeating the entire transcript under every heading. It cannot reliably fix misheard names or technical terms. The prompt forbids guessing unclear words, but model edits still require comparison with the original. No measured Korean recognition-accuracy improvement is claimed.

### Token usage for each lecture

The note header displays reported input, output and total tokens, plus the number of summary requests. Counts come from ChatGPT response usage fields, not a character-count estimate. Attempts are saved with their source blocks and restored with the lecture; retries add separate entries, including reported usage when a completed response is not valid summary JSON. Unreported, interrupted and legacy requests are marked **사용량 미확인** and excluded from the known total. They are not assumed to be free. This is the saved lecture's known usage, not your account-wide subscription balance or cost; **계정 사용량** opens the official usage page. A browser crash before autosave can also leave an attempt's usage unknown.

The 75-second cadence affects paraphrasing only; STT draft timing is unchanged. Manual **지금 정리** and ending the lecture can flush a shorter remainder. Very large blocks retain the 6,400-character input bound and are split into additional requests.

### ChatGPT subscription sign-in (no Ollama)

This branch uses the [official OpenAI Sign in with ChatGPT DevKit](https://github.com/openai/sign-in-with-chatgpt-devkit), with browser authorization, subscription permission, model discovery and text Responses. It does not automate or read the ChatGPT desktop app, use its cookies, or require an OpenAI API key. Your existing fine-tuned Qwen STT settings and saved lectures are preserved. Old `OLLAMA_MODEL`, `SUMMARY_MODE` and `LECTURE_SUMMARY_MODE` settings are ignored; no requests are made to Ollama.

On your Mac, run `npm ci` once after switching branches, then `npm start` and open http://localhost:3000. Node.js 22+ is required. Installation builds the pinned official SDK sources. Click **설정** to open the ChatGPT panel:

1. Click **Sign in with ChatGPT** and complete the browser authorization. macOS may ask for Keychain access.
2. Allow subscription usage if your account is eligible. If permission is missing, click **구독 사용 허용 / 다시 연결** to explicitly request it again.
3. Choose a model from your account's returned catalog. The app does not guess which models your subscription supports.
4. Check **강의 텍스트 전송 · 자동 정리 허용**. This choice resets when you reload, change models, disconnect, or a request fails.
5. Start a lecture or open a saved record and click **지금 정리**. **사용량 관리** opens ChatGPT's usage controls; **연결 해제** removes this app's saved credentials and attempts remote revocation.

Each request sends only its current finalized transcript block and Korean editing instructions. Microphone audio, other lectures, prior blocks and the glossary are not sent. The SDK requests `store: false`; this is not a claim of zero retention under the service's policies. Summarizing needs internet and consumes eligible ChatGPT plan usage. Subscription eligibility, region, model access and usage limits are enforced by OpenAI; a paid plan alone does not establish access. Requests may be refused until the account/app is enabled.

One request runs at a time, separate from STT. On errors, incomplete/malformed results, or exhausted usage, no replacement summary is fabricated: the original block remains saved, automatic requests pause, and local transcription continues. Re-enable text sharing and click **지금 정리** to retry that same block. Disconnecting cancels in-flight SDK requests. Unchecking sharing stops new requests; already submitted requests can finish. No automatic retries consume extra plan usage after an error.

Credentials are encrypted with AES-256-GCM using a key stored through macOS Keychain; encrypted SDK state lives in `~/Library/Application Support/Lecture Note/chatgpt`, outside the repository. Tokens are never sent to browser JavaScript or saved in lecture records. This branch's credential provider supports macOS; other systems can still view records/demo and use their existing local STT. Loss/denial of Keychain access fails closed; it does not overwrite unreadable credentials. This is a local browser app; do not expose its localhost server through a public proxy.

The DevKit is vendored from revision `f723814abdccec135b519c451fb6e1992ee5e933`, with a documented token-usage extension and its [noncommercial license](vendor/siwc-local/LICENSE). The license permits personal noncommercial use/development; commercial distribution requires separate permission. This branch does not include the deferred Electron packaging work.

Validation uses synthetic credentials and mocked ChatGPT responses. Live OAuth, your account's eligibility and macOS Keychain prompts still need an on-device check; Linux CI cannot verify them. Browser block QA: `CHROMIUM_BIN=/path/to/chromium python3 test/browser_blocks.py http://localhost:3000`.

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

## Preparing a private Qwen3-ASR dataset

The dataset tools operate locally and never launch training. Keep raw exports in
the sibling `../voiceinput_output/` directory and generated material in ignored
`data/qwen3_asr/`. Only reusable code and synthetic tests belong in Git.

Install the extra Korean alignment dependency in the existing Qwen environment:

```sh
.venv-qwen/bin/python -m pip install -r requirements-qwen-dataset.txt
```

Preparation pairs `.m4a` with `스크립트.txt` using Unicode-normalized names;
`문서.txt` and `요약.txt` are excluded. It removes export metadata, speaker labels,
and known non-speech events while preserving the spoken wording and English
terms. Original files, hashes, removal logs, and cleaned paragraph offsets are
retained. Whole lectures are assigned to train/validation/test in the private
`source_config.json` before any clipping. Repeated lecturer titles must share a
split, but titles are not verified speaker identities.

```sh
.venv-qwen/bin/python scripts/prepare_qwen_dataset.py \
  --split-config data/qwen3_asr/source_config.json
.venv-qwen/bin/python scripts/align_qwen_dataset.py --stage asr
.venv-qwen/bin/python scripts/align_qwen_dataset.py --stage align \
  --aligner /absolute/path/to/local/Qwen3-ForcedAligner-0.6B-8bit/snapshot
.venv-qwen/bin/python scripts/build_qwen_review.py
```

The aligner snapshot can be downloaded once from
[`mlx-community/Qwen3-ForcedAligner-0.6B-8bit`](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit)
into the ignored dataset model cache. The acoustic audit uses the cached
Qwen3-ASR model independently of the exported labels, then forced-aligns matching
reference spans. It rejects poor text agreement, truncated ASR output, invalid
timestamps, partial words, implausible durations, overlaps, and duplicate audio.
ASR and alignment caches are resumable and checked against source provenance.
Fixed windows are used to find reference text; final clips use aligned word
boundaries. No timing is inferred from paragraph lengths or export timestamps.

`alignment_report.json` records counts and limitations. `segments.jsonl` retains
per-clip evidence; `quarantine.jsonl` records exclusions. `candidates/*.jsonl`
uses the [official Qwen fine-tuning format](https://github.com/QwenLM/Qwen3-ASR/blob/main/finetuning/README.md):

```json
{"audio":"/absolute/path/clip.wav","text":"language Korean<asr_text>전사 내용"}
```

`boundary_review/*.jsonl` is a separate review queue, not an alignment pass.
It retains clips with positive boundary timestamps but sparse isolated internal
zero-duration words, high ASR agreement, and no other alignment failures. Their
warnings remain attached; no words are removed to make a clip pass. The review
page labels this queue explicitly, and export requires human listening approval.

Candidates are machine-checked labels, not verified ground truth. Open the local
`data/qwen3_asr/review.html`, listen to the clips, check verbatim text and word
boundaries, and export review decisions. Approvals are tied to audio/text hashes;
changed labels require a fresh alignment. Publish only approved examples with:

```sh
.venv-qwen/bin/python scripts/export_reviewed_qwen_dataset.py \
  --review-file /absolute/path/to/review-decisions.jsonl
```

High ASR disagreement may reflect recognition errors on medical terminology,
not an incorrect reference. Quarantine preserves these examples for correction
and alignment review; do not interpret rejection rates as transcript accuracy.
Preserve original exports when correcting labels, and prepare corrected copies
into a new ignored output directory with the same lecture split configuration.

This produces `reviewed/train.jsonl`, `reviewed/validation.jsonl`, and
`reviewed/test.jsonl` only after consistency checks pass. Before training,
establish baseline CER on the reviewed validation set. Use validation with the
official trainer's `--eval_file`; reserve test for the final comparison and keep
it out of training, glossary tuning, and model selection. Small held-out lecture
sets do not establish general performance across speakers or courses.

Run the preparation checks with:

```sh
.venv-qwen/bin/python -m unittest discover -s test -p '*qwen*dataset*test.py'
.venv-qwen/bin/python -m unittest discover -s test -p 'qwen_alignment_checks_test.py'
```

## Original Tiro paragraph timestamps

When the original Tiro share contains paragraph start and end times, use those
times directly with the original exported labels. Run whole-lecture preparation
above first to freeze the train/validation/test assignments. Create the private
`data/tiro_timestamps/sources.json` with a `sources` list containing each frozen
`lecture_id` and its explicit Tiro share `url`, then run:

```sh
.venv-qwen/bin/python scripts/fetch_tiro_timestamps.py \
  --sources data/tiro_timestamps/sources.json \
  --output data/tiro_timestamps/raw
.venv-qwen/bin/python scripts/prepare_tiro_dataset.py \
  --base-dataset data/qwen3_asr --output data/tiro_timestamps \
  --max-duration 90
```

Fetching reads only the listed share pages and reuses cached pages. Preparation
runs locally without model inference. It checks cached page hashes and parsed
timestamps, original source hashes, share titles, and the frozen lecture split.
The complete cleaned share transcript must match the complete local original
script, allowing only Unicode composition and whitespace differences. Labels
are never replaced by Qwen predictions, and ASR agreement does not select clips.

Each accepted WAV is an exact PCM slice of its normalized lecture, using the
paragraph's own integer millisecond start and end times. Gaps remain gaps. Missing,
overlapping, unordered, or out-of-bounds times, paragraphs longer than the limit,
empty speech labels, unusable audio, and exact duplicate audio are withheld whole
in `quarantine.jsonl`. Times are never interpolated or clamped. Stale note-level
duration metadata is reported separately; valid paragraph bounds must still fit
the waveform.

`segments.jsonl` records all source intervals, labels, hashes, acoustic checks,
and exclusions. `report.json` records coverage and provenance. The resulting
`train.jsonl`, `validation.jsonl`, and `test.jsonl` use the Qwen format shown above.
Use validation for model selection and reserve test for the final evaluation.
These source timestamps are not a claim of new manual word-level verification;
source transcription errors may remain. New clips retain `human_review: pending`.
Keep pages, source URLs, clips, and reports under ignored `data/`, and adapters
under ignored `checkpoints/`. Reruns verify existing artifacts and refuse changed
inputs or output.

```sh
.venv-qwen/bin/python -m unittest discover -s test -p 'prepare_tiro_dataset_test.py'
```

## Local Qwen LoRA training on Apple Silicon

`scripts/train_qwen_lora.py` adapts the last eight decoder layers of a local
8-bit Qwen3-ASR snapshot. The audio encoder and base weights stay frozen; only
rank-eight query/value adapters are optimized. It uses the original labels,
including the first transcript token and the end-of-transcript token, and does
not apply a loss to the audio prompt. This implementation is checked against
MLX 0.32.3, MLX Audio 0.5.7, and Transformers 5.18.0 in `.venv-qwen`.

```sh
.venv-qwen/bin/python scripts/train_qwen_lora.py \
  --model /absolute/path/to/local/Qwen3-ASR-1.7B-8bit/snapshot \
  --train-file data/tiro_timestamps/train.jsonl \
  --validation-file data/tiro_timestamps/validation.jsonl \
  --output checkpoints/qwen3_asr_lora \
  --epochs 3 --learning-rate 0.00005
```

Use a fresh ignored output directory for each run. The trainer verifies lecture,
file, and PCM separation between training and validation, and never discovers a
test manifest. It rejects clips above 90 seconds or labels above the token limit
instead of truncating them. Frozen intermediate features are cached privately
on disk to limit memory use. No data is uploaded and no weights are downloaded.

The run records input audio/text hashes, training history, baseline and adapted
validation transcripts, and a `training_report.json`. Selection uses validation
token loss, with the unchanged baseline also eligible. `selected.safetensors`
contains the chosen adapter; `best_trained.safetensors` preserves the best trained
checkpoint even if the baseline wins. Compare validation CER as well as loss
before adopting the adapter. Test remains reserved for a final evaluation.

Reload the saved adapter for local recognition with:

```sh
.venv-qwen/bin/python scripts/train_qwen_lora.py \
  --model /absolute/path/to/local/Qwen3-ASR-1.7B-8bit/snapshot \
  --output checkpoints/qwen3_asr_lora \
  --infer-audio /absolute/path/to/mono-16000hz-pcm16.wav
```

Inference accepts clips up to 90 seconds and returns JSON with the transcript
and decoding evidence. Successful full-clip predictions remain unchanged. If a
prediction reaches its token limit or repeats a lexical phrase ten times in a
row, the decoder retries complete, nonoverlapping audio sections of at most
15 seconds, with bounded further bisection for failed sections. It never uses a
reference transcript for these retries. Unresolved outputs remain flagged.

Training does not change the running application's model configuration. Adapter
reload verifies the base weights and saved adapter hashes. Training completion
also checks that the base weights, input manifests, and input WAVs are unchanged.

Once the checkpoint is fixed, evaluate the reserved test lecture separately:

```sh
.venv-qwen/bin/python scripts/evaluate_qwen_lora.py \
  --model /absolute/path/to/local/Qwen3-ASR-1.7B-8bit/snapshot \
  --run checkpoints/qwen3_asr_lora \
  --test-file data/tiro_timestamps/test.jsonl \
  --output data/qwen3_asr_lora_test
```

This checks test lecture/audio separation against the run's saved provenance,
then compares the base model and the already selected adapter using the identical
bounded decoding policy. Both raw initial and final CER are recorded, together
with retries and unresolved outputs. Training reports retain their original raw
greedy validation results. Unresolved outputs remain in the reported score. Evaluation
does not train, change the checkpoint, or select a different one based on test
results. Each evaluation requires a new ignored output directory.

Build an offline comparison after evaluation finishes:

```sh
.venv-qwen/bin/python scripts/build_qwen_training_comparison.py \
  --evaluation-dir data/qwen3_asr_lora_test
```

Open the generated `comparison.html` to listen to each test clip and compare the
original transcript, baseline prediction, and adapted prediction. The page shows
both raw and final character error rates, highlights text differences, and keeps
retry evidence visible. The page and its private transcripts stay under `data/`.

```sh
.venv-qwen/bin/python -m unittest discover -s test -p 'qwen_lora_test.py'
.venv-qwen/bin/python -m unittest discover -s test -p 'evaluate_qwen_lora_test.py'
```

### Portable adapter inference

An exported adapter package can be loaded without the private training manifests
or local run paths using `scripts/infer_qwen_adapter.py`:

```sh
.venv-qwen/bin/python scripts/infer_qwen_adapter.py \
  --model /absolute/path/to/pinned/base/snapshot \
  --adapter /absolute/path/to/downloaded/adapter \
  --audio /absolute/path/to/clip.wav
```

The package must use `qwen3-asr-mlx-lora-v1` configuration and contain
`selected.safetensors`. The loader verifies base configuration and weight hashes,
adapter integrity, and the fixed decoding policy. It accepts mono 16 kHz PCM16 WAV
clips up to 90 seconds and returns JSON with raw/final transcripts and retry
evidence. This custom MLX format is not a standalone Transformers or PEFT model.
Keep downloaded base models and adapters under ignored `models/` or `checkpoints/`.

## Full Qwen transcripts and comparison

Export all Qwen recognition output independently of alignment eligibility:

```sh
.venv-qwen/bin/python scripts/export_qwen_transcripts.py
.venv-qwen/bin/python scripts/build_qwen_comparison.py
```

The exporter reuses the complete ASR pass and retranscribes token-limited windows
in shorter intervals, without prompting with reference words. It writes plain
and timed transcripts under ignored `data/qwen_transcripts/transcripts/`, plus
`comparison.json`. Times identify input windows rather than word boundaries.
The frozen fine-tuning dataset and original exports remain unchanged.

Open `data/qwen_transcripts/comparison.html` to compare wording, inspect full
transcripts, and listen to the corresponding audio. The percentage is normalized
character edit distance against the original script, not model accuracy.
Spacing and punctuation are excluded from the score; English spelling versus
phonetic Korean still counts as a difference. The visual diff also shows
punctuation and spacing changes.

## Privacy and limitations

The server binds only to localhost, rejects cross-origin API calls, and processes speech locally. Initial package/model downloads require internet; local STT does not download weights during a lecture. Optional ChatGPT paraphrasing sends transcript text to OpenAI over the internet after consent. The legacy Zipformer model is from the official [sherpa-onnx Korean streaming release](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-korean-2024-06-16.tar.bz2). Training provenance is linked in the [upstream model documentation](https://github.com/k2-fsa/sherpa/blob/master/docs/source/onnx/pretrained_models/online-transducer/zipformer-transducer-models.rst).

No system/tab audio capture, speaker diarization, permanent audio storage, or cross-device access is implemented. Microphone interruptions, sleep, and tab closure can interrupt a lecture; check the save indicator before leaving. University recording rules and lecturer consent still apply.
