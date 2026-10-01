export function pcmBase64(samples, sampleRate) {
  const count = Math.floor(samples.length * 16000 / sampleRate), bytes = new Uint8Array(count * 2), view = new DataView(bytes.buffer);
  for (let i = 0; i < count; i++) {
    const from = Math.floor(i * sampleRate / 16000), to = Math.min(samples.length, Math.max(from + 1, Math.floor((i + 1) * sampleRate / 16000)));
    let total = 0; for (let j = from; j < to; j++) total += samples[j];
    const value = Math.max(-1, Math.min(1, total / (to - from)));
    view.setInt16(i * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
  }
  let binary = ''; for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(binary);
}

// Prefer a short pause after 3 seconds; cap continuous speech at 10 seconds.
// Overlap forced splits by 400 ms to reduce cut-off words.
export class Segmenter {
  constructor(sampleRate, onChunk) { this.rate = sampleRate; this.onChunk = onChunk; this.parts = []; this.length = 0; this.total = 0; this.silent = 0; this.overlap = false; this.newSamples = 0; }
  push(samples) {
    this.parts.push(samples); this.length += samples.length; this.total += samples.length; this.newSamples += samples.length;
    const rms = Math.sqrt(samples.reduce((sum, value) => sum + value * value, 0) / samples.length);
    this.silent = rms < 0.008 ? this.silent + samples.length : 0;
    if (this.length >= this.rate * 3 && this.silent >= this.rate * 0.7) this.emit(false);
    else if (this.length >= this.rate * 10) this.emit(true);
  }
  emit(overlapNext) {
    if (this.length < this.rate * 0.1 || !this.newSamples) return;
    const samples = new Float32Array(this.length); let offset = 0;
    for (const part of this.parts) { samples.set(part, offset); offset += part.length; }
    const chunk = { audio: pcmBase64(samples, this.rate), seconds: (this.total - this.length) / this.rate, overlap: this.overlap };
    const tail = overlapNext ? samples.slice(-Math.floor(this.rate * 0.4)) : new Float32Array();
    this.parts = tail.length ? [tail] : []; this.length = tail.length; this.silent = 0; this.overlap = overlapNext; this.newSamples = 0;
    this.onChunk(chunk);
  }
  flush() { this.emit(false); this.parts = []; this.length = 0; }
}

export function removeOverlap(previous, current) {
  const normalize = text => [...text].filter(c => /[\p{L}\p{N}]/u.test(c)).join('');
  const left = normalize(previous).slice(-60), right = normalize(current);
  for (let size = Math.min(left.length, right.length, 60); size >= 6; size--) {
    if (left.slice(-size) !== right.slice(0, size)) continue;
    let count = 0, index = 0;
    for (const char of current) { index += char.length; if (/[\p{L}\p{N}]/u.test(char)) count++; if (count === size) break; }
    return current.slice(index).replace(/^[\s,.!?。]+/, '');
  }
  return current;
}

// One request at a time. Failed chunks remain first in line for an explicit retry.
export class AudioQueue {
  constructor(process, onChange = () => {}, limit = 8) { this.process = process; this.onChange = onChange; this.limit = limit; this.items = []; this.running = false; this.failed = null; this.waiters = []; }
  enqueue(item) { if (this.items.length >= this.limit) return false; this.items.push(item); this.onChange(); void this.pump(); return true; }
  async pump() {
    if (this.running || this.failed) return;
    this.running = true;
    try { while (this.items.length) { await this.process(this.items[0]); this.items.shift(); this.onChange(); } }
    catch (error) { this.failed = error; }
    finally { this.running = false; this.onChange(); this.waiters.splice(0).forEach(resolve => resolve()); }
  }
  async settle() { if (this.running) await new Promise(resolve => this.waiters.push(resolve)); }
  retry() { this.failed = null; return this.pump(); }
}

export class Microphone {
  async start(onChunk, onEnded) {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }, video: false });
      this.context = new AudioContext(); this.segmenter = new Segmenter(this.context.sampleRate, onChunk);
      await this.context.audioWorklet.addModule('/capture-worklet.js');
      this.source = this.context.createMediaStreamSource(this.stream);
      this.node = new AudioWorkletNode(this.context, 'lecture-capture');
      this.node.port.onmessage = ({ data }) => { if (data.samples) this.segmenter.push(data.samples); if (data.flushed) this.flushed?.(); };
      this.source.connect(this.node); this.node.connect(this.context.destination);
      for (const track of this.stream.getTracks()) track.onended = () => onEnded('마이크 연결이 끊겼습니다.');
      this.context.onstatechange = () => { if (!this.stopping && ['suspended', 'interrupted'].includes(this.context.state)) onEnded('마이크 처리가 중단되었습니다. 브라우저 탭을 활성화해 주세요.'); };
      await this.context.resume();
    } catch (error) { this.dispose(); throw error; }
  }
  async stop() {
    this.stopping = true;
    this.source?.disconnect(); this.stream?.getTracks().forEach(track => { track.onended = null; track.stop(); });
    if (this.node && this.context?.state === 'running') {
      await new Promise(resolve => { const timeout = setTimeout(resolve, 1000); this.flushed = () => { clearTimeout(timeout); resolve(); }; this.node.port.postMessage('flush'); });
    }
    this.segmenter?.flush(); this.dispose();
  }
  dispose() {
    this.stopping = true;
    this.stream?.getTracks().forEach(track => { track.onended = null; track.stop(); });
    this.source?.disconnect(); this.node?.disconnect(); this.node?.port.close();
    if (this.context && !this.closing) { this.closing = true; this.context.onstatechange = null; void this.context.close().catch(() => {}); }
  }
}
