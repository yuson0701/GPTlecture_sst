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

// Speech-gated rolling recognition. Silence never creates a transcript request.
export class Segmenter {
  constructor(sampleRate, onChunk, { threshold = 0.008, previewSeconds = 0.35, pauseSeconds = 0.5 } = {}) {
    this.rate = sampleRate; this.onChunk = onChunk; this.threshold = threshold;
    this.previewSamples = sampleRate * previewSeconds; this.pauseSamples = sampleRate * pauseSeconds;
    this.parts = []; this.length = 0; this.total = 0; this.silent = 0; this.speech = 0;
    this.preRoll = []; this.preLength = 0; this.active = false; this.serial = 0;
    this.overlap = false; this.lastPreview = 0;
  }
  push(samples) {
    this.total += samples.length;
    const rms = Math.sqrt(samples.reduce((sum, value) => sum + value * value, 0) / samples.length);
    const voiced = rms >= this.threshold;
    if (!this.active) {
      if (!voiced) {
        this.preRoll.push(samples); this.preLength += samples.length;
        while (this.preLength > this.rate * 0.2 && this.preRoll.length > 1) this.preLength -= this.preRoll.shift().length;
        return;
      }
      this.active = true; this.id = `speech-${this.serial++}`;
      this.parts = this.preRoll; this.length = this.preLength; this.preRoll = []; this.preLength = 0;
      this.start = (this.total - samples.length - this.length) / this.rate;
      this.speech = 0; this.silent = 0; this.lastPreview = 0;
    }
    this.parts.push(samples); this.length += samples.length;
    if (voiced) { this.speech += samples.length; this.silent = 0; } else this.silent += samples.length;
    if (this.silent >= this.pauseSamples) this.finish(false);
    else if (this.length >= this.rate * 8) this.finish(true);
    else if (voiced && this.speech >= this.rate * 0.16 && this.length - this.lastPreview >= this.previewSamples) {
      this.emit(false); this.lastPreview = this.length;
    }
  }
  samples() {
    const samples = new Float32Array(this.length); let offset = 0;
    for (const part of this.parts) { samples.set(part, offset); offset += part.length; }
    return samples;
  }
  emit(final) {
    this.onChunk({ id: this.id, audio: pcmBase64(this.samples(), this.rate), seconds: this.start, overlap: this.overlap, final });
  }
  finish(continueSpeech) {
    if (!this.active) return;
    // Reject short clicks; the model's neural VAD additionally checks speech.
    if (this.speech >= this.rate * 0.16) this.emit(true);
    const tail = continueSpeech ? this.samples().slice(-Math.floor(this.rate * 0.4)) : new Float32Array();
    this.parts = []; this.length = 0; this.active = false; this.speech = 0; this.silent = 0;
    this.preRoll = tail.length ? [tail] : []; this.preLength = tail.length; this.overlap = continueSpeech;
  }
  flush() { this.finish(false); this.preRoll = []; this.preLength = 0; }
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

// Keep final audio reliable, but coalesce obsolete previews on slower hardware.
export class AudioQueue {
  constructor(process, onChange = () => {}, limit = 8) { this.process = process; this.onChange = onChange; this.limit = limit; this.items = []; this.running = false; this.current = null; this.failed = null; this.waiters = []; }
  enqueue(item) {
    if (item.id !== undefined) {
      // Never mutate an in-flight request. A final replaces queued previews.
      const index = this.items.findIndex(x => x !== this.current && x.id === item.id);
      if (index >= 0) {
        if (this.items[index].final !== false && item.final === false) return true;
        this.items.splice(index, 1);
      }
      // Preview requests are expendable; final requests are not.
      if (item.final === false && this.items.some(x => x !== this.current && x.final !== false)) return true;
      if (item.final !== false) this.items = this.items.filter(x => x === this.current || x.final !== false || x.id !== item.id);
    }
    if (this.items.length >= this.limit) {
      const preview = this.items.findIndex(x => x !== this.current && x.final === false);
      if (preview >= 0) this.items.splice(preview, 1);
      else return item.final === false;
    }
    this.items.push(item); this.onChange(); void this.pump(); return true;
  }
  async pump() {
    if (this.running || this.failed) return;
    this.running = true;
    try {
      while (this.items.length) {
        this.current = this.items[0];
        try { await this.process(this.current); }
        catch (error) { if (this.current.final !== false) throw error; }
        this.items.shift(); this.current = null; this.onChange();
      }
    } catch (error) { this.failed = error; }
    finally { this.current = null; this.running = false; this.onChange(); this.waiters.splice(0).forEach(resolve => resolve()); }
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
