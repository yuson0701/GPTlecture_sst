import { pcmBase64 } from './audio.js';

// Fixed 200 ms packets containing NEW audio only. Never overlap or replay previews.
export class StreamingFrames {
  constructor(rate, onPacket) {
    this.rate = rate; this.onPacket = onPacket; this.buffer = new Float32Array(Math.round(rate / 5));
    this.offset = 0; this.sequence = 0; this.total = 0; this.closed = false;
  }
  push(samples) {
    if (this.closed) return;
    let cursor = 0;
    while (cursor < samples.length) {
      const count = Math.min(samples.length - cursor, this.buffer.length - this.offset);
      this.buffer.set(samples.subarray(cursor, cursor + count), this.offset);
      this.offset += count; this.total += count; cursor += count;
      if (this.offset === this.buffer.length) this.emit(false);
    }
  }
  emit(final) {
    const audio = pcmBase64(this.buffer.subarray(0, this.offset), this.rate);
    this.onPacket({ sequence: this.sequence++, audio, final, seconds: this.total / this.rate, capturedAt: performance.now() });
    this.offset = 0;
  }
  flush() { if (!this.closed) { this.closed = true; this.emit(true); } }
}

// All packets matter to a streaming decoder; no coalescing or silent dropping.
export class StreamingQueue {
  constructor(process, onChange = () => {}) { this.process = process; this.onChange = onChange; this.items = []; this.current = null; this.running = false; this.failed = null; this.waiters = []; }
  get waiting() { return this.items.filter(x => x !== this.current); }
  enqueue(packet) {
    if (this.items.length >= 50) return false;
    this.items.push(packet); this.onChange(); void this.pump(); return true;
  }
  async pump() {
    if (this.running || this.failed) return;
    this.running = true;
    try { while (this.items.length) { this.current = this.items[0]; await this.process(this.current); this.items.shift(); this.current = null; this.onChange(); } }
    catch (error) { this.failed = error; }
    finally { this.current = null; this.running = false; this.onChange(); this.waiters.splice(0).forEach(resolve => resolve()); }
  }
  async settle() { if (this.running) await new Promise(resolve => this.waiters.push(resolve)); }
  retry() { this.failed = null; return this.pump(); }
}
