class LectureCapture extends AudioWorkletProcessor {
  constructor() {
    super(); this.buffer = new Float32Array(2048); this.offset = 0; this.stopped = false;
    this.port.onmessage = ({ data }) => { if (data === 'flush') { this.stopped = true; this.flush(); this.port.postMessage({ flushed: true }); } };
  }
  flush() { if (this.offset) { const samples = this.buffer.slice(0, this.offset); this.port.postMessage({ samples }, [samples.buffer]); this.offset = 0; } }
  process(inputs) {
    if (this.stopped) return false;
    const channels = inputs[0];
    if (channels?.length) for (let i = 0; i < channels[0].length; i++) {
      let value = 0; for (const channel of channels) value += channel[i];
      this.buffer[this.offset++] = value / channels.length;
      if (this.offset === this.buffer.length) this.flush();
    }
    // Outputs stay silent: no microphone feedback through speakers.
    return true;
  }
}
registerProcessor('lecture-capture', LectureCapture);
