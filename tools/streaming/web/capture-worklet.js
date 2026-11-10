/* Clocked, streaming 16 kHz capture. A 63-tap polyphase low-pass FIR prevents
 * frequencies above the speech Nyquist band aliasing during 44.1/48 kHz capture.
 * Only output samples are convolved (~1M multiply-adds/s); time stamps compensate
 * the causal FIR group delay, preserving alignment with camera frame times. */
class FalconCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.ratio = sampleRate / this.targetRate;
    this.nextPosition = null;
    this.expectedFrame = null;
    this.taps = 63;
    this.groupDelay = (this.taps - 1) / 2;
    this.history = new Float32Array(this.taps + 2);
    this.historyIndex = 0;
    const cutoff = Math.min(7200, sampleRate * .45) / sampleRate;
    this.phases = Array.from({ length: 128 }, (_, phase) => {
      const coefficients = new Float32Array(this.taps);
      let sum = 0;
      for (let k = 0; k < this.taps; k++) {
        const x = k - this.groupDelay - phase / 128;
        const sinc = Math.abs(x) < 1e-8 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * x) / (Math.PI * x);
        coefficients[k] = sinc * (.54 - .46 * Math.cos(2 * Math.PI * k / (this.taps - 1)));
        sum += coefficients[k];
      }
      for (let k = 0; k < this.taps; k++) coefficients[k] /= sum;
      return coefficients;
    });
    this.packet = new Float32Array(320);
    this.packetLength = 0;
    this.packetStart = 0;
    this.stopped = false;
    this.port.onmessage = ({ data }) => {
      if (data?.type !== "flush" || this.stopped) return;
      // Flush only the FIR group-delay tail, preserving source-time coverage.
      // The synthetic zeros are never made visible beyond that exact time span.
      if (this.expectedFrame !== null)
        this.process([[new Float32Array(this.groupDelay)]], [], this.expectedFrame);
      this.stopped = true;
      if (this.packetLength) {
        const samples = this.packet.slice(0, this.packetLength);
        this.port.postMessage({ samples, startTime: this.packetStart }, [samples.buffer]);
        this.packetLength = 0;
      }
      // FIFO guarantees that all emitted audio reaches the client before this acknowledgment.
      this.port.postMessage({ type: "flushed" });
    };
  }
  process(inputs, outputs, virtualStart) {
    const channels = inputs[0];
    const frameStart = virtualStart ?? currentFrame;
    // The connected output is deliberately silent: raw microphone audio never monitors itself.
    for (const channel of outputs[0] || []) channel.fill(0);
    if (this.stopped || !channels || !channels.length || !channels[0].length) return true;
    if (this.nextPosition !== null && this.expectedFrame !== null && this.expectedFrame !== frameStart)
      this.port.postMessage({ type: "discontinuity" });
    if (this.nextPosition === null || (this.expectedFrame !== null && this.expectedFrame !== frameStart)) {
      this.nextPosition = frameStart;
      this.history.fill(0);
      this.packetLength = 0;
    }
    const count = channels[0].length;
    this.expectedFrame = frameStart + count;
    for (let i = 0; i < count; i++) {
      let value = 0;
      for (const channel of channels) value += channel[i] || 0;
      value /= channels.length;
      this.history[this.historyIndex] = value;
      const position = frameStart + i;
      while (this.nextPosition <= position + 1e-8) {
        const phase = Math.max(0, Math.min(127, Math.round((position - this.nextPosition) * 128)));
        const coefficients = this.phases[phase];
        let sample = 0;
        for (let k = 0; k < this.taps; k++) sample += this.history[(this.historyIndex - k + this.history.length) % this.history.length] * coefficients[k];
        if (!this.packetLength) this.packetStart = (this.nextPosition - this.groupDelay) / sampleRate;
        this.packet[this.packetLength++] = sample;
        this.nextPosition += this.ratio;
        if (this.packetLength === this.packet.length) {
          const packet = this.packet;
          this.port.postMessage({ samples: packet, startTime: this.packetStart }, [packet.buffer]);
          this.packet = new Float32Array(320);
          this.packetLength = 0;
        }
      }
      this.historyIndex = (this.historyIndex + 1) % this.history.length;
    }
    return true;
  }
}
registerProcessor('falcon-capture', FalconCapture);
