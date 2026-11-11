export const RATE = 16000, CHUNK = 1920, BIN = 640, FRAME = 96 * 96;
export class AudioRing {
  constructor(capacity = RATE * 8) { this.data = new Float32Array(capacity); this.total = 0; }
  push(values) { for (const v of values) { if (!Number.isFinite(v)) throw new Error("非有限音频采样"); this.data[this.total++ % this.data.length] = v; } }
  read(start, length) {
    if (!Number.isSafeInteger(start) || start < Math.max(0, this.total - this.data.length) || start + length > this.total) throw new Error("音频缓存不足或已覆盖");
    const out = new Float32Array(length);
    for (let i = 0; i < length; i++) out[i] = this.data[(start + i) % this.data.length];
    return out;
  }
}
// Pad only the final partial block; valid samples are retained for trimming output.
export function finalChunk(ring, start, end) {
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start || end > ring.total)
    throw new Error("收尾音频范围无效");
  const valid = Math.min(CHUNK, end - start);
  if (!valid) return null;
  const audio = new Float32Array(CHUNK);
  audio.set(ring.read(start, valid));
  return {audio, valid};
}
export function selectFrames(frames, start, maxAge = CHUNK, valid = CHUNK) {
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(valid) || valid < 1 || valid > CHUNK) throw new Error("音频时间无效");
  const picked = [];
  for (let i = 0; i < CHUNK / BIN; i++) {
    const target = start + i * BIN;
    let best = null;
    for (const frame of frames) if (frame.sample <= target && (!best || frame.sample >= best.sample)) best = frame;
    if (!best || (i * BIN < valid && target - best.sample > maxAge)) return null;
    if (!((best.data instanceof Uint8Array && best.data.length === FRAME) || (best.jpeg instanceof Uint8Array && best.jpeg.length > 0 && best.jpeg.length <= 16000))) throw new Error("嘴部图像尺寸无效");
    picked.push(best);
  }
  return picked;
}
export function makePayload(sequence, startAbs, audio, frames, valid = CHUNK) {
  if (audio.length !== CHUNK || frames.length !== CHUNK / BIN || !Number.isSafeInteger(valid) || valid < 1 || valid > CHUNK) throw new Error("输入块尺寸无效");
  const bytes = new Uint8Array(CHUNK * 4), view = new DataView(bytes.buffer);
  for (let i=0;i<CHUNK;i++) view.setFloat32(i*4, audio[i], true);
  const payload={type:"chunk",sequence,start_sample:sequence*CHUNK,video_samples:frames.map(x=>x.sample-startAbs+sequence*CHUNK),audio_f32le:encode(bytes)};
  if(valid < CHUNK) payload.valid_samples=valid;
  if(frames.every(frame=>frame.jpeg instanceof Uint8Array)) payload.mouth_jpeg=frames.map(frame=>encode(frame.jpeg));
  else {const pictures=new Uint8Array((CHUNK / BIN)*FRAME);frames.forEach((frame,i)=>pictures.set(frame.data,i*FRAME));payload.mouth_gray_u8=encode(pictures);}
  return payload;
}
function encode(bytes) { let s=""; for(let i=0;i<bytes.length;i+=8192) s+=String.fromCharCode(...bytes.subarray(i,i+8192)); return btoa(s); }
export function decodeAudio(encoded) {
  const str=atob(encoded); if(str.length!==CHUNK*4) throw new Error("输出音频长度无效");
  const data=new Uint8Array(str.length); for(let i=0;i<str.length;i++) data[i]=str.charCodeAt(i);
  const view=new DataView(data.buffer), out=new Float32Array(CHUNK);
  for(let i=0;i<CHUNK;i++){out[i]=view.getFloat32(i*4,true);if(!Number.isFinite(out[i])) throw new Error("输出含非有限采样");}
  return out;
}
