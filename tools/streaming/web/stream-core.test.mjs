import test from "node:test";
import assert from "node:assert/strict";
import { AudioRing, CHUNK, FRAME, finalChunk, selectFrames, makePayload, decodeAudio } from "./stream-core.mjs";
const f=(sample,byte=0)=>({sample,data:new Uint8Array(FRAME).fill(byte)});
test("selects only the latest available past frame for each interval",()=>{
  const x=selectFrames([f(999),f(1001),f(1500),f(1641),f(2300),f(2921)],1000);
  assert.deepEqual(x.map(y=>y.sample),[999,1500,1641,2300]);
  assert.equal(selectFrames([f(1001)],1000),null);
  assert.equal(selectFrames([f(-3000)],1000),null);
});
test("audio ring preserves sequence and detects overwrite",()=>{
  const r=new AudioRing(8);r.push([0,1,2,3,4,5]);assert.deepEqual([...r.read(1,4)],[1,2,3,4]);
  r.push([6,7,8,9]);assert.throws(()=>r.read(1,1));assert.deepEqual([...r.read(2,8)],[2,3,4,5,6,7,8,9]);
  assert.throws(()=>r.push([NaN]));
});
test("payload has causal relative timestamps, sequence and little-endian float",()=>{
  const a=new Float32Array(CHUNK);a[0]=.5;a[1]=-.25;
  const frames=[f(999,1),f(1500,2),f(1641,3),f(2300,4)];
  const q=makePayload(2,1000,a,frames);
  assert.equal(q.start_sample,5120);assert.deepEqual(q.video_samples,[5119,5620,5761,6420]);
  assert.deepEqual([...decodeAudio(q.audio_f32le).slice(0,2)],[.5,-.25]);
  assert.equal(atob(q.mouth_gray_u8).length,4*FRAME);
  assert.throws(()=>decodeAudio(btoa("abc")));
});
test("JPEG transport sends four compressed frames with unchanged timestamps",()=>{
  const fs=[0,640,1280,1920].map((sample,i)=>({sample,jpeg:new Uint8Array([255,216,i,255,217])}));
  assert.equal(selectFrames(fs,0).length,4);
  const payload=makePayload(0,0,new Float32Array(CHUNK),fs);
  assert.equal(payload.mouth_gray_u8,undefined);assert.equal(payload.mouth_jpeg.length,4);
  assert.deepEqual(payload.video_samples,[0,640,1280,1920]);assert.equal(atob(payload.mouth_jpeg[0]).length,5);
});

test("final partial block is padded without losing or fabricating valid samples",()=>{
  const ring=new AudioRing(CHUNK*3), source=Float32Array.from({length:CHUNK+19},(_,i)=>i+1);
  ring.push(source);
  const first=finalChunk(ring,0,source.length), tail=finalChunk(ring,CHUNK,source.length);
  assert.equal(first.valid,CHUNK);assert.deepEqual(first.audio,source.subarray(0,CHUNK));
  assert.equal(tail.valid,19);assert.deepEqual(tail.audio.subarray(0,19),source.subarray(CHUNK));
  assert.ok(tail.audio.subarray(19).every(x=>x===0));
  assert.equal(finalChunk(ring,source.length,source.length),null);
  assert.throws(()=>finalChunk(ring,source.length+1,source.length));
});

test("only real audio intervals enforce video freshness in final partial block",()=>{
  const old=f(-2000);
  assert.equal(selectFrames([old],0),null);
  const picked=selectFrames([old],0,CHUNK,20);
  assert.equal(picked.length,4);assert.ok(picked.every(frame=>frame.sample===-2000));
  const q=makePayload(0,0,new Float32Array(CHUNK),picked,20);
  assert.equal(q.valid_samples,20);
  assert.equal(makePayload(0,0,new Float32Array(CHUNK),picked).valid_samples,undefined);
});
