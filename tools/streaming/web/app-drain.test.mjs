import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import * as core from './stream-core.mjs';

test('stop drains in-flight, full and partial audio before closing playback', async () => {
  const elements = new Map();
  globalThis.document = { getElementById(id) {
    if (!elements.has(id)) elements.set(id, {disabled:false,textContent:'',srcObject:null,addEventListener(){}});
    return elements.get(id);
  }};
  globalThis.window = {addEventListener(){}};
  globalThis.__stream_core_test = core;
  let source = await readFile(new URL('./app.mjs', import.meta.url), 'utf8');
  source = source.replace(/^import \{[^\n]+\} from "\.\/stream-core\.mjs";/m,
    'const { AudioRing, RATE, CHUNK, FRAME, finalChunk, selectFrames, makePayload, decodeAudio } = globalThis.__stream_core_test;');
  source = source.replace(/^import \{ createPlaybackUpsampler \} from "\/studio-assets\/playback-resampler\.js";/m,
    'const createPlaybackUpsampler = () => { throw new Error("unexpected resampling"); };');
  source += '\nexport { dispatch, message, stopGracefully, pcm, captureFlushed, shutdown }; export function setCurrent(s){current=s;}';
  const app = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
  const sent = [], played = [], workletMessages = [];
  const context = {sampleRate:16000, currentTime:1, state:'running',
    createBuffer(channels,length){return {duration:length/16000, copyToChannel(data){played.push([...data]);}};},
    createBufferSource(){return {connect(){},disconnect(){},start(){},stop(){}};},
    close(){this.state='closed';return Promise.resolve();}};
  const ring = new core.AudioRing(); ring.push(new Float32Array(core.CHUNK+19).fill(.1));
  const frames=[0,320,640,960,1280,1600,1920,2240,2400,2560].map(sample=>({sample,data:new Uint8Array(core.FRAME)}));
  const s={ring,frames,base:0,next:0,sequence:0,origin:0,ready:true,inflight:null,
    context,gain:{},playAt:0,playing:new Set(),gaps:0,socket:{readyState:1,send(x){sent.push(JSON.parse(x));},close(){}},
    stream:{getTracks(){return [];}},worklet:{port:{postMessage(x){workletMessages.push(x);}},disconnect(){}}};
  globalThis.WebSocket={OPEN:1};
  app.setCurrent(s);app.dispatch(s);
  assert.equal(sent.length,1);assert.ok(s.inflight);
  app.stopGracefully();
  assert.equal(s.stopping,true);assert.equal(s.draining,undefined);
  assert.deepEqual(workletMessages,[{type:"flush"}]);
  app.pcm(s,{samples:new Float32Array(7).fill(.2),startTime:ring.total/16000});
  app.captureFlushed(s);
  assert.equal(s.end,core.CHUNK+26);
  assert.equal(sent.length,1);assert.equal(sent[0].sequence,0);
  function result(sequence,value,valid=core.CHUNK){
    const audio=new Float32Array(core.CHUNK).fill(value), bytes=new Uint8Array(audio.buffer);
    return JSON.stringify({type:'result',sequence,start_sample:sequence*core.CHUNK,end_sample:sequence*core.CHUNK+valid,valid_samples:valid,
      guard_samples:0,audio_f32le:Buffer.from(bytes).toString('base64'),timing_ms:{session:1}});
  }
  app.message(s,result(0,.25));
  assert.equal(sent.length,2);assert.equal(sent[1].sequence,1);
  assert.equal(sent[1].valid_samples,26);
  app.message(s,result(1,.75,26));
  assert.deepEqual(played.map(x=>x.length),[core.CHUNK,26]);
  assert.ok(played[1].every(x=>x===.75));
  // The finite timer waits for queued playback; cancel it in this fake clock.
  clearTimeout(s.finishTimer);app.setCurrent(null);
  let closed=0;
  const stalled={...s,draining:false,stopping:false,finishing:false,finishTimer:null,
    socket:{...s.socket,close(){closed++;}},worklet:{port:{postMessage(){}},disconnect(){}}};
  app.setCurrent(stalled);app.stopGracefully();
  await new Promise(resolve=>setTimeout(resolve,1100));
  assert.equal(closed,1);assert.match(elements.get("status").textContent,/超时/);
  // A late acknowledgment from the closed session must have no effect.
  app.captureFlushed(stalled);assert.equal(closed,1);
});
