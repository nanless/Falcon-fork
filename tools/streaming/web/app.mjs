import { AudioRing, RATE, CHUNK, FRAME, finalChunk, selectFrames, makePayload, decodeAudio } from "./stream-core.mjs";
import { createPlaybackUpsampler } from "/studio-assets/playback-resampler.js";
const $ = id => document.getElementById(id);
let current = null, generation = 0;
const diagnosticRecords=[];
window.falconStreamDiagnostics=()=>diagnosticRecords.slice();
const info = msg => { $("status").textContent = msg; };
function shutdown(message) {
  const s=current; current=null; ++generation;
  if(s){clearInterval(s.timer);clearTimeout(s.inflight?.timeout);clearTimeout(s.finishTimer);clearTimeout(s.flushTimer);try{s.worker?.postMessage({type:"stop"});s.worker?.terminate();}catch{};try{s.socket?.close();}catch{};
    try{s.source?.disconnect();s.worklet?.disconnect();}catch{};s.stream?.getTracks().forEach(t=>t.stop());
    for(const n of s.playing||[])try{n.stop();}catch{};if(s.context && s.context.state!=="closed")void s.context.close();}
  $("camera").srcObject=null;$("start").disabled=false;$("stop").disabled=true;$("tracking").textContent="嘴部跟踪：未启动";if(message)info(message);
}
function pcm(s, message) {
  if(current!==s || s.draining || !message?.samples?.length)return;
  const start=message.startTime;
  if (!Number.isFinite(start)) throw new Error("音频时钟无效");
  if(s.origin===null)s.origin=start;
  else if(Math.abs(start-(s.origin+s.ring.total/RATE))>.01){
    if(s.base===null){s.origin=start;s.ring=new AudioRing();s.frames=[];}
    else throw new Error(`音频时钟不连续（偏差 ${((start-(s.origin+s.ring.total/RATE))*1000).toFixed(1)} ms），请重新开始`);
  }
  s.ring.push(message.samples);dispatch(s);
}
function dispatch(s) {
  if(current!==s||s.socket?.readyState!==WebSocket.OPEN||!s.ready||s.inflight||s.origin===null||(!s.draining&&s.frames.length<10))return;
  if(s.draining && s.next>=s.end){finishPlayback(s);return;}
  if(s.base===null){
    // Anchor after a mouth frame has actually arrived. Its availability time is
    // conservative: never pretend camera processing completed in the past.
    const newest=s.frames[s.frames.length-1].sample;
    s.base=Math.ceil(Math.max(s.ring.total,newest)/320)*320;s.next=s.base;
  }
  if(s.ring.total-s.next>4*CHUNK)throw new Error("处理积压过多，请检查负载后重新开始");
  if(!s.draining && s.ring.total<s.next+CHUNK)return;
  const chunk=s.draining?finalChunk(s.ring,s.next,s.end):{audio:s.ring.read(s.next,CHUNK),valid:CHUNK};
  if(!chunk){finishPlayback(s);return;}
  const {audio,valid}=chunk, seq=s.sequence;
  const frames=selectFrames(s.frames,s.next,CHUNK,valid);
  if(!frames){
    // Once a chunk is complete, a past-only frame cannot arrive with an
    // earlier availability time. Stop rather than silently desynchronize.
    throw new Error("对应时间的嘴部画面缺失或过旧，请正对摄像头并重新开始");
  }
  const payload=makePayload(seq,s.next,audio,frames,valid);
  s.inflight={seq,start:s.next,valid,sent:performance.now(),dispatchMs:(s.context.currentTime-(s.origin+(s.next+CHUNK)/RATE))*1000,timeout:setTimeout(()=>{if(current===s)shutdown("模型响应超时，会话已停止。");},5000)};
  s.socket.send(JSON.stringify(payload));
}
function play(s, samples, captureStart) {
  const context=s.context;
  let data=samples;
  if(context.sampleRate!==RATE){if(!s.upsampler)s.upsampler=createPlaybackUpsampler(context.sampleRate,RATE);data=s.upsampler.push(samples);}
  if(!data.length)return;
  const buffer=context.createBuffer(1,data.length,context.sampleRate);buffer.copyToChannel(data,0);
  const source=context.createBufferSource();source.buffer=buffer;source.connect(s.gain);
  const when=Math.max(context.currentTime+.03,s.playAt);
  if(s.playAt && s.playAt<context.currentTime) s.gaps++;
  source.start(when);s.playAt=when+buffer.duration;s.playing.add(source);source.onended=()=>{s.playing.delete(source);source.disconnect();};
  const estimate=(when-(s.origin+captureStart/RATE))*1000;
  $("metrics").textContent=`音频时钟估算延迟 ${estimate.toFixed(0)} ms；播放断续 ${s.gaps} 次（不含设备固有延迟）。`;
  return {estimateMs:estimate,playQueueMs:(when-context.currentTime)*1000};
}
function warmupNext(s) {
  const seq=s.warmupSeq;
  if(seq>=3){s.socket.send(JSON.stringify({type:"reset"}));s.warmupReset=true;return;}
  const frames=Array.from({length:CHUNK / 640},(_,i)=>({sample:seq*CHUNK+i*640,data:new Uint8Array(FRAME).fill(128)}));
  s.socket.send(JSON.stringify(makePayload(seq,seq*CHUNK,new Float32Array(CHUNK),frames)));
}
function message(s, raw) {
  const result=JSON.parse(raw);
  if(result.type==="ready") {s.warmupSeq=0;info(`模型已就绪（${result.training_status}），正在预热连接。`);warmupNext(s);return;}
  if(s.warmupReset && result.type==="reset"){s.warmupReset=false;s.ready=true;info("连接预热完成，等待嘴部跟踪并开始实时处理。");dispatch(s);return;}
  if(!s.ready && result.type==="result"){if(result.sequence!==s.warmupSeq)throw new Error("预热响应序号无效");s.warmupSeq++;warmupNext(s);return;}
  if(result.type==="error")throw new Error(result.message||"模型服务拒绝输入");
  if(result.type!=="result"||!s.inflight||result.sequence!==s.inflight.seq||result.start_sample!==s.inflight.seq*CHUNK||result.end_sample!==s.inflight.seq*CHUNK+s.inflight.valid||result.valid_samples!==s.inflight.valid||result.guard_samples!==0)throw new Error("模型输出序号或时间无效");
  const y=decodeAudio(result.audio_f32le), item=s.inflight;clearTimeout(item.timeout);s.inflight=null;
  const arrivalMs=performance.now()-item.sent;
  const playback=play(s,y.subarray(0,item.valid),item.start);diagnosticRecords.push({sequence:item.seq,dispatchMs:item.dispatchMs,roundtripMs:arrivalMs,serverMs:result.timing_ms, ...playback});if(diagnosticRecords.length>300)diagnosticRecords.shift();
  s.sequence++;s.next+=CHUNK;
  s.frames=s.frames.filter(x=>x.sample>=s.next-2*CHUNK);
  info(`正在处理：第 ${s.sequence} 块；服务 ${Number(result.timing_ms?.session||0).toFixed(1)} ms。`);
  dispatch(s);
}
async function videoMessage(s, event) {
  if(current!==s || s.stopping)return;
  const m=event.data;
  if(m.type==="ack"){s.bitmapBusy=false;return;}
  if(m.type==="tracking"){
    $("tracking").textContent=`嘴部跟踪：${m.status}${m.message?" · "+m.message:""}`;
    if(m.status==="error"||m.requiresReacquire)throw new Error(m.message||"目标跟踪中断，请重新开始");
    return;
  }
  if(m.type==="error")throw new Error(m.message||"视频采集失败");
  if(m.type!=="frame"||s.origin===null)return;
  const jpeg=new Uint8Array(m.jpeg);
  if(!jpeg.length || jpeg.length>16000)throw new Error("嘴部图像数据大小无效");
  // The frame availability time is at least its actual capture time; using
  // availability in the audio clock conservatively excludes future frames.
  const sample=Math.floor((s.context.currentTime-s.origin)*RATE);
  if(sample>=0){s.frames.push({sample,jpeg});s.frames.sort((a,b)=>a.sample-b.sample);if(s.frames.length>250)s.frames.splice(0,s.frames.length-250);dispatch(s);}
}
function finishPlayback(s) {
  if(current!==s || s.finishing)return;
  s.finishing=true;
  // Drain the group-delay tail of the playback resampler before disconnecting.
  if(s.upsampler && s.upsampler.groupDelay)play(s,new Float32Array(s.upsampler.groupDelay),s.end);
  const delay=Math.max(0,(s.playAt-s.context.currentTime)*1000)+20;
  info("正在播放剩余音频。");
  s.finishTimer=setTimeout(()=>{if(current===s)shutdown("已停止，剩余音频已处理。");},delay);
}
function captureFlushed(s) {
  if(current!==s || !s.stopping || s.draining)return;
  clearTimeout(s.flushTimer);
  s.draining=true;s.end=s.ring.total;
  try{s.source?.disconnect();s.worklet?.disconnect();}catch{}
  s.stream?.getTracks().forEach(t=>t.stop());
  info("正在处理剩余音频。");
  try{dispatch(s);}catch(err){shutdown(err.message||String(err));}
}
function stopGracefully() {
  const s=current;
  if(!s)return;
  if(s.stopping)return;
  if(!s.ready || s.base===null){shutdown("已停止。");return;}
  s.stopping=true;
  $("stop").disabled=true;
  clearInterval(s.timer);
  try{s.worker?.postMessage({type:"stop"});s.worker?.terminate();}catch{}
  info("正在收取最后的音频。");
  if(!s.worklet?.port){captureFlushed(s);return;}
  s.flushTimer=setTimeout(()=>{if(current===s)shutdown("收取最后的音频超时，会话已停止。");},1000);
  s.worklet.port.postMessage({type:"flush"});
}
async function start() {
  if(current)return;
  diagnosticRecords.length=0;
  const id=++generation, s={id,origin:null,base:null,next:null,sequence:0,frames:[],ring:new AudioRing(),inflight:null,ready:false,playAt:0,playing:new Set(),gaps:0,bitmapBusy:false};
  current=s;$("start").disabled=true;$("stop").disabled=false;info("等待摄像头和麦克风授权。");
  try {
    if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia)throw new Error("设备访问需要本机安全上下文");
    s.context=new AudioContext({latencyHint:"interactive"});await s.context.resume();if(current!==s)return;
    s.context.onstatechange=()=>{if(current===s && s.context.state!=="running")shutdown("音频设备已暂停，会话已停止。");};
    const stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},frameRate:{ideal:25}},audio:{channelCount:{ideal:1},echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
    if(current!==s){stream.getTracks().forEach(t=>t.stop());return;}s.stream=stream;
    stream.getTracks().forEach(t=>t.addEventListener("ended",()=>{if(current===s&&!s.stopping)shutdown("设备连接已结束。");}));
    const video=$("camera");video.srcObject=stream;await video.play();if(current!==s)return;
    s.gain=s.context.createGain();s.gain.gain.value=$("monitor").checked?1:0;s.gain.connect(s.context.destination);
    await s.context.audioWorklet.addModule("/stream-ui/capture-worklet.js");if(current!==s)return;
    s.worklet=new AudioWorkletNode(s.context,"falcon-capture",{numberOfInputs:1,numberOfOutputs:1,outputChannelCount:[1]});
    s.source=s.context.createMediaStreamSource(stream);s.source.connect(s.worklet);s.worklet.connect(s.context.destination);
    s.worklet.port.onmessage=e=>{try{
      if(e.data?.type==="flushed")captureFlushed(s);
      else if(e.data?.type==="discontinuity"){
        if(s.base!==null)throw new Error("音频采集不连续，会话已停止。");
        s.origin=null;s.ring=new AudioRing();s.frames=[];
      } else pcm(s,e.data);
    }catch(err){if(current===s)shutdown(err.message);}};
    if(!window.OffscreenCanvas||!window.Worker||!window.createImageBitmap)throw new Error("当前浏览器缺少视频处理支持，请使用新版 Chrome");
    s.worker=new Worker("/studio-assets/video-worker.js",{type:"module"});
    s.worker.onmessage=e=>{void videoMessage(s,e).catch(err=>{if(current===s)shutdown(err.message);});};
    s.worker.onerror=e=>{if(current===s)shutdown(`嘴部跟踪失败：${e.message}`);};
    s.worker.postMessage({type:"start",tracking:"auto",trackingEpoch:1,roi:{cx:.5,cy:.6,size:.2}});
    s.timer=setInterval(async()=>{
      if(current!==s||s.bitmapBusy||video.readyState<2)return;
      s.bitmapBusy=true;
      try{const bitmap=await createImageBitmap(video);if(current!==s || s.stopping){bitmap.close();return;}s.worker.postMessage({type:"frame",bitmap,wallTime:performance.timeOrigin+performance.now(),trackingEpoch:1},[bitmap]);}
      catch(err){s.bitmapBusy=false;if(current===s)shutdown(`摄像头处理失败：${err.message}`);}
    },40);
    s.socket=new WebSocket(`${location.protocol==="https:"?"wss":"ws"}://${location.host}/ws`);
    s.socket.onmessage=e=>{if(current===s)try{message(s,e.data);}catch(err){shutdown(err.message);}};
    s.socket.onerror=()=>{if(current===s)shutdown("模型连接失败，请检查服务。");};
    s.socket.onclose=()=>{if(current===s)shutdown("模型连接已断开，请重新开始；会话状态已清空。");};
  } catch(err){if(current===s)shutdown(err.message||String(err));}
}
$("start").addEventListener("click",()=>void start());
$("stop").addEventListener("click",stopGracefully);
$("monitor").addEventListener("change",()=>{if(current?.gain)current.gain.gain.value=$("monitor").checked?1:0;});
window.addEventListener("pagehide",()=>shutdown());
