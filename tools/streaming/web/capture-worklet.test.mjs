import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

async function makeProcessor(rate){
  const source=await readFile(new URL('./capture-worklet.js',import.meta.url),'utf8');
  let Processor;
  class FakeProcessor {constructor(){this.messages=[];this.port={postMessage:(message)=>this.messages.push(message)};}}
  const sandbox={AudioWorkletProcessor:FakeProcessor,sampleRate:rate,currentFrame:0,
    registerProcessor:(name,klass)=>{assert.equal(name,'falcon-capture');Processor=klass;}};
  vm.runInNewContext(source,sandbox);
  return {instance:new Processor(),sandbox};
}
for(const rate of [16000,44100,48000])test(`flush preserves last impulse and stops at source boundary ${rate} Hz`,async()=>{
  const {instance,sandbox}=await makeProcessor(rate);
  const input=new Float32Array(400);input[399]=1;
  instance.process([[input]],[[new Float32Array(400)]]);
  instance.port.onmessage({data:{type:'flush'}});
  assert.equal(instance.messages.at(-1).type,'flushed');
  const packets=instance.messages.filter(m=>m.samples);
  const samples=packets.flatMap(m=>Array.from(m.samples));
  const expectedCount=Math.floor((400+31-1)/(rate/16000))+1;
  assert.equal(samples.length,expectedCount);
  // Initial time includes the 31-source-sample FIR delay; the last emitted
  // sample must never represent a source time beyond the real input interval.
  const firstTime=packets[0].startTime,lastTime=firstTime+(samples.length-1)/16000;
  assert.ok(lastTime < 400/rate && lastTime >= (400-rate/16000)/rate);
  assert.ok(samples.slice(-20).some(x=>Math.abs(x)>1e-5),'last input impulse must be represented');
  for(let i=1;i<packets.length;i++)assert.ok(Math.abs(packets[i].startTime-(packets[i-1].startTime+packets[i-1].samples.length/16000))<1e-12);
  const count=instance.messages.length;sandbox.currentFrame=400;instance.process([[input]],[[new Float32Array(400)]]);
  instance.port.onmessage({data:{type:'flush'}});assert.equal(instance.messages.length,count);
});

test('a missing render quantum is reported before further audio',async()=>{
  const {instance,sandbox}=await makeProcessor(16000);
  instance.process([[new Float32Array(128)]],[[new Float32Array(128)]]);
  sandbox.currentFrame=129;instance.process([[new Float32Array(128)]],[[new Float32Array(128)]]);
  assert.equal(instance.messages[0].type,'discontinuity');
});
