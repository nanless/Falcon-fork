/* Real localization-module checks with a small DOM. No browser permissions,
 * network, camera, audio capture, or application reset is requested.
 * Run: node tests/check_i18n.mjs
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../web');
const coreURL = pathToFileURL(path.join(web, 'i18n.js')).href;
const han = /\p{Script=Han}/u;
let serial = 0, passed = 0;

function createDOM() {
  const all = [];
  class Element {
    constructor(tagName = 'div') {
      this.tagName = tagName.toUpperCase();
      this.attrs = new Map();
      this.textContent = ''; this.value = ''; this.lang = '';
      this.listeners = new Map(); this.children = [];
      this.dataset = new Proxy({}, {
        get: (_, key) => this.getAttribute(`data-${String(key).replace(/[A-Z]/g, (s) => `-${s.toLowerCase()}`)}`) ?? undefined,
        set: (_, key, value) => { this.setAttribute(`data-${String(key).replace(/[A-Z]/g, (s) => `-${s.toLowerCase()}`)}`, value); return true; },
      });
      all.push(this);
    }
    get id() { return this.getAttribute('id') || ''; }
    set id(value) { this.setAttribute('id', value); }
    get attributes() { return [...this.attrs].map(([name, value]) => ({ name, value })); }
    setAttribute(name, value) { this.attrs.set(name, String(value)); }
    getAttribute(name) { return this.attrs.get(name) ?? null; }
    hasAttribute(name) { return this.attrs.has(name); }
    removeAttribute(name) { this.attrs.delete(name); }
    addEventListener(type, callback) { this.listeners.set(type, callback); }
    append(...children) { this.children.push(...children); }
    matches(selector) {
      return selector.split(',').some((part) => {
        part = part.trim();
        if (part === '*') return true;
        if (part.startsWith('#')) return this.id === part.slice(1);
        const attribute = part.match(/^\[([^=\]]+)(?:=["']?([^"'\]]+)["']?)?\]$/);
        return attribute ? this.hasAttribute(attribute[1]) && (attribute[2] === undefined || this.getAttribute(attribute[1]) === attribute[2]) : this.tagName.toLowerCase() === part;
      });
    }
    querySelectorAll(selector) { return all.filter((node) => node !== this && node.matches(selector)); }
    querySelector(selector) { return this.querySelectorAll(selector)[0] ?? null; }
  }
  const document = {
    title: '', documentElement: new Element('html'), body: new Element('body'),
    createElement: (name) => new Element(name),
    getElementById: (id) => all.find((node) => node.id === id) ?? null,
    querySelectorAll: (selector) => all.filter((node) => node.matches(selector)),
    querySelector: (selector) => all.find((node) => node.matches(selector)) ?? null,
    addEventListener() {},
  };
  const select = document.createElement('select'); select.id = 'languageSelect';
  return { document, Element, select };
}

async function load({ saved = null, blocked = false } = {}) {
  const dom = createDOM();
  const values = new Map(saved === null ? [] : [['falcon.language', saved]]);
  const storage = {
    getItem(key) { if (blocked) throw new Error('Storage disabled'); return values.get(key) ?? null; },
    setItem(key, value) { if (blocked) throw new Error('Storage disabled'); values.set(key, value); },
  };
  globalThis.document = dom.document;
  globalThis.localStorage = storage;
  globalThis.window = { localStorage: storage, addEventListener() {} };
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { language: 'zh-CN', languages: ['zh-CN'] } });
  const api = await import(`${coreURL}?test=${++serial}`);
  api.initializeI18n();
  return { api, ...dom, values };
}

async function test(name, run) {
  await run();
  passed++;
  console.log(`PASS ${name}`);
}

await test('first launch is English even when browser language is Chinese', async () => {
  const { api, document, select } = await load();
  assert.equal(api.getLanguage(), 'en');
  assert.equal(document.documentElement.lang, 'en');
  assert.equal(select.value, 'en');
});

await test('saved Chinese preference is respected', async () => {
  const { api, document, select } = await load({ saved: 'zh' });
  assert.equal(api.getLanguage(), 'zh');
  assert.ok(document.documentElement.lang.startsWith('zh'));
  assert.equal(select.value, 'zh');
});

await test('invalid preference falls back to English', async () => {
  const { api } = await load({ saved: 'fr' });
  assert.equal(api.getLanguage(), 'en');
});

await test('blocked storage does not prevent startup or in-memory language changes', async () => {
  const { api } = await load({ blocked: true });
  assert.equal(api.getLanguage(), 'en');
  assert.doesNotThrow(() => api.setLanguage('zh'));
  assert.equal(api.getLanguage(), 'zh');
});

const dictionaries = [];
for (const filename of ['i18n-static.js', 'i18n-dynamic.js', 'i18n-errors.js', 'i18n-audio.js']) {
  const module = await import(pathToFileURL(path.join(web, filename)).href);
  for (const object of Object.values(module)) {
    if (object && typeof object === 'object' && !Array.isArray(object)) {
      const entries = Object.entries(object).filter(([, value]) => typeof value === 'string');
      if (entries.length) dictionaries.push(...entries);
    }
  }
}
assert.ok(dictionaries.length > 50, 'expected real static and dynamic translations');

await test('all Chinese dictionary keys translate and preserve every placeholder', async () => {
  const { api } = await load();
  for (const [key, expected] of dictionaries) {
    const names = [...key.matchAll(/\{(\w+)\}/g)].map((match) => match[1]);
    const params = Object.fromEntries(names.map((name) => [name, `TEST_${name}`]));
    const translated = api.t(key, params);
    const expectedText = expected.replace(/\{(\w+)\}/g, (token, name) => params[name] ?? token);
    assert.equal(translated, expectedText, `Dictionary mappings must agree for shared keys: ${key}`);
    if (!(key === '中文' && expected === '中文')) assert.ok(!han.test(translated), `Untranslated key: ${key} → ${translated}`);
    for (const name of names) assert.ok(translated.includes(params[name]), `Lost placeholder ${name}: ${key}`);
  }
  api.setLanguage('zh');
  assert.equal(api.t('缺少 {name}', { name: 'example.ckpt' }), '缺少 example.ckpt');
});

await test('all declared static HTML localization keys have English translations', async () => {
  const { api } = await load();
  const html = fs.readFileSync(path.join(web, 'index.html'), 'utf8');
  assert.match(html, /<html\s+lang="en"/);
  for (const match of html.matchAll(/>([^<>]*)</g)) {
    const text = match[1].trim();
    if (text !== '中文') assert.ok(!han.test(text), `Initial English HTML contains Chinese display text: ${text}`);
  }
  const decode = (s) => s.replaceAll('&quot;', '"').replaceAll('&amp;', '&').replaceAll('&#39;', "'").replaceAll('&lt;', '<').replaceAll('&gt;', '>');
  const keys = [...html.matchAll(/data-i18n(?:-title|-aria-label|-aria-valuetext|-placeholder)?="([^"]+)"/g)].map((match) => decode(match[1]));
  assert.ok(keys.length > 30, 'expected localization bindings in actual HTML');
  for (const key of keys) if (han.test(key) && key !== '中文') assert.ok(!han.test(api.t(key)), `Missing static translation: ${key}`);
});

await test('raw known errors and dynamic notices translate without hiding technical details', async () => {
  const { api } = await load();
  const exact = '模型仍在加载，请稍后重试。';
  assert.ok(!han.test(api.translateMessage(exact)), 'server loading error needs English');
  const dynamic = api.translateMessage('模型加载失败：E_TEST_DISK example.ckpt');
  assert.ok(!han.test(dynamic));
  assert.ok(dynamic.includes('E_TEST_DISK example.ckpt'));
  assert.equal(api.translateMessage('E_UNKNOWN: /private/example.ckpt'), 'E_UNKNOWN: /private/example.ckpt');
  const device = api.t('已连接 {device} · 请保持 Falcon 窗口可见。', { device: '李凯的耳机' });
  assert.ok(device.includes('李凯的耳机'), 'user-provided device names must remain untouched');
  api.setLanguage('zh');
  assert.equal(api.translateMessage(dynamic), '模型加载失败：E_TEST_DISK example.ckpt');
});

await test('switch updates bound text and attributes, preserves device selection and other state', async () => {
  const { api, document, values } = await load();
  const label = document.createElement('span');
  const device = document.createElement('select'); device.value = 'physical-mic-42';
  const control = document.createElement('button');
  const metrics = document.createElement('span');
  const recording = [new Float32Array([.1, .2])];
  const capture = { running: true, generation: 7, recording };
  let changes = 0;
  api.onLanguageChange(() => { changes++; });
  api.setText(label, '开启设备');
  api.setAttr(control, 'title', '停止并关闭设备');
  api.setAttr(control, 'aria-valuetext', '横向 {x}%，纵向 {y}%，大小 {size}%', { x: 25, y: 60, size: 20 });
  api.setText(metrics, '已跳过 {count} 个旧窗口，优先处理最新声音', { count: 7 });
  const englishTitle = control.getAttribute('title');
  assert.ok(!han.test(label.textContent));
  assert.ok(!han.test(englishTitle));
  api.setLanguage('zh');
  assert.equal(label.textContent, '开启设备');
  assert.equal(control.getAttribute('title'), '停止并关闭设备');
  assert.equal(control.getAttribute('aria-valuetext'), '横向 25%，纵向 60%，大小 20%');
  assert.equal(metrics.textContent, '已跳过 7 个旧窗口，优先处理最新声音');
  assert.equal(values.get('falcon.language'), 'zh');
  api.setLanguage('en');
  assert.ok(!han.test(label.textContent));
  assert.equal(control.getAttribute('title'), englishTitle);
  assert.ok(!han.test(control.getAttribute('aria-valuetext')));
  assert.ok(metrics.textContent.includes('7'));
  assert.equal(device.value, 'physical-mic-42');
  assert.equal(capture.recording, recording);
  assert.equal(capture.running, true);
  assert.equal(capture.generation, 7);
  assert.equal(changes, 2);
});

await test('every Chinese literal in the Studio app has an English translation', async () => {
  const { api } = await load();
  const source = fs.readFileSync(path.join(web, 'app.js'), 'utf8');
  const literals = [...source.matchAll(/'((?:[^'\\\n]|\\.)*)'|"((?:[^"\\\n]|\\.)*)"/g)]
    .map((match) => match[1] ?? match[2])
    .filter((text) => han.test(text) && !text.includes('${') && text !== '中文');
  assert.ok(literals.length > 100, `expected Chinese UI literals in app.js, found ${literals.length}`);
  for (const key of new Set(literals)) {
    assert.ok(!han.test(api.t(key)), `app.js literal without an English translation: ${key}`);
  }
});

/* The local model server raises English errors from falcon_runtime.py; the UI shows them
 * through translateMessage(), so every one of them must reverse-map to Chinese. */
function firstRaiseMessage(block) {
  const matches = [...block.matchAll(/"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'/g)];
  if (!matches.length) return null;
  let text = matches[0][1] ?? matches[0][2];
  for (let index = 1; index < matches.length; index++) {
    // Python concatenates adjacent literals ("a" "b"); unrelated literals are not joined.
    const gap = block.slice(matches[index - 1].index + matches[index - 1][0].length, matches[index].index);
    if (!/^\s*$/.test(gap)) break;
    text += matches[index][1] ?? matches[index][2];
  }
  return text;
}

function raiseMessages(file) {
  const source = fs.readFileSync(file, 'utf8');
  const messages = [];
  for (const match of source.matchAll(/raise\s+\w+\s*\(/g)) {
    let index = match.index + match[0].length, depth = 1, quote = null, block = '';
    while (index < source.length && depth > 0) {
      const character = source[index];
      if (quote) {
        block += character;
        if (character === quote && source[index - 1] !== '\\') quote = null;
      } else if (character === '"' || character === "'") { quote = character; block += character; }
      else if (character === '(') { depth++; block += character; }
      else if (character === ')') { depth--; if (depth > 0) block += character; }
      else block += character;
      index++;
    }
    const message = firstRaiseMessage(block);
    if (message) messages.push(message);
  }
  return messages;
}

await test('every English runtime error maps back to Chinese', async () => {
  const { api } = await load();
  const messages = raiseMessages(path.resolve(web, '..', 'falcon_runtime.py'));
  assert.ok(messages.length >= 15, `expected FalconRuntime raise sites, found ${messages.length}`);
  for (const message of messages) {
    if (han.test(message)) continue;
    const translated = api.translateMessage(message, 'zh');
    assert.ok(han.test(translated), `runtime error without a Chinese translation: ${message} → ${translated}`);
  }
  const offline = api.translateMessage('Could not download falcon.pt from Hugging Face repo JusperLee/Falcon: offline. Set HF_ENDPOINT to a mirror, or pass a local weights file.', 'zh');
  assert.ok(han.test(offline) && offline.includes('HF_ENDPOINT'), `download failure must translate and keep the mirror hint: ${offline}`);
  const device = api.translateMessage("device must be 'auto', 'cpu', 'cuda' or 'mps'", 'zh');
  assert.ok(han.test(device) && device.includes('cuda'), `device hint must translate and list cuda: ${device}`);
});

console.log(`${passed} localization checks passed (${dictionaries.length} dictionary entries audited).`);
