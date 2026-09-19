/* Real-word detection demo — serves everything from the public artifact
 * chain: registry.json → raw-host mirrors (mini onnx + vocab, full vocab,
 * ctx bigram tables) + the pruned confusion table shipped beside this page.
 * Zero LFS, zero server. See demos/realword and TODO.deploy/4. */
"use strict";

const RAW = "https://raw.githubusercontent.com/kotoshu/models-fasttext-onnx/main";
const LANG = "en";
const DIR = `${RAW}/models/${LANG}`;
const WORD_RE = /[A-Za-z']+/g;

const $ = (id) => document.getElementById(id);
const specs = (line) => {
  const li = document.createElement("li");
  li.textContent = line;
  $("load-specs").appendChild(li);
};
const progress = (frac) => { $("bar-fill").style.width = `${Math.round(frac * 100)}%`; };

async function fetchProgress(url, label) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${label}: HTTP ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) progress(received / total);
    $("load-msg").textContent = `${label}: ${(received / 1e6).toFixed(1)} MB` +
      (total ? ` of ${(total / 1e6).toFixed(1)} MB` : "");
  }
  const out = new Uint8Array(received);
  let at = 0;
  for (const c of chunks) { out.set(c, at); at += c.length; }
  return out;
}

/* npz = uncompressed zip of .npy members. fflate unwraps the zip; we parse
 * the minimal npy v1.0 header (<u4, C order) ourselves. */
function parseNpy(bytes) {
  const magic = String.fromCharCode(...bytes.subarray(0, 6));
  if (!magic.startsWith("\x93NUMPY")) throw new Error("not a npy member");
  const headerLen = bytes[8] | (bytes[9] << 8);
  const header = new TextDecoder().decode(bytes.subarray(10, 10 + headerLen));
  const shapeMatch = header.match(/'shape': \(([^)]*)\)/);
  const n = parseInt(shapeMatch[1].trim().replace(/,$/, ""), 10);
  const dataStart = 10 + headerLen;
  const out = new Uint32Array(n);
  for (let i = 0; i < n; i += 1) {
    const at = bytes.byteOffset + dataStart + i * 4;
    out[i] = bytes[at] | (bytes[at + 1] << 8) | (bytes[at + 2] << 16) | (bytes[at + 3] << 24);
  }
  return out;
}

function parseNpz(bytes) {
  const files = fflate.unzipSync(bytes);
  const out = {};
  for (const name of Object.keys(files)) {
    out[name.replace(/\.npy$/, "")] = parseNpy(files[name]);
  }
  return out;
}

/* blakejs blake2b over "a|b", 4 bytes, big-endian uint32 — identical to
 * scripts/train_ctx_lm.py hash_pair. */
function hashPair(a, b) {
  const digest = blake.blake2b(new TextEncoder().encode(`${a}|${b}`), null, 4);
  return ((digest[0] << 24) | (digest[1] << 16) | (digest[2] << 8) | digest[3]) >>> 0;
}

/* Sorted-key binary search: the artifact stores keys ascending. */
class BigramTable {
  constructor(keys, counts) { this.keys = keys; this.counts = counts; }
  get(a, b) {
    const key = hashPair(a, b);
    let lo = 0;
    let hi = this.keys.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const k = this.keys[mid];
      if (k === key) return this.counts[mid];
      if (k < key) lo = mid + 1; else hi = mid - 1;
    }
    return 0;
  }
}

const state = { ready: false, embCache: new Map() };

async function load() {
  const t0 = performance.now();
  specs(`registry: ${RAW}/registry.json`);
  const registry = JSON.parse(new TextDecoder().decode(
    await fetchProgress(`${RAW}/registry.json`, "registry")));
  const entry = registry.resources[`kotoshu://models/${LANG}/mini`];
  if (!entry) throw new Error("no en mini entry in registry");
  specs(`en mini mirror: ${entry.urls.mirror}`);

  const confusion = JSON.parse(new TextDecoder().decode(
    await fetchProgress("confusion-en.json", "confusion table"))).table;
  specs(`confusion table: ${Object.keys(confusion).length} words`);

  const fullVocab = JSON.parse(new TextDecoder().decode(
    await fetchProgress(`${DIR}/fasttext.${LANG}.vocab.json`, "full vocab")));
  state.w2i = fullVocab.word_to_idx || fullVocab;
  specs(`full vocab: ${Object.keys(state.w2i).length} ids (bigram space)`);

  const npz = parseNpz(await fetchProgress(`${DIR}/fasttext.${LANG}.ctx.npz`, "bigram tables"));
  state.table = new BigramTable(npz.bigram_keys, npz.bigram_counts);
  specs(`bigrams: ${npz.bigram_keys.length.toLocaleString()} types, binary search`);

  const miniBytes = await fetchProgress(`${DIR}/fasttext.${LANG}.mini.onnx`, "mini model");
  state.session = await ort.InferenceSession.create(miniBytes.buffer,
    { executionProviders: ["wasm"] });
  const miniVocabBytes = await fetchProgress(`${DIR}/fasttext.${LANG}.mini.vocab.json`, "mini vocab");
  const parsed = JSON.parse(new TextDecoder().decode(miniVocabBytes));
  state.miniVocab = parsed.word_to_idx || parsed;
  specs(`mini model: onnxruntime-web wasm, ${Object.keys(state.miniVocab).length} words`);

  progress(1);
  $("load-msg").textContent = `ready in ${((performance.now() - t0) / 1000).toFixed(1)} s`;
  state.ready = true;
  state.registry = registry;
  state.confusion = confusion;
  populateLanguages(registry);
  $("loader").classList.add("hidden");
  $("demo").classList.remove("hidden");
  $("explorer").classList.remove("hidden");
  $("check").disabled = false;
}

/* Language explorer: every language's mini model through its registry
 * mirror, nearest neighbours over a deterministic vocabulary sample. */
const langSessions = new Map();

function populateLanguages(registry) {
  const select = $("lang");
  const langs = Object.keys(registry.resources)
    .filter((id) => id.startsWith("kotoshu://models/") && id.endsWith("/mini"))
    .map((id) => id.split("/")[3])
    .sort();
  for (const lang of langs) {
    const option = document.createElement("option");
    option.value = lang;
    option.textContent = lang;
    select.appendChild(option);
  }
  select.value = "en";
}

async function sessionFor(lang) {
  if (langSessions.has(lang)) return langSessions.get(lang);
  const entry = state.registry.resources[`kotoshu://models/${lang}/mini`];
  const onnxBytes = await fetchProgress(entry.urls.mirror, `${lang} mini`);
  const session = await ort.InferenceSession.create(onnxBytes.buffer,
    { executionProviders: ["wasm"] });
  const vocabRaw = await fetchProgress(
    entry.urls.mirror.replace(/[^/]+$/, `fasttext.${lang}.mini.vocab.json`),
    `${lang} vocab`);
  const vocab = JSON.parse(new TextDecoder().decode(vocabRaw));
  const wordToIdx = vocab.word_to_idx || vocab;
  const idxToWord = Object.keys(wordToIdx);
  // deterministic sample: every k-th entry up to 400
  const step = Math.max(1, Math.floor(idxToWord.length / 400));
  const sample = [];
  for (let i = 0; i < idxToWord.length && sample.length < 400; i += step) sample.push(i);
  const rec = { session, wordToIdx, idxToWord, sample, cache: new Map() };
  langSessions.set(lang, rec);
  return rec;
}

async function explorerVec(rec, idx) {
  if (!rec.cache.has(idx)) {
    const input = new ort.Tensor("int64", BigInt64Array.from([BigInt(idx)]), [1]);
    const v = await rec.session.run({ word_index: input }).then((o) => {
      const d = o.embedding.data;
      let norm = 0;
      for (let i = 0; i < d.length; i += 1) norm += d[i] * d[i];
      norm = Math.sqrt(norm) || 1;
      const out = new Float32Array(d.length);
      for (let i = 0; i < d.length; i += 1) out[i] = d[i] / norm;
      return out;
    });
    rec.cache.set(idx, v);
  }
  return rec.cache.get(idx);
}

$("neighbors").addEventListener("click", async () => {
  const lang = $("lang").value;
  const word = $("word").value.trim();
  const out = $("word-result");
  if (!word) { out.textContent = "Type a word first."; return; }
  $("neighbors").disabled = true;
  out.textContent = `loading ${lang}…`;
  try {
    const t0 = performance.now();
    const rec = await sessionFor(lang);
    const idx = rec.wordToIdx[word];
    if (idx === undefined) {
      out.textContent = `"${word}" is not in the ${lang} mini vocabulary (10k most frequent words). Try another.`;
      return;
    }
    const target = await explorerVec(rec, idx);
    const scored = [];
    for (const i of rec.sample) {
      if (i === idx) continue;
      const v = await explorerVec(rec, i);
      let s = 0;
      for (let k = 0; k < v.length; k += 1) s += v[k] * target[k];
      scored.push([s, rec.idxToWord[i]]);
    }
    scored.sort((a, b) => b[0] - a[0]);
    const ms = (performance.now() - t0).toFixed(0);
    out.innerHTML = `<strong>${lang}</strong> · nearest to <em>${escapeHtml(word)}</em>:` +
      `<ol>${scored.slice(0, 5).map(([s, w]) =>
        `<li>${escapeHtml(w)} <span class="muted">${s.toFixed(3)}</span></li>`).join("")}</ol>` +
      `<p class="verdict">sampled ${rec.sample.length} of ${rec.idxToWord.length} vocabulary entries · ${ms} ms</p>`;
  } catch (err) {
    out.textContent = `error: ${err.message}`;
  }
  $("neighbors").disabled = false;
});

function embedding(word) {
  const cache = state.embCache;
  if (!cache.has(word)) {
    const idx = state.miniVocab[word];
    const input = new ort.Tensor("int64", BigInt64Array.from([BigInt(idx)]), [1]);
    cache.set(word, state.session.run({ word_index: input }).then((o) => {
      const v = o.embedding.data;
      let norm = 0;
      for (let i = 0; i < v.length; i += 1) norm += v[i] * v[i];
      norm = Math.sqrt(norm) || 1;
      const out = new Float32Array(v.length);
      for (let i = 0; i < v.length; i += 1) out[i] = v[i] / norm;
      return out;
    }));
  }
  return cache.get(word);
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) s += a[i] * b[i];
  return s;
}

async function similarity(context, x) {
  const ex = await embedding(x);
  let acc = 0;
  let wsum = 0;
  for (const { word, weight } of context) {
    acc += weight * dot(ex, await embedding(word));
    wsum += weight;
  }
  return acc / wsum;
}

async function detect(text, tauB) {
  const { confusion, w2i, table, miniVocab } = state;
  const toks = text.match(WORD_RE) || [];
  const lower = toks.map((t) => t.toLowerCase());
  const ids = lower.map((t) =>
    Object.prototype.hasOwnProperty.call(w2i, t) ? w2i[t] : -1);
  const flags = [];
  for (let i = 0; i < lower.length; i += 1) {
    const w = lower[i];
    if (!Object.prototype.hasOwnProperty.call(confusion, w) ||
        !Object.prototype.hasOwnProperty.call(miniVocab, w)) continue;
    const candidates = confusion[w].filter((c) => c !== w &&
      Object.prototype.hasOwnProperty.call(w2i, c) &&
      Object.prototype.hasOwnProperty.call(miniVocab, c));
    let prev = null;
    for (let j = i - 1; j >= 0 && prev === null; j -= 1) if (ids[j] >= 0) prev = ids[j];
    let next = null;
    for (let j = i + 1; j < ids.length && next === null; j += 1) if (ids[j] >= 0) next = ids[j];

    const context = [];
    for (let j = 0; j < lower.length; j += 1) {
      if (Math.abs(i - j) > 4 || j === i) continue;
      const t = lower[j];
      if (!Object.prototype.hasOwnProperty.call(miniVocab, t)) continue;
      if (t === w || candidates.includes(t)) continue;
      context.push({ word: t, weight: Math.log2(2 + miniVocab[t]) });
    }
    context.sort((a, b) => b.weight - a.weight);
    if (context.length === 0) continue;
    const top = context.slice(0, 3);

    const semBase = await similarity(top, w);
    let best = null;
    for (const c of candidates) {
      const ds = (await similarity(top, c)) - semBase;
      if (ds <= 0.15) continue;
      const wi = w2i[w];
      const ci = w2i[c];
      const left = prev !== null
        ? Math.log((table.get(prev, ci) + 1) / (table.get(prev, wi) + 1)) : 0;
      const right = next !== null
        ? Math.log((table.get(ci, next) + 1) / (table.get(wi, next) + 1)) : 0;
      const ratio = left + right;
      if (ratio > tauB && (!best || ratio > best.ratio)) {
        best = { suggestion: c, sem: ds, ratio };
      }
    }
    if (best) flags.push({ index: i, word: w, ...best });
  }
  return { flags, tokens: toks, lower };
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
}

function render(text, detection) {
  const { flags } = detection;
  const byIndex = new Map(flags.map((f) => [f.index, f]));
  let html = "";
  let last = 0;
  let tokenIndex = 0;
  const re = new RegExp(WORD_RE.source, "g");
  let match;
  while ((match = re.exec(text)) !== null) {
    const flag = byIndex.get(tokenIndex);
    tokenIndex += 1;
    if (flag) {
      html += escapeHtml(text.slice(last, match.index));
      html += `<mark title="${escapeHtml(flag.word)} → ${escapeHtml(flag.suggestion)} (semantic +${flag.sem.toFixed(2)}, bigram ×${flag.ratio.toFixed(1)})">${escapeHtml(match[0])}</mark>`;
      last = match.index + match[0].length;
    } else {
      last = match.index + match[0].length;
    }
  }
  html += escapeHtml(text.slice(last));
  $("result").innerHTML = html +
    (flags.length
      ? `<p class="verdict">${flags.length} real-word error(s) — hover the highlights.</p>`
      : `<p class="verdict ok">No real-word errors found at this strictness.</p>`);
}

$("tau").addEventListener("input", (e) => { $("tau-val").textContent = Number(e.target.value).toFixed(1); });
$("check").addEventListener("click", async () => {
  $("check").disabled = true;
  const t0 = performance.now();
  try {
    const detection = await detect($("input").value, Number($("tau").value));
    render($("input").value, detection);
    const ms = (performance.now() - t0).toFixed(0);
    $("telemetry").textContent =
      `tokens=${detection.tokens.length} flags=${detection.flags.length} ` +
      `latency=${ms} ms strictness=${Number($("tau").value).toFixed(1)} ` +
      `scorer=bigram-ratio ∧ semantic-agreement (uncalibrated research preview)`;
  } catch (err) {
    $("telemetry").textContent = `error: ${err.message}`;
  }
  $("check").disabled = false;
});

load().catch((err) => {
  $("load-msg").textContent = `failed: ${err.message}`;
  throw err;
});
