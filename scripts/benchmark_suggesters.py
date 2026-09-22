#!/usr/bin/env python3
"""Benchmark the suggestion engines on the frozen corpus splits
(TODO.compare/1): kotoshu vs Hunspell vs SymSpell (+ LanguageTool on a
bounded subsample) at top-1/3/5 exact-match of the human correction.

    python scripts/benchmark_suggesters.py --lang en [--max 2000] [--languagetool]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DICTS = Path("/tmp/bench-dicts")


def load_pairs(lang, klass, max_n):
    data = json.loads((REPO / f"eval/realword/{lang}.suggest-{klass}.json").read_text())
    return data["pairs"][:max_n]


# ---------------------------------------------------------------- engines
def run_hunspell(words, dict_base):
    proc = subprocess.run(
        ["hunspell", "-a", "-d", str(DICTS / dict_base)],
        input="\n".join(words) + "\n", capture_output=True, text=True, timeout=600)
    out = {}
    word_i = 0
    for line in proc.stdout.splitlines():
        if line.startswith("&"):
            head, _, sugs = line.partition(":")
            word = head.split()[1]
            out[word] = [s.strip() for s in sugs.split(",")]
        elif line.startswith("#"):
            out[words[word_i] if word_i < len(words) else "?"] = []
        word_i += 1
    return [out.get(w, []) for w in words]


def _kelly_freq_path(lang):
    """Field lane source: the same published kotoshu/frequency-list-kelly
    list the gem's SymSpell channel indexes (apples-to-apples with the
    kotoshu lane). Falls back to the training ctx table for en/de."""
    ctx = REPO / f"models/{lang}/fasttext.{lang}.ctx.npz"
    if ctx.exists():
        z = np.load(ctx)
        uni = z["unigram_counts"]
        vocab = json.loads((REPO / f"models/{lang}/fasttext.{lang}.vocab.json").read_text())
        vocab = vocab.get("word_to_idx", vocab)
        idx2word = {i: w for w, i in vocab.items()}
        fh = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False)
        for i, count in enumerate(uni):
            if int(count) > 0 and i in idx2word:
                fh.write(f"{idx2word[i]}\t{int(count)}\n")
        fh.close()
        return fh.name

    cache = Path.home() / ".cache/kotoshu/frequency-lists" / lang / "frequency.json"
    if not cache.exists():
        url = f"https://raw.githubusercontent.com/kotoshu/frequency-list-kelly/main/data/{lang}.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "kotoshu-bench/1.0"})
        cache.write_bytes(urllib.request.urlopen(req).read())
    data = json.loads(cache.read_text())
    base_rank = {e["word"]: e["rank"] for e in data.get("full_list", [])}
    fh = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False)
    for e in data.get("full_list", []):
        # wordfreq ranks: synthetic weight = N - rank so relative order holds
        fh.write(f"{e['word']}\t{len(data['full_list']) - e['rank'] + 1}\n")
    fh.close()
    return fh.name


def run_symspell(words, lang):
    from symspellpy import SymSpell, Verbosity
    freq_path = _kelly_freq_path(lang)
    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym.load_dictionary(freq_path, term_index=0, count_index=1, separator="\t", encoding="utf-8")
    results = []
    for w in words:
        got = sym.lookup(w, Verbosity.TOP, 2)
        results.append([s.term for s in got[:8]])
    return results


def run_kotoshu(words, lang="en"):
    env = dict(os.environ, BENCH_LANG=lang)
    proc = subprocess.run(
        ["ruby", "/tmp/bench_kotoshu.rb"],
        input="\n".join(words) + "\n", capture_output=True, text=True,
        cwd=str(Path.home() / "src/kotoshu/kotoshu"), env=env, timeout=10800)
    mapping = {}
    for line in proc.stdout.splitlines():
        try:
            mapping.update(json.loads(line))
        except json.JSONDecodeError:
            pass
    return [mapping.get(w, []) for w in words]


def run_languagetool(words, lang):
    results = []
    for w in words:
        body = json.dumps({"text": w, "language": lang}).encode()
        req = urllib.request.Request(
            "https://api.languagetool.org/v2/check", data=body,
            headers={"Content-Type": "application/json"})
        sugs = []
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                for m in json.loads(resp.read()).get("matches", []):
                    for r in m.get("replacements", [])[:8]:
                        sugs.append(r["value"].strip())
        except Exception:
            pass
        results.append(sugs)
    return results


ENGINES = {"hunspell": run_hunspell, "symspell": run_symspell, "kotoshu": run_kotoshu}


def score(predictions, pairs):
    n = len(pairs)
    top = {1: 0, 3: 0, 5: 0}
    for pred, pair in zip(predictions, pairs):
        target = pair["correction"].lower()
        norm = [p.lower().strip() for p in pred if p.strip()]
        for k in top:
            if target in norm[:k]:
                top[k] += 1
    return {f"top{k}": round(v / n, 4) for k, v in top.items()} | {"n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--split2", action="store_true",
                    help="wave-2 synthetic splits (suggest2-*, per-class tags)")
    ap.add_argument("--languagetool", action="store_true",
                    help="add the public-API engine on a bounded subsample")
    args = ap.parse_args()

    report = {"spec": "kotoshu.suggest-benchmark/v1", "language": args.lang,
              "match": "case-insensitive exact", "engines": {}}
    all_words = {}
    for klass in ("nonword", "realword"):
        if args.split2:
            s2 = REPO / f"eval/realword/{args.lang}.suggest2-{klass}.json"
            payload = json.loads(s2.read_text())
            pairs = payload["pairs"][: args.max]
        else:
            pairs = load_pairs(args.lang, klass, args.max)
        words = [p["typo"] for p in pairs]
        all_words[klass] = (pairs, words)
    if args.split2:
        report["split"] = "wave2-synthetic"

    dict_base = {"en": "en_US", "de": "de_DE_frami", "es": "es_ES", "fr": "fr_FR",
                 "pt": "pt_PT", "ru": "ru_RU", "it": "it_IT", "nl": "nl_NL",
                 "pl": "pl_PL"}.get(args.lang, args.lang)
    for name, fn in ENGINES.items():
        per_class = {}
        for klass, (pairs, words) in all_words.items():
            if name == "hunspell":
                preds = fn(words, dict_base)
            elif name == "symspell":
                preds = fn(words, args.lang)
            else:
                preds = fn(words, args.lang)
            per_class[klass] = score(preds, pairs)
        report["engines"][name] = per_class
        print(f"{name}: {json.dumps(per_class)}", flush=True)

    if args.languagetool:
        sub_pairs = all_words["nonword"][0][:150] + all_words["realword"][0][:150]
        sub_words = [p["typo"] for p in sub_pairs]
        preds = run_languagetool(sub_words, {"en": "en-US"}.get(args.lang, args.lang))
        report["engines"]["languagetool"] = {
            "note": "public API, bounded 300-pair subsample, rate-limit aware",
            "nonword+realword": score(preds, sub_pairs)}
        print(f"languagetool: {report['engines']['languagetool']}", flush=True)

    suffix = "-wave2" if args.split2 else ""
    out = REPO / f"eval/reports/suggest-benchmark-{args.lang}{suffix}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
