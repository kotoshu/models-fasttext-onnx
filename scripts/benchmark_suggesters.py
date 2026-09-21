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


def run_symspell(words, lang):
    from symspellpy import SymSpell, Verbosity
    z = np.load(REPO / f"models/{lang}/fasttext.{lang}.ctx.npz")
    uni = z["unigram_counts"]
    vocab = json.loads((REPO / f"models/{lang}/fasttext.{lang}.vocab.json").read_text())
    vocab = vocab.get("word_to_idx", vocab)
    idx2word = {i: w for w, i in vocab.items()}
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as fh:
        for i, count in enumerate(uni):
            if int(count) > 0 and i in idx2word:
                fh.write(f"{idx2word[i]}\t{int(count)}\n")
        freq_path = fh.name
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
        cwd=str(Path.home() / "src/kotoshu/kotoshu"), env=env, timeout=3600)
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
    ap.add_argument("--languagetool", action="store_true",
                    help="add the public-API engine on a bounded subsample")
    args = ap.parse_args()

    report = {"spec": "kotoshu.suggest-benchmark/v1", "language": args.lang,
              "match": "case-insensitive exact", "engines": {}}
    all_words = {}
    for klass in ("nonword", "realword"):
        pairs = load_pairs(args.lang, klass, args.max)
        words = [p["typo"] for p in pairs]
        all_words[klass] = (pairs, words)

    dict_base = {"en": "en_US", "de": "de_DE_frami"}.get(args.lang, args.lang)
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

    out = REPO / f"eval/reports/suggest-benchmark-{args.lang}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
