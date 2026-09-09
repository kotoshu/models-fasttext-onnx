#!/usr/bin/env python3
"""Plan 114: build and freeze the C-benchmark (repo-clean real + labeled synth).

The bake-off v1 bench evaluated candidate C on the corpus_bench pool minus
train-tainted pairs, which left only 257/2000 English pairs (and 8-13 for
de/ru/es) — too small for a ship decision. This script grows the honest bench
as far as the corpus allows and freezes it as the C-benchmark:

- REAL component: every held-out-repo-clean corpus pair (both words in the
  language's full fastText vocabulary), with NO 2000-pair sampling cap and a
  70/30 repo split (v1 used 80/20; the extra 10% of repos moves en from 1618
  to ~2500 clean pairs while keeping ~38k unique real training pairs).
  Clean rule is v1's strict one, evaluated language-agnostically on the
  (typo, correction) STRING pair: a pair is clean iff it occurs in NO train
  repo under ANY language label (the char encoder memorizes strings, not
  language tags).
- SYNTH component (de/ru/es only; en real already exceeds the 2000 target):
  keyboard-aware typos from eval/noise.py (the repo's own generator, the same
  model the tier gates use), seeded independently of training generation.
  Corrections are drawn uniformly from the 30k most frequent vocab words;
  the typo is rejection-sampled until it is also in the full vocabulary so
  every model (fastText tiers included) is scored under the identical
  corpus_bench rule. A synth bench pair is additionally required to match NO
  training pair of ANY source (real train, synth train, any language label).

Training-side outputs (for scripts/train_typo_biencoder_v2.py) are written
under eval/candidates/p114/ (gitignored): real train pairs with occurrence
counts, synth train pairs for the thin languages, and the clean string-pair
set used as the exclusion rule.

The freeze receipt (sizes, seeds, sha256 per bench file, split counts) is
committed at eval/reports/cbench.frozen.json; the bench pair files under
eval/cbench/ are gitignored and deterministically rebuildable from the
committed corpus fetch + this script.

Determinism: repo split default_rng([42]); synth train default_rng([42,
crc32(lang), 1140]); synth bench default_rng([42, crc32(lang), 1141]).
Everything is a pure function of the vendored corpus + committed vocabs.

Usage:
  python3 scripts/build_cbench.py --repo-root .
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import zlib
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np

SEED = 42
SPLIT_TRAIN_FRAC = 0.7  # v2 split: more held-out repos than v1's 0.8, bigger clean bench
TARGET_LANGS = ("en", "de", "ru", "es")
SYNTH_TRAIN_LANGS = ("de", "ru", "es")  # corpora are thin only for these
SYNTH_BENCH_LANGS = ("de", "ru", "es")  # en real alone exceeds the 2000 target

SYNTH_TRAIN_PER_LANG = 8000
SYNTH_BENCH_PER_LANG = 2000
SYNTH_RANK_WINDOW = 30_000  # corrections drawn from the 30k most frequent words
SYNTH_ATTEMPT_CAP = 50  # per-word rejection attempts, same as run_eval probes

SYNTH_TRAIN_SEED_TAG = 1140
SYNTH_BENCH_SEED_TAG = 1141


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_repo_pairs(corpus_path: Path):
    """Yield (repo, lang, typo, correction) using fetch_corpus extraction rules."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fetch_corpus import LANG_MAP, _word_pair

    with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            repo = obj.get("repo", "")
            for edit in obj.get("edits", []):
                src = edit.get("src") or {}
                lang = LANG_MAP.get(src.get("lang", ""))
                if lang is None:
                    continue
                tgt = edit.get("tgt") or {}
                src_toks = src.get("text", "").split()
                tgt_toks = tgt.get("text", "").split()
                if not src_toks or len(src_toks) != len(tgt_toks):
                    continue
                diffs = [(a, b) for a, b in zip(src_toks, tgt_toks) if a != b]
                if len(diffs) != 1:
                    continue
                pair = _word_pair(*diffs[0])
                if pair is None:
                    continue
                yield repo, lang, pair[0], pair[1]


def synth_pairs(
    rng: np.random.Generator,
    make_typo,
    lang: str,
    words: list[str],
    in_vocab: set[str],
    n_wanted: int,
    exclude: set[tuple[str, str]],
) -> tuple[list[tuple[str, str]], dict]:
    """Rejection-sample (typo, word) pairs with typo and word both in vocabulary.

    ``exclude`` pairs are skipped (training pairs for the bench generator,
    anything training-tainted for the bench itself); returns unique pairs only.
    """
    out: dict[tuple[str, str], None] = {}
    words_tried = 0
    words_resisted = 0
    excluded_hits = 0
    while len(out) < n_wanted and words_tried < n_wanted * 40:
        w = words[int(rng.integers(len(words)))]
        words_tried += 1
        got = None
        for _ in range(SYNTH_ATTEMPT_CAP):
            typo = make_typo(rng, w, lang)
            if typo is not None and typo != w and typo in in_vocab:
                got = typo
                break
        if got is None:
            words_resisted += 1
            continue
        if (got, w) in exclude or (got, w) in out:
            excluded_hits += 1
            continue
        out[(got, w)] = None
    return list(out), {
        "words_tried": words_tried,
        "words_no_in_vocab_typo_after_50_attempts": words_resisted,
        "pairs_skipped_excluded_or_duplicate": excluded_hits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan 114: build and freeze the C-benchmark")
    parser.add_argument("--repo-root", default=".", help="repo root (default: cwd)")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    corpus_path = repo / "eval" / "corpora" / "github-typo-corpus.v1.0.0.jsonl.gz"
    if not corpus_path.exists():
        parser.error(f"{corpus_path} missing — run scripts/fetch_corpus.py first")

    sys.path.insert(0, str(repo / "eval"))
    from noise import make_typo

    print("streaming corpus (repo-grouped pairs, fetch_corpus extraction rules)...")
    pair_repos: dict[tuple[str, str, str], set[str]] = {}
    repo_pairs: dict[str, Counter] = {}
    lang_counter: Counter = Counter()
    for repo_url, lang, typo, corr in stream_repo_pairs(corpus_path):
        key = (lang, typo, corr)
        pair_repos.setdefault(key, set()).add(repo_url)
        repo_pairs.setdefault(repo_url, Counter())[key] += 1
        lang_counter[lang] += 1
    repos = sorted(repo_pairs)
    print(f"repos={len(repos)} pairs_unique={len(pair_repos)} edits_used={sum(lang_counter.values())}")

    rng = np.random.default_rng([SEED])
    perm = rng.permutation(len(repos))
    n_train = int(len(repos) * SPLIT_TRAIN_FRAC)
    train_repos = {repos[i] for i in perm[:n_train]}
    heldout_repos = {repos[i] for i in perm[n_train:]}
    assert not (train_repos & heldout_repos)

    train_counter: Counter = Counter()
    for r in train_repos:
        train_counter.update(repo_pairs[r])
    heldout_clean = {k for k, v in pair_repos.items() if v <= heldout_repos}
    # language-AGNOSTIC string pairs: the char encoder can memorize a string
    # pair regardless of which language the corpus tagged it with (v1 rule)
    clean_str_pairs = {(t, c) for (lang, t, c) in heldout_clean}
    train_str_pairs = {(t, c) for (lang, t, c) in train_counter}
    print(
        f"split {SPLIT_TRAIN_FRAC:g}/{1 - SPLIT_TRAIN_FRAC:g}: {len(train_repos)} train repos / "
        f"{len(heldout_repos)} held-out repos; {len(train_counter)} unique real train pairs; "
        f"{len(heldout_clean)} held-out-clean (lang, pair), {len(clean_str_pairs)} clean string pairs"
    )

    vocabs: dict[str, list[str]] = {}
    vocab_sets: dict[str, set[str]] = {}
    for lang in TARGET_LANGS:
        v = json.loads((repo / "models" / lang / f"fasttext.{lang}.vocab.json").read_text(encoding="utf-8"))
        w2i = v["word_to_idx"]
        words = [None] * len(w2i)
        for w, i in w2i.items():
            words[i] = w
        vocabs[lang] = words
        vocab_sets[lang] = set(w2i)

    # ---- real bench component: ALL clean in-vocab pairs per language, no cap
    bench_real: dict[str, list[tuple[str, str]]] = {}
    for lang in TARGET_LANGS:
        inv = vocab_sets[lang]
        pairs = sorted(
            {(t, c) for (l, t, c) in heldout_clean if l == lang and t in inv and c in inv}
        )
        bench_real[lang] = pairs
        print(f"  real clean in-vocab {lang}: {len(pairs)} pairs")

    # ---- training-side outputs (consumed by train_typo_biencoder_v2.py)
    p114 = repo / "eval" / "candidates" / "p114"
    p114.mkdir(parents=True, exist_ok=True)
    (p114 / "train_pairs.json").write_text(
        json.dumps([[l, t, c, n] for (l, t, c), n in sorted(train_counter.items())], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (p114 / "clean_str_pairs.json").write_text(
        json.dumps(sorted([list(p) for p in clean_str_pairs]), ensure_ascii=False) + "\n", encoding="utf-8"
    )

    synth_train_all: dict[str, list] = {}
    synth_train_meta: dict[str, dict] = {}
    synth_words_window = {lang: vocabs[lang][:SYNTH_RANK_WINDOW] for lang in TARGET_LANGS}
    for lang in SYNTH_TRAIN_LANGS:
        st_rng = np.random.default_rng([SEED, zlib.crc32(lang.encode("utf-8")), SYNTH_TRAIN_SEED_TAG])
        # never train on a pair the real bench will evaluate
        pairs, meta = synth_pairs(
            st_rng, make_typo, lang, synth_words_window[lang], vocab_sets[lang],
            SYNTH_TRAIN_PER_LANG, exclude=clean_str_pairs,
        )
        synth_train_all[lang] = pairs
        synth_train_meta[lang] = meta
        print(f"  synth train {lang}: {len(pairs)} pairs {meta}")
        (p114 / f"synth_train.{lang}.json").write_text(
            json.dumps([list(p) for p in pairs], ensure_ascii=False) + "\n", encoding="utf-8"
        )

    # ---- synth bench component: disjoint from EVERY training source
    bench_synth: dict[str, list[tuple[str, str]]] = {}
    synth_bench_meta: dict[str, dict] = {}
    train_everything: set[tuple[str, str]] = set(train_str_pairs)
    for lang, pairs in synth_train_all.items():
        train_everything.update(pairs)
    for lang in SYNTH_BENCH_LANGS:
        sb_rng = np.random.default_rng([SEED, zlib.crc32(lang.encode("utf-8")), SYNTH_BENCH_SEED_TAG])
        pairs, meta = synth_pairs(
            sb_rng, make_typo, lang, synth_words_window[lang], vocab_sets[lang],
            SYNTH_BENCH_PER_LANG, exclude=train_everything,
        )
        bench_synth[lang] = pairs
        synth_bench_meta[lang] = meta
        print(f"  synth bench {lang}: {len(pairs)} pairs {meta}")

    # ---- freeze
    cbench_dir = repo / "eval" / "cbench"
    cbench_dir.mkdir(parents=True, exist_ok=True)
    frozen = {
        "plan": "114 C-benchmark (frozen)",
        "purpose": (
            "the repo-clean real-typo bench for the typo bi-encoder ship decision; "
            "real = held-out-repo-clean corpus pairs (no sampling cap), synth = "
            "noise.py generator pairs (labeled; generator-domain evidence only)"
        ),
        "split": {
            "rule": (
                "repos shuffled with default_rng([42]), first 70% train, rest held-out; "
                "no repo in both; a bench pair must occur in no train repo under any language label"
            ),
            "n_repos_total": len(repos),
            "n_repos_train": len(train_repos),
            "n_repos_heldout": len(heldout_repos),
            "pairs_unique_total": len(pair_repos),
            "real_train_unique": len(train_counter),
            "real_train_weighted": sum(train_counter.values()),
            "pairs_heldout_clean_lang_tagged": len(heldout_clean),
            "clean_str_pairs": len(clean_str_pairs),
            "edits_used": sum(lang_counter.values()),
            "per_lang_edits": dict(lang_counter.most_common()),
        },
        "synth": {
            "generator": "eval/noise.py make_typo (keyboard-aware, repo tier-gate model)",
            "rank_window": SYNTH_RANK_WINDOW,
            "attempt_cap": SYNTH_ATTEMPT_CAP,
            "train_seed_rule": f"default_rng([{SEED}, crc32(lang), {SYNTH_TRAIN_SEED_TAG}])",
            "bench_seed_rule": f"default_rng([{SEED}, crc32(lang), {SYNTH_BENCH_SEED_TAG}])",
            "train_per_lang_target": {lang: SYNTH_TRAIN_PER_LANG for lang in SYNTH_TRAIN_LANGS},
            "train_actual": {lang: len(synth_train_all[lang]) for lang in SYNTH_TRAIN_LANGS},
            "train_meta": synth_train_meta,
            "bench_per_lang_target": {lang: SYNTH_BENCH_PER_LANG for lang in SYNTH_BENCH_LANGS},
            "bench_actual": {lang: len(bench_synth[lang]) for lang in SYNTH_BENCH_LANGS},
            "bench_meta": synth_bench_meta,
            "bench_exclusion_rule": (
                "a synth bench pair matches no real train pair, no synth train pair "
                "(string-level, any language label) and no held-out-clean pair"
            ),
        },
        "corpus_sha256": file_sha256(corpus_path),
        "determinism": {
            "seed": SEED,
            "rng": "np.random.default_rng with the per-purpose seeds recorded above",
        },
        "languages": {},
        "generated_at": iso_now(),
    }
    for lang in TARGET_LANGS:
        data = {
            "language": lang,
            "real": [list(p) for p in bench_real[lang]],
            "synth": [list(p) for p in bench_synth.get(lang, [])],
            "note": (
                "real pairs occur only in held-out repos (strict repo-level split); "
                "synth pairs are noise.py generator pairs, disjoint from all training data"
            ),
        }
        path = cbench_dir / f"cbench.{lang}.json"
        path.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
        frozen["languages"][lang] = {
            "real_pairs": len(bench_real[lang]),
            "synth_pairs": len(bench_synth.get(lang, [])),
            "file": f"eval/cbench/cbench.{lang}.json",
            "sha256": file_sha256(path),
        }
        print(f"froze {lang}: real={len(bench_real[lang])} synth={len(bench_synth.get(lang, []))}")

    out = repo / "eval" / "reports" / "cbench.frozen.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
