#!/usr/bin/env python3
"""Plan 123: synthesize verifiable typo corpora from the dictionaries.

The plan-115 hybrid thread was blocked on real de/es typo corpora (the
GitHub corpus yields only 14/20 usable pairs). This generator builds the
synthetic counterpart with an explicit verification system — the
admission rule of the (problem, correction, verification) formalism:

  a pair (typo, correction) is admitted iff
    1. the correction is a base form in the language's staged hunspell
       dictionary (pure letters, 3..14 chars, no affix flags needed),
    2. the correction also sits in the model's frequency-ranked
       vocabulary (sampled from the dictionary∩vocab intersection in
       rank order) so the pairs measure ranking quality, not the
       vocabulary cut — the lesson the plan-114 builder learned when
       77% of raw dictionary stems landed outside the 100k vocab,
    3. the typo is NOT a word of that dictionary (it must be wrong),
    4. the typo is reachable from the correction by the declared noise
       operation mix (eval/noise.py — the same keyboard-aware generator
       the tier gates use). The typo's own vocabulary membership is
       deliberately NOT constrained: OOV typos are the hybrid thread's
       reason to exist (the char bi-encoder embeds them), and the tier
       bench counts them as typo-OOV by its existing discipline.

Difficulty calibration is declared, not emergent: ~70% single-op and
~30% two-op corruptions; per-pair provenance (ops, edit distance) is
aggregated into histograms in the file's metadata. Everything is a pure
function of (dictionary pin, seed), so the checked-in corpus is
byte-rebuildable — tests assert exactly that.

Output schema mirrors eval/corpora/{lang}.json (pairs are
[typo, correction, count] triples with count 1) so every corpus
consumer works unchanged; provenance lives at the file level.

Usage:
  python3 eval/synthesize_corpora.py --repo-root . --lang de es \
      --dictionaries-root ../dictionaries --pairs 5000

Deterministic: seed = [42, crc32(lang), 1230].
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import zlib
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np

SEED = [42, 1230]
SINGLE_OP_FRACTION = 0.70
MIN_WORD_LEN = 3
MAX_WORD_LEN = 14
CORRECTIONS_SAMPLE = 40_000
ATTEMPTS_PER_WORD = 4


def _is_pure_letters(word: str) -> bool:
    """Letters only, no marks/digits/punct — the noise ops need clean
    strings and the corpora must stay script-homogeneous."""
    if not word.isalpha():
        return False
    return all(unicodedata.category(ch).startswith("L") for ch in word)


def load_dictionary_words(dict_root: Path, lang: str) -> tuple[set[str], str]:
    """Base-form word list from the staged hunspell dictionary.

    Both upstream layouts are honored ({lang}/spelling/index.dic and
    the flat {lang}/index.dic). Comment lines (leading tab) and the
    count header are skipped; affix flags after ``/`` are dropped —
    only un-flagged stems qualify as corrections (a correction must be
    a standalone valid word).
    """
    for candidate in (dict_root / lang / "spelling" / "index.dic", dict_root / lang / "index.dic"):
        if candidate.exists():
            path = candidate
            break
    else:
        raise SystemExit(f"no index.dic for {lang} under {dict_root}")

    words: set[str] = set()
    raw = path.read_bytes()
    for line in raw.decode("utf-8", errors="strict").splitlines():
        if not line or line.startswith("\t") or line.startswith("#"):
            continue
        stem = line.split("/", 1)[0]
        if stem.startswith("\t") or not stem:
            continue
        if MIN_WORD_LEN <= len(stem) <= MAX_WORD_LEN and stem == stem.lower() and _is_pure_letters(stem):
            words.add(stem)
    if len(words) < 1000:
        raise SystemExit(f"{path}: only {len(words)} usable stems — refusing to build a corpus")
    return words, sha256(raw).hexdigest()


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def load_vocab_ranks(repo_root: Path, lang: str) -> dict[str, int]:
    """The full model's vocabulary in frequency-rank order (index 0 =
    most frequent). The corpus samples corrections from the top of the
    dictionary∩vocab intersection so pairs measure ranking, not the
    vocabulary cut."""
    vocab = json.loads((repo_root / "models" / lang / f"fasttext.{lang}.vocab.json").read_text(encoding="utf-8"))
    return vocab["word_to_idx"]


def synthesize(lang: str, pairs_target: int, dict_root: Path, repo_root: Path) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from noise import make_typo_traced

    dictionary, dict_sha = load_dictionary_words(dict_root, lang)
    vocab = load_vocab_ranks(repo_root, lang)
    # rank-ordered dictionary-valid stems: frequent words first
    candidates = sorted((w for w in dictionary if w in vocab), key=lambda w: vocab[w])
    if len(candidates) < 5000:
        raise SystemExit(f"{lang}: dictionary∩vocab intersection is only {len(candidates)} — cannot build a corpus")
    rng = np.random.default_rng([*SEED, zlib.crc32(lang.encode("utf-8"))])

    sampled = rng.choice(candidates, size=min(CORRECTIONS_SAMPLE, len(candidates)), replace=False)

    pairs: list[list] = []
    ops_hist: Counter[str] = Counter()
    dist_hist: Counter[int] = Counter()
    seen: set[tuple[str, str]] = set()

    for word in sampled:
        if len(pairs) >= pairs_target:
            break
        two_ops = rng.random() > SINGLE_OP_FRACTION
        for _ in range(ATTEMPTS_PER_WORD):
            traced = make_typo_traced(rng, str(word), lang)
            if traced is None:
                continue
            typo, op = traced
            applied = [op]
            if two_ops:
                traced2 = make_typo_traced(rng, typo, lang)
                if traced2 is None or traced2[0] in dictionary:
                    continue
                typo, op2 = traced2
                applied.append(op2)
            if typo == word or typo in dictionary:
                continue
            if not _is_pure_letters(typo) or len(typo) > MAX_WORD_LEN + 1:
                continue
            key = (typo, str(word))
            if key in seen:
                continue
            seen.add(key)
            distance = _levenshtein(typo, str(word))
            if distance == 0 or distance > 2:
                continue
            pairs.append([typo, str(word), 1])
            ops_hist["+".join(applied)] += 1
            dist_hist[distance] += 1
            break

    pairs.sort(key=lambda p: (p[1], p[0]))
    return {
        "language": lang,
        "corpus": {
            "name": "Synthetic keyboard-noise typo corpus",
            "version": "1.0.0",
            "synthetic": True,
            "generator": "eval/synthesize_corpora.py",
            "generator_seed": SEED + [zlib.crc32(lang.encode("utf-8"))],
            "noise_model": "eval/noise.py make_typo_traced (the tier-gate generator)",
            "dictionary": {"path_layout": "{lang}/spelling/index.dic", "sha256": dict_sha},
            "vocab_cut": {"source": f"models/{lang}/fasttext.{lang}.vocab.json",
                          "rule": "corrections sampled from the rank-ordered dictionary-vocab intersection",
                          "intersection_size": len(candidates)},
            "admission_rule": "correction is an un-flagged pure-letter dictionary stem inside the "
            "model vocabulary; typo is not a dictionary word and is reachable by the declared noise ops",
            "caveat": "generator-domain evidence: the plan-114 bi-encoder trained on the same "
            "noise model, so this corpus never replaces the frozen real components",
            "license": "generated data; dictionary licenses per the dictionaries repository",
        },
        "extraction": {
            "rule": "40k sample from the rank-ordered dictionary-vocab intersection; "
            "70% single-op / 30% two-op corruptions; dedup by (typo, correction); "
            "distance capped at 2",
            "single_op_fraction": SINGLE_OP_FRACTION,
        },
        "extraction_stats": {
            "ops_histogram": dict(sorted(ops_hist.items())),
            "distance_histogram": {str(k): v for k, v in sorted(dist_hist.items())},
        },
        "n_pairs_unique": len(pairs),
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan 123 synthetic typo corpora")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--lang", nargs="+", default=["de", "es"])
    parser.add_argument("--dictionaries-root", required=True,
                        help="checkout of kotoshu/dictionaries (the staged pin)")
    parser.add_argument("--pairs", type=int, default=5000)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    dict_root = Path(args.dictionaries_root).resolve()
    out_dir = repo / "eval" / "corpora" / "synth"
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang in args.lang:
        corpus = synthesize(lang, args.pairs, dict_root, repo)
        out = out_dir / f"{lang}.json"
        out.write_text(json.dumps(corpus, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{lang}: {corpus['n_pairs_unique']} pairs -> {out}")
        print(f"  ops: {corpus['extraction_stats']['ops_histogram']}")
        print(f"  distance: {corpus['extraction_stats']['distance_histogram']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
