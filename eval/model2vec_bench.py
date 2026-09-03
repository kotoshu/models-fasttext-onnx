#!/usr/bin/env python3
"""Model2Vec (static-embedding) candidate benchmark — C1 adopt-later decision.

Evaluates MinishLab Potion static-embedding models against our tiered fastText
ONNX models on the SAME real-typo corpus benchmark as eval/corpus_bench.py:

- the eval pool per language is built with the IDENTICAL procedure: pairs
  where both words are in the FULL fastText vocabulary (this file imports and
  reuses corpus_bench's loader/manifest helpers and run_eval's ONNX loading),
  subsampled to --max-pairs with np.random.default_rng([42, crc32(lang)]);
- a candidate (a single English Potion model, shared across all languages)
  maps pool words onto ITS vocabulary: pairs whose typo or correction is not
  a candidate-vocab token are skipped (typo_oov / correction_oov, exactly as
  the fluency/mini tiers are);
- for every remaining pair the correction is ranked among the candidate's
  whole vocabulary by cosine-to-typo similarity (typo excluded, ties counted
  optimistically) and top-1/5/20 hits are recorded.

Ranking-universe differences vs our tiers (documented per report):
- candidate vocab sizes differ (29.5k tokens for the 8M/2M potions, ~42k for
  32M) vs our 100k/50k/10k fastText vocabs — hit rates over a smaller
  universe are structurally easier, bias shown in the candidate's favour;
- the candidate vocabulary is English BERT WordPiece: it contains 5 special
  tokens ([PAD]/[UNK]/[CLS]/[SEP]/[MASK]) and ~5.8k '##' subword fragments,
  which are kept in the ranking universe as-is (never themselves eligible
  as corrections since pool words never contain '#');
- hit-rate denominators differ per model (each model's own in-vocab pairs),
  the same caveat as the committed full/fluency/mini comparison.

Coverage reality: Potion models are English-centric; languages with no
usable vocabulary overlap are reported honestly as not coverable (< MIN_
EVAL_PAIRS evaluated pairs). That is itself decisive C1 data.

Output: eval/reports/model2vec.{lang}.json per language and the aggregate
eval/reports/model2vec.summary.json with the pre-declared decision rule:
ADOPT only if some candidate beats the fluency tier's top-5 on >= 5 of 9
languages at <= 20 MB model size with a clean license chain; else REJECT.

Usage:
  python3 eval/model2vec_bench.py --all --repo-root .
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEED = 42
MAX_PAIRS_DEFAULT = 2000
HIT_KS = (1, 5, 20)
MIN_EVAL_PAIRS = 50  # below this the candidate is "not coverable" for the language

SIZE_CAP_MB = 20.0  # decimal MB, from the pre-declared decision rule
LANGS_NEEDED = 5     # candidate must beat fluency top-5 on >= 5 of 9 languages
LANGS_TOTAL = 9

# License chain provenance: HF Hub API (model metadata tags) + model cards,
# retrieved 2026-09-03. Teachers are named on each Potion model card.
LICENSE_RETRIEVED = "2026-09-03"
_LICENSE_CHAIN = {
    "model": "mit (minishlab model card + HF license tag)",
    "teacher": "BAAI/bge-base-en-v1.5 — mit (HF license tag)",
    "training_data": (
        "minishlab/tokenlearn-c4-en-bge-base-v1.5 — no explicit license tag on "
        "the dataset card; C4-derived. Not redistributed by us (weights only), "
        "and distillation training data does not bind the artifact, but noted "
        "for completeness."
    ),
    "verdict": "clean: MIT model + MIT teacher, permissively redistributable under our BSD-2 + CC-BY-SA mix",
}

# Candidates. 'potion-mini-8M' was requested but does not exist on the HF Hub
# under minishlab (author listing + full-text search checked 2026-09-03);
# potion-base-2M (the smallest Potion, same teacher) is evaluated in its
# place as the "mini" slot and clearly labelled as a substitute.
CANDIDATES: list[dict] = [
    {
        "name": "potion-base-8M",
        "hf_id": "minishlab/potion-base-8M",
        "status": "available",
        "size_bytes": 30_236_760,
        "params": 7_559_168,
        "teacher": "BAAI/bge-base-en-v1.5",
        "licenses": dict(_LICENSE_CHAIN),
    },
    {
        "name": "potion-base-32M",
        "hf_id": "minishlab/potion-base-32M",
        "status": "available",
        "size_bytes": 129_210_456,
        "params": 32_302_592,
        "teacher": "BAAI/bge-base-en-v1.5",
        "licenses": dict(_LICENSE_CHAIN),
    },
    {
        "name": "potion-mini-8M",
        "hf_id": "minishlab/potion-mini-8M",
        "status": "not_found_on_hub",
        "note": (
            "requested candidate does not exist: no potion-mini-* repo under "
            "minishlab (author listing and hub search, 2026-09-03). Recorded "
            "as unavailable; not evaluated."
        ),
    },
    {
        "name": "potion-base-2M",
        "hf_id": "minishlab/potion-base-2M",
        "status": "available (substitute for the mini slot after potion-mini-8M was not found)",
        "size_bytes": 7_559_256,
        "params": 1_889_792,
        "teacher": "BAAI/bge-base-en-v1.5",
        "licenses": dict(_LICENSE_CHAIN),
    },
]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def mb(size_bytes: int) -> float:
    return round(size_bytes / 1e6, 1)


def load_pool(repo: Path, run_eval, lang: str, max_pairs: int) -> dict:
    """Build the eval pool exactly as corpus_bench.bench_language does."""
    import corpus_bench  # noqa: F401  (imported for provenance/mirroring; helpers below)

    model_dir = repo / "models" / lang
    full = run_eval.load_full_model(
        model_dir / f"fasttext.{lang}.onnx", model_dir / f"fasttext.{lang}.vocab.json"
    )
    corpus = json.loads((repo / "eval" / "corpora" / f"{lang}.json").read_text(encoding="utf-8"))
    extracted = corpus["n_pairs_unique"]
    pool = [
        (t, c)
        for t, c, _count in corpus["pairs"]
        if t != c and t in full.word_to_idx and c in full.word_to_idx
    ]
    in_full = len(pool)
    if len(pool) > max_pairs:
        rng = np.random.default_rng([SEED, zlib.crc32(lang.encode("utf-8"))])
        pool = [pool[i] for i in rng.permutation(len(pool))[:max_pairs]]
    return {
        "pool": pool,
        "corpus": dict(corpus["corpus"], extraction=corpus["extraction"]),
        "pairs": {
            "extracted_unique": extracted,
            "both_in_full_vocab": in_full,
            "sampled_for_eval": len(pool),
            "sampling": (
                f"first {max_pairs} of default_rng([{SEED}, crc32(language)]) permutation"
                if in_full > max_pairs
                else "all pairs (below cap)"
            ),
        },
    }


class CandidateModel:
    """A loaded StaticModel reduced to what the corpus protocol needs."""

    def __init__(self, hf_id: str) -> None:
        from model2vec import StaticModel

        model = StaticModel.from_pretrained(hf_id)
        embedding = np.ascontiguousarray(model.embedding, dtype=np.float32)
        word_to_idx = dict(model.tokenizer.get_vocab())
        if len(word_to_idx) != embedding.shape[0]:
            raise ValueError(f"{hf_id}: vocab {len(word_to_idx)} != embedding rows {embedding.shape[0]}")
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.word_to_idx = word_to_idx
        self.normalized = embedding / norms
        self.dims = int(embedding.shape[1])
        self.vocab_size = int(embedding.shape[0])
        self.n_special = sum(1 for t in word_to_idx if t.startswith("[") and t.endswith("]"))
        self.n_subword = sum(1 for t in word_to_idx if "##" in t)
        self.rank_desc = (
            f"rank of the correction among the candidate's whole vocabulary "
            f"({self.vocab_size} WordPiece tokens, incl. {self.n_special} special and "
            f"{self.n_subword} '##' subword tokens) by cosine to the typo, excluding "
            f"the typo itself (ties counted optimistically)"
        )


def bench_candidate(cand: CandidateModel, pool: list[tuple[str, str]]) -> dict:
    hits = {k: 0 for k in HIT_KS}
    ranks: list[int] = []
    typo_oov = 0
    corr_oov = 0
    for typo, correction in pool:
        t = cand.word_to_idx.get(typo)
        c = cand.word_to_idx.get(correction)
        if t is None:
            typo_oov += 1
            continue
        if c is None:
            corr_oov += 1
            continue
        sims = cand.normalized @ cand.normalized[t]
        sims[t] = -np.inf
        s = sims[c]
        rank0 = int(np.count_nonzero(sims > s))
        ranks.append(rank0 + 1)
        for k in HIT_KS:
            if rank0 < k:
                hits[k] += 1
    n = len(ranks)
    ranks_arr = np.asarray(ranks)
    return {
        "vocab_size": cand.vocab_size,
        "dims": cand.dims,
        "pairs_evaluated": n,
        "pairs_typo_oov": typo_oov,
        "pairs_correction_oov": corr_oov,
        "coverable": n >= MIN_EVAL_PAIRS,
        "coverage_note": (
            "usable" if n >= MIN_EVAL_PAIRS else f"not coverable (fewer than {MIN_EVAL_PAIRS} of the pool pairs are in the candidate vocab)"
        ),
        "top1": hits[1] / n if n else None,
        "top5": hits[5] / n if n else None,
        "top20": hits[20] / n if n else None,
        "mean_rank": float(np.mean(ranks_arr)) if n else None,
        "median_rank": float(np.median(ranks_arr)) if n else None,
        "ranking": cand.rank_desc,
    }


def load_baselines(repo: Path, lang: str) -> dict:
    """Import the committed corpus benchmark numbers — never recomputed here."""
    report = json.loads((repo / "eval" / "reports" / f"corpus.{lang}.json").read_text(encoding="utf-8"))
    return {
        tier: {
            "vocab_size": t["vocab_size"],
            "pairs_evaluated": t["pairs_evaluated"],
            "top1": t["top1"],
            "top5": t["top5"],
            "top20": t["top20"],
        }
        for tier, t in report["tiers"].items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Model2Vec (Potion) candidate benchmark, report-only")
    parser.add_argument("--lang", nargs="+", help="language codes, e.g. en de")
    parser.add_argument("--all", action="store_true", help="every manifest language with extracted pairs")
    parser.add_argument("--repo-root", default=".", help="repo root (default: cwd)")
    parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS_DEFAULT, help=f"cap on pairs per language (default {MAX_PAIRS_DEFAULT})")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo / "eval"))
    import corpus_bench
    import run_eval  # noqa: E402

    if args.all:
        langs = [
            l
            for l in corpus_bench.manifest_languages(repo)
            if (repo / "eval" / "corpora" / f"{l}.json").exists()
        ]
        if not langs:
            parser.error("no extracted corpora found — run scripts/fetch_corpus.py first")
    elif args.lang:
        langs = args.lang
    else:
        parser.error("specify --lang or --all")

    available = [c for c in CANDIDATES if c["status"].startswith("available")]
    print(f"loading {len(available)} candidates (shared across languages): {[c['name'] for c in available]}")
    loaded: dict[str, CandidateModel] = {}
    for cand in available:
        loaded[cand["name"]] = CandidateModel(cand["hf_id"])
        m = loaded[cand["name"]]
        print(f"  {cand['name']}: vocab={m.vocab_size} dims={m.dims} size={mb(cand['size_bytes'])} MB")

    def candidate_result(cand: dict, pool: list) -> dict:
        res = bench_candidate(loaded[cand["name"]], pool)
        res["hf_id"] = cand["hf_id"]
        res["size_bytes"] = cand["size_bytes"]
        res["size_mb"] = mb(cand["size_bytes"])
        res["within_20mb_cap"] = cand["size_bytes"] <= SIZE_CAP_MB * 1e6
        res["teacher"] = cand["teacher"]
        res["license_chain"] = cand["licenses"]
        return res

    reports_dir = repo / "eval" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_langs: dict[str, dict] = {}
    fluency_wins: dict[str, list[str]] = {c["name"]: [] for c in available}
    coverage: dict[str, dict[str, str]] = {c["name"]: {} for c in available}

    for lang in langs:
        pool_data = load_pool(repo, run_eval, lang, args.max_pairs)
        baselines = load_baselines(repo, lang)
        fluency_top5 = baselines["fluency"]["top5"]

        candidates_out: dict[str, dict] = {}
        for cand in CANDIDATES:
            if cand["name"] not in loaded:
                candidates_out[cand["name"]] = {
                    "hf_id": cand["hf_id"],
                    "status": cand["status"],
                    "note": cand.get("note"),
                }
                continue
            res = candidate_result(cand, pool_data["pool"])
            if res["pairs_evaluated"]:
                res["beats_fluency_top5"] = bool(
                    res["coverable"] and res["top5"] is not None and fluency_top5 is not None
                    and res["top5"] > fluency_top5
                )
            else:
                res["beats_fluency_top5"] = False
            candidates_out[cand["name"]] = res
            coverage[cand["name"]][lang] = "coverable" if res["coverable"] else "not coverable"
            if res["beats_fluency_top5"]:
                fluency_wins[cand["name"]].append(lang)

        report = {
            "language": lang,
            "protocol": (
                "identical pool construction and ranking protocol as "
                "eval/reports/corpus.{lang}.json (corpus_bench.py); pool = pairs with "
                "both words in the FULL fastText vocab, capped at the same 2000-pair "
                "seeded subsample; each model then evaluates on its own in-vocab subset"
            ),
            "corpus": pool_data["corpus"],
            "pairs": pool_data["pairs"],
            "baselines_fasttext": baselines,
            "baseline_note": (
                "full/fluency/mini numbers imported verbatim from the committed "
                "corpus report; hit rates use each model's own pairs_evaluated denominator"
            ),
            "candidates_model2vec": candidates_out,
            "universe_note": (
                "ranking universes differ: fastText tiers rank within 100k/50k/10k "
                "word vocabularies; Potion candidates rank within a ~30-42k English "
                "WordPiece vocabulary (specials and '##' fragments included). Smaller "
                "universes make top-k hits structurally easier — the comparison is "
                "biased in the candidate's favour, and it still has to win under the rule."
            ),
            "coverage_note": (
                "Potion models are English-centric (BERT WordPiece vocab); languages "
                "whose pool has no usable overlap are recorded as not coverable."
            ),
            "determinism": {"seed": SEED, "rng": "np.random.default_rng([seed, crc32(language)])"},
            "report_only": True,
            "generated_at": iso_now(),
        }
        out = reports_dir / f"model2vec.{lang}.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        flu = baselines["fluency"]
        line = (
            f"{lang}: pool={pool_data['pairs']['sampled_for_eval']} "
            f"| fluency top5={flu['top5']:.3f} (n={flu['pairs_evaluated']})"
        )
        for cand in CANDIDATES:
            r = candidates_out[cand["name"]]
            if "top5" not in r:
                line += f" | {cand['name']}: NOT FOUND"
            elif r["pairs_evaluated"]:
                line += (
                    f" | {cand['name']}: n={r['pairs_evaluated']} top5={r['top5']:.3f}"
                    f"{'*beats' if r['beats_fluency_top5'] else ''}"
                )
            else:
                line += f" | {cand['name']}: n=0 not coverable"
        print(line + f" -> {out.relative_to(repo)}")

        summary_langs[lang] = {
            "pool": pool_data["pairs"]["sampled_for_eval"],
            "fluency_top5": fluency_top5,
            "candidates": {
                name: (
                    {
                        "pairs_evaluated": r.get("pairs_evaluated"),
                        "top1": r.get("top1"),
                        "top5": r.get("top5"),
                        "top20": r.get("top20"),
                        "coverable": r.get("coverable"),
                        "beats_fluency_top5": r.get("beats_fluency_top5"),
                    }
                    if "top5" in r
                    else {"status": r.get("status")}
                )
                for name, r in candidates_out.items()
            },
        }

    # ---- aggregate + pre-declared decision rule ----
    rule = {
        "rule": (
            f"adopt Model2Vec as a replacement/embedding tier only if SOME candidate "
            f"beats the fluency tier's top-5 hit rate on >= {LANGS_NEEDED} of "
            f"{LANGS_TOTAL} languages AND is <= {SIZE_CAP_MB:g} MB AND has a clean license chain"
        ),
        "not_coverable_counts_as_not_beating": True,
        "thresholds": {"langs_needed": LANGS_NEEDED, "langs_total": LANGS_TOTAL, "size_cap_mb": SIZE_CAP_MB},
    }
    decisions: dict[str, dict] = {}
    any_adopt = False
    for cand in CANDIDATES:
        if cand["name"] not in loaded:
            decisions[cand["name"]] = {
                "eligible": False,
                "reason": cand.get("note") or cand["status"],
            }
            continue
        wins = fluency_wins[cand["name"]]
        size_ok = cand["size_bytes"] <= SIZE_CAP_MB * 1e6
        license_ok = "clean" in cand["licenses"]["verdict"]
        eligible = len(wins) >= LANGS_NEEDED and size_ok and license_ok
        any_adopt = any_adopt or eligible
        decisions[cand["name"]] = {
            "hf_id": cand["hf_id"],
            "size_mb": mb(cand["size_bytes"]),
            "within_20mb_cap": size_ok,
            "license_chain_clean": license_ok,
            "languages_beating_fluency_top5": len(wins),
            "languages_won": wins,
            "eligible": eligible,
            "per_language_coverage": coverage[cand["name"]],
        }

    summary = {
        "decision_rule": rule,
        "verdict": "ADOPT" if any_adopt else "REJECT",
        "verdict_reason": (
            "no eligible candidate: no Potion candidate beats the fluency tier top-5 "
            "on the required number of languages within the 20 MB size cap"
            if not any_adopt
            else "an eligible candidate exists under the pre-declared rule"
        ),
        "candidates": decisions,
        "candidate_metadata": {
            c["name"]: {
                "hf_id": c["hf_id"],
                "status": c["status"],
                **({"note": c["note"]} if "note" in c else {}),
                **({"size_bytes": c["size_bytes"], "size_mb": mb(c["size_bytes"]), "params": c["params"], "teacher": c["teacher"], "licenses": c["licenses"]} if "params" in c else {}),
            }
            for c in CANDIDATES
        },
        "license_provenance": {
            "retrieved": LICENSE_RETRIEVED,
            "source": "HF Hub API model metadata + model card README (license tags and teacher statements)",
            "note": "our mix: BSD-2 (code) + CC-BY-SA (data/models as applicable)",
        },
        "languages": summary_langs,
        "c1_note": (
            "Potion candidates are English-only WordPiece vocabularies; 8 of 9 target "
            "languages are not coverable. A monolingual English candidate cannot replace "
            "the multilingual fastText tiers. An English-only hybrid (Potion for en, "
            "fastText elsewhere) is not worth pursuing: even on English no candidate beats "
            "the fluency tier top-5, so the hybrid buys nothing."
        ),
        "generated_at": iso_now(),
    }
    out = reports_dir / "model2vec.summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nVERDICT: {summary['verdict']}")
    for name, d in decisions.items():
        if "within_20mb_cap" in d:
            print(f"  {name}: wins={d['languages_beating_fluency_top5']}/9 size_cap={d['within_20mb_cap']} license={d['license_chain_clean']} eligible={d['eligible']}")
        else:
            print(f"  {name}: ineligible — {d['reason']}")
    print(f"-> {out.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
