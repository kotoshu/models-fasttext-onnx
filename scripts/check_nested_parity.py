#!/usr/bin/env python3
"""Check nested-artifact views against the shipped tiers — plan 71.

Two levels of parity, both of which must be EXACT (any delta is a bug,
never noise — the shipped tiers were built from the same matrix rows
with the same per-row int8 quantization):

1. Array parity: the first-N row-prefix view of the nested artifact's
   q_embeddings + row_scale must be array-equal to the shipped separate
   tier file's arrays.
2. Metric parity: the view, materialized as a tier .onnx via
   build_tiers.make_tier_model and loaded through run_eval.load_tier_model
   (no monkeypatching — the real loader, incl. its onnxruntime spot
   check), is evaluated with the shipped gate metrics and the numbers
   must equal the committed eval/reports/{lang}.{tier}.json values
   exactly (same seeded RNG stream).

Writes eval/reports/nested.parity.json and exits nonzero on any
mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tiers  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import run_eval  # noqa: E402


def truncations_from_tiers_json(repo: Path, lang: str) -> dict[str, int]:
    tiers = json.loads((repo / "models" / lang / "tiers.json").read_text(encoding="utf-8"))
    return {tier: int(entry["vocab_size"]) for tier, entry in sorted(tiers["tiers"].items())}


def arrays(model: onnx.ModelProto) -> tuple[np.ndarray, np.ndarray]:
    q = build_tiers.constant_array(model, "q_embeddings")
    scale = build_tiers.constant_array(model, "row_scale")
    if q.dtype != np.int8 or scale.dtype != np.float32:
        raise ValueError("nested artifact: expected int8 q_embeddings / fp32 row_scale")
    return q, scale


def exact_metric_parity(repo: Path, lang: str, tier: str, n: int, view_q: np.ndarray, view_scale: np.ndarray) -> dict:
    """Evaluate the view exactly as run_eval.evaluate() would and compare."""
    model_dir = repo / "models" / lang
    full = run_eval.load_full_model(
        model_dir / f"fasttext.{lang}.onnx", model_dir / f"fasttext.{lang}.vocab.json"
    )

    with tempfile.TemporaryDirectory(prefix=f"nested-parity-{lang}-{tier}-") as tmp:
        view_path = Path(tmp) / f"fasttext.{lang}.{tier}.onnx"
        onnx.save(build_tiers.make_tier_model(np.ascontiguousarray(view_q), np.ascontiguousarray(view_scale), tier), str(view_path))
        # real loader path: consistency checks + onnxruntime spot check
        lm = run_eval.load_tier_model(view_path, model_dir / f"fasttext.{lang}.{tier}.vocab.json")

    rng = run_eval.make_rng(lang)
    metrics = {
        "rank_corr": run_eval.rank_corr_metric(full, lm, rng),
        "top1_agreement": run_eval.top1_agreement_metric(full, lm, rng, lang),
        "coverage": run_eval.coverage_metric(full.vocab_size, lm.vocab_size),
    }

    committed = json.loads((repo / "eval" / "reports" / f"{lang}.{tier}.json").read_text(encoding="utf-8"))
    mismatches = []
    for name, got in metrics.items():
        want = committed["metrics"][name]
        if got != want:
            for key in want:
                if got.get(key) != want[key]:
                    mismatches.append(f"{name}.{key}: view={got.get(key)!r} committed={want[key]!r}")

    return {
        "metrics": metrics,
        "mismatches": mismatches,
        "exact": not mismatches,
        "rank_corr_mean_spearman": metrics["rank_corr"]["mean_spearman"],
        "top1_agreement": metrics["top1_agreement"]["agreement"],
    }


def check_language(repo: Path, lang: str) -> dict:
    model_dir = repo / "models" / lang
    nested = onnx.load(str(model_dir / f"fasttext.{lang}.nested.onnx"))
    nested_meta = run_eval.metadata_dict(nested)
    nq, ns = arrays(nested)

    truncations = truncations_from_tiers_json(repo, lang)
    meta_truncations = json.loads(nested_meta.get("truncations", "{}"))
    if meta_truncations != truncations:
        raise ValueError(f"{lang}: metadata truncations {meta_truncations} != tiers.json {truncations}")
    if int(nested_meta["vocabulary_size"]) != nq.shape[0]:
        raise ValueError(f"{lang}: metadata vocabulary_size != {nq.shape[0]}")

    result: dict = {"truncations": truncations, "tiers": {}}
    for tier, n in truncations.items():
        shipped = onnx.load(str(model_dir / f"fasttext.{lang}.{tier}.onnx"))
        sq, ss = arrays(shipped)
        q_equal = bool(np.array_equal(nq[:n], sq))
        scale_equal = bool(np.array_equal(ns[:n], ss))
        dequant_equal = bool(
            np.array_equal(nq[:n].astype(np.float32) * ns[:n, None], sq.astype(np.float32) * ss[:, None])
        )

        entry = {
            "vocab_size": n,
            "array_parity": {"q_embeddings": q_equal, "row_scale": scale_equal, "dequantized": dequant_equal},
        }
        if q_equal and scale_equal:
            metric = exact_metric_parity(repo, lang, tier, n, nq[:n], ns[:n])
            entry["metric_parity"] = {
                "exact": metric["exact"],
                "mismatches": metric["mismatches"],
                "rank_corr_mean_spearman": metric["rank_corr_mean_spearman"],
                "top1_agreement": metric["top1_agreement"],
            }
            entry["passed"] = metric["exact"]
        else:
            entry["metric_parity"] = {"exact": False, "mismatches": ["array parity failed; metrics not evaluated"]}
            entry["passed"] = False
        result["tiers"][tier] = entry
        print(
            f"  {lang}/{tier} (n={n}): arrays q={q_equal} scale={scale_equal} dequant={dequant_equal}, "
            f"metrics exact={entry['metric_parity']['exact']} "
            f"(rank_corr={entry['metric_parity'].get('rank_corr_mean_spearman')!r}, "
            f"top1={entry['metric_parity'].get('top1_agreement')!r})"
        )
    result["passed"] = all(t["passed"] for t in result["tiers"].values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify nested views match shipped tiers (arrays + metrics), plan 71")
    parser.add_argument("--lang", nargs="+", help="language codes, e.g. en de")
    parser.add_argument("--all", action="store_true", help="check every language in manifest.json")
    parser.add_argument("--repo-root", default=".", help="repo root (default: cwd)")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    known = build_tiers.manifest_languages(repo)
    if args.all:
        langs = known
    elif args.lang:
        unknown = [l for l in args.lang if l not in known]
        if unknown:
            parser.error(f"unknown languages {unknown}; manifest has {known}")
        langs = args.lang
    else:
        parser.error("specify --lang or --all")

    results: dict[str, dict] = {}
    failures: list[str] = []
    for lang in langs:
        print(f"[{lang}] checking nested views")
        results[lang] = check_language(repo, lang)
        if not results[lang]["passed"]:
            failures.append(lang)

    report = {
        "languages": langs,
        "all_exact": not failures,
        "failed_languages": failures,
        "results": results,
        "determinism": {"seed": run_eval.SEED, "rng": "np.random.default_rng([seed, crc32(language)])"},
        "generated_at": build_tiers.iso_now(),
    }
    out = repo / "eval" / "reports" / "nested.parity.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")

    if failures:
        print(f"PARITY FAILURES: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"all {len(langs)} languages x 2 tiers: EXACT parity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
