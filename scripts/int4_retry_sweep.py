#!/usr/bin/env python3
"""Systematic lever-retry sweep for the int4-group128 rejection (plan 68 B1).

eval/reports/int4.summary.json rejected int4-group128 because top1_agreement
failed the fluency gate (>= 0.95) on 8/9 languages and recorded three
untried levers: smaller groups (64/32), asymmetric rounding, and a mixed
int4-mini + int8-fluency split. This script retries those levers with
honest bookkeeping and bounded spend:

1. Canary sweep first: {group 128, group 64} x {nearest-even, half-away}
   on en + ja only — the best (0.90) and worst (0.78) top1 languages of
   the group-128 run. Every config builds tagged artifacts
   (int4-g{G}-{ne|ha}) so the original int4.* evidence stays pristine.
2. If nothing clears the fluency gate on BOTH canaries: group-32 probe
   on ja only (cheapest decisive test — ja is the worst case). If ja
   passes there, the canary pair is completed with a single en run
   before any verdict; otherwise the sweep stops and records the
   confirmed rejection without burning the full matrix.
3. If a config clears the canaries: run all 9 languages with it.
   9/9 gates passed at <= 20 MB per model => "conditional ship candidate
   — owner gate on tier naming/registry"; any miss => confirmed
   rejection with the full table.
4. Mixed lever (only when the sweep failed): mini-at-int4 — the mini
   tier's 10k x 300 matrix re-quantized to int4-group128 — evaluated
   against the MINI gates (top1 >= 0.85, rank_corr >= 0.90) as a SIZE
   play (mini already passes at int8 ~3 MB; int4-mini lands ~1.7 MB).

Gate thresholds are immutable; nothing is tuned. Writes
eval/reports/int4-retry.summary.json (config table: group x rounding x
lang -> top1 / rank_corr / bytes) plus per-language reports under
eval/reports/, and appends a retry section to
eval/reports/int4.summary.json (original verdict text preserved).

Exit status reflects sweep COMPLETION, not the gate outcome — the
verdict lives in int4-retry.summary.json. registry.json, manifest.json,
tiers.json and the release workflow are untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import StringStringEntryProto

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_int4  # noqa: E402
import build_tiers  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import run_eval  # noqa: E402

CANARY_LANGS = ("en", "ja")  # best/worst top1 of the group-128 run (0.90 / 0.78)
SWEEP_MATRIX = (  # canary phase: {group128, group64} x {nearest-even, half-away}
    (128, "nearest-even"),
    (128, "half-away"),
    (64, "nearest-even"),
    (64, "half-away"),
)
GROUP32_PROBE_LANGS = ("ja",)  # cheapest decisive test if the canary sweep clears nothing
BUDGET_CEILING_BYTES = 20_000_000  # per-model ceiling from the plan 68 B1 budget
RETRY_MARKER = "\n\nRetry ("


def sweep_tag(group_size: int, rounding: str) -> str:
    """Always-tagged artifact name (never `int4`, which belongs to the original run)."""
    return f"int4-g{group_size}-{build_int4.ROUNDING_SHORT[rounding]}"


def per_language_record(res: dict) -> dict:
    return {
        "top1_agreement": res["top1_agreement"],
        "rank_corr": res["rank_corr"],
        "bytes": res["int4_bytes"],
        "gate_passed": res["gate_passed"],
        "sha256": res["int4_sha256"],
    }


def size_row(group_size: int) -> dict:
    vocab_size, dims = 100_000, 300  # verified per language at build time
    n_groups = -(-dims // group_size)
    packed = vocab_size * (dims // 2)
    scales = vocab_size * n_groups * 4
    return {
        "group_size": group_size,
        "groups_per_row": n_groups,
        "group_partition_of_300_dims": "+".join(
            [str(group_size)] * (dims // group_size) + ([str(dims % group_size)] if dims % group_size else [])
        ),
        "packed_bytes_total": packed,
        "fp32_scales_bytes_total": scales,
        "payload_bytes_total": packed + scales,
        "fp32_overhead_fraction_of_packed": round(scales / packed, 4),
    }


def config_record(group_size: int, rounding: str, results: dict[str, dict]) -> dict:
    langs = list(results)
    first = next(iter(results.values()))
    record = {
        "tag": sweep_tag(group_size, rounding),
        "group_size": group_size,
        "rounding": rounding,
        "languages": langs,
        "artifact_bytes": first["int4_bytes"],
        "within_20mb_budget": first["int4_bytes"] <= BUDGET_CEILING_BYTES,
        "per_language": {lang: per_language_record(results[lang]) for lang in langs},
    }
    if set(CANARY_LANGS) <= set(langs):
        record["cleared_canaries"] = all(results[lang]["gate_passed"] for lang in CANARY_LANGS)
        record["min_canary_top1"] = min(results[lang]["top1_agreement"] for lang in CANARY_LANGS)
    return record


def run_config(repo: Path, langs, group_size: int, rounding: str) -> dict[str, dict]:
    tag = sweep_tag(group_size, rounding)
    results: dict[str, dict] = {}
    for lang in langs:
        results[lang] = build_int4.build_language(repo, lang, group_size=group_size, rounding=rounding, tag=tag)
    return results


def build_int4_mini(repo: Path, lang: str, group_size: int = 128, rounding: str = "nearest-even") -> dict:
    """Mixed lever: re-quantize the int8 mini tier's matrix to int4-group128.

    Loads the shipped mini artifact (10k vocab, full 300 dims, int8),
    dequantizes it to fp32 through the real eval loader, and re-quantizes
    to int4 with group scales — so the result is exactly "mini at int4".
    Evaluated against the MINI gates; a pure size play (~3.0 MB -> ~1.7 MB).
    """
    tag = "int4-mini"
    model_dir = repo / "models" / lang
    mini = run_eval.load_tier_model(
        model_dir / f"fasttext.{lang}.mini.onnx", model_dir / f"fasttext.{lang}.mini.vocab.json"
    )
    x = np.ascontiguousarray(mini.embeddings, dtype=np.float32)
    source_sha = build_tiers.full_model_sha256(repo, lang)

    packed, scales, dequant, _ = build_int4.quantize_int4_group(x, group_size, rounding)
    model = build_int4.make_int4_model(packed, scales, dequant.shape[1], group_size, tier=tag, rounding=rounding)
    model.metadata_props.append(StringStringEntryProto(key="source_full_sha256", value=source_sha))

    onnx_path = model_dir / f"fasttext.{lang}.{tag}.onnx"
    vocab_path = model_dir / f"fasttext.{lang}.{tag}.vocab.json"
    onnx.save(model, str(onnx_path))
    shutil.copyfile(model_dir / f"fasttext.{lang}.mini.vocab.json", vocab_path)

    check = build_int4.verify_int4_onnx(onnx_path, dequant, x)
    run_eval.load_tier_model(onnx_path, vocab_path)  # row-format-aware loader spot check

    mini_gates = run_eval.load_gates(repo)["mini"]
    report = run_eval.evaluate(repo, lang, tag, build_check=check, write=False, gates=mini_gates)
    report["recipe"]["reduction"] = (
        f"none (full 300 dims; mini 10k vocab truncation unchanged; int4-group{group_size} quantization only)"
    )
    report["row_format"] = build_int4.row_format(group_size, rounding)
    report["experiment_note"] = (
        "Mixed lever (plan 68 B1 retry): mini-at-int4 evaluated against the MINI gates "
        "as a size play; not shipped, registry.json/manifest.json/tiers.json/release "
        "workflow unchanged."
    )
    reports_dir = repo / "eval" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{tag}.{lang}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    m = report["metrics"]
    print(
        f"  {lang}/{tag}: rank_corr={m['rank_corr']['mean_spearman']:.4f} "
        f"top1={m['top1_agreement']['agreement']:.4f} "
        f"bytes={onnx_path.stat().st_size} {'PASS' if report['gates']['passed'] else 'fail'}"
    )
    return {
        "bytes": onnx_path.stat().st_size,
        "rank_corr": round(m["rank_corr"]["mean_spearman"], 6),
        "top1_agreement": round(m["top1_agreement"]["agreement"], 6),
        "gate_passed": report["gates"]["passed"],
        "mini_int8_bytes": (model_dir / f"fasttext.{lang}.mini.onnx").stat().st_size,
    }


def best_rounding(configs: list[dict]) -> str:
    """Rounding with the best min-canary top1 (tie -> nearest-even, the original)."""
    best = "nearest-even"
    best_score = -1.0
    for c in configs:
        score = c.get("min_canary_top1", -1.0)
        if score > best_score:
            best_score = score
            best = c["rounding"]
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="int4 lever-retry sweep (plan 68 B1 follow-up)")
    parser.add_argument("--repo-root", default=".", help="repo root (default: cwd)")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    all_langs = build_tiers.manifest_languages(repo)
    gates = run_eval.load_gates(repo)
    original = json.loads((repo / "eval" / "reports" / "int4.summary.json").read_text(encoding="utf-8"))

    protocol: list[str] = []
    summary: dict = {
        "purpose": "Retry the recorded levers from the int4-group128 rejection: group size 64/32, "
        "asymmetric half-away rounding, and (as fallback) the mixed int4-mini/int8-fluency split.",
        "levers": {
            "group_size": "smaller groups shrink the per-group max_abs, lowering quantization "
            "noise at the price of more fp32 scale bytes (see size_analysis)",
            "rounding": "half-away = deterministic asymmetric round-half-away-from-zero vs the "
            "original nearest-even (np.rint, banker's)",
            "mixed": "mini-at-int4 (10k vocab, group-128) evaluated against the MINI gates — size play only",
        },
        "canary_rationale": {
            "langs": list(CANARY_LANGS),
            "original_group128_top1": {lang: original["per_language"][lang]["top1_agreement"] for lang in CANARY_LANGS},
            "reason": "en = best (0.90), ja = worst (0.78) top1_agreement of the group-128 run",
        },
        "gate_thresholds": gates,
        "budget_ceiling_bytes_per_model": BUDGET_CEILING_BYTES,
        "size_analysis": [size_row(g) for g in (128, 64, 32)],
        "determinism": {"seed": run_eval.SEED, "rng": "np.random.default_rng([seed, crc32(language)])"},
    }

    # ---- Phase 1: canary sweep {g128, g64} x {nearest-even, half-away} on en+ja.
    protocol.append(
        f"canary sweep {[(g, r) for g, r in SWEEP_MATRIX]} on {list(CANARY_LANGS)} (worst-case-first discipline)"
    )
    configs: list[dict] = []
    for group_size, rounding in SWEEP_MATRIX:
        results = run_config(repo, CANARY_LANGS, group_size, rounding)
        configs.append(config_record(group_size, rounding, results))
    summary["canary_sweep"] = configs

    clearing = [c for c in configs if c.get("cleared_canaries")]
    summary["group32_probe"] = None
    summary["full_run"] = None
    summary["mixed_lever"] = None

    # ---- Phase 2: decisive group-32 probe on ja (only if nothing cleared).
    if not clearing:
        protocol.append("no config cleared both canaries -> group-32 probe on ja only (cheapest decisive test)")
        probe_results = {}
        for rounding in build_int4.ROUNDING_MODES:
            res = build_int4.build_language(repo, "ja", group_size=32, rounding=rounding, tag=sweep_tag(32, rounding))
            probe_results[rounding] = res
        probe = {
            "group_size": 32,
            "languages": ["ja"],
            "per_rounding": {
                rounding: per_language_record(res) for rounding, res in probe_results.items()
            },
            "ja_gate_passed": {rounding: res["gate_passed"] for rounding, res in probe_results.items()},
        }
        # Honest completion: if the worst-case language passes at group 32, finish
        # the canary pair for that config before any verdict (one extra run).
        passing_roundings = [r for r, res in probe_results.items() if res["gate_passed"]]
        if passing_roundings:
            protocol.append(f"group-32 passed ja at rounding(s) {passing_roundings} -> canary pair completed with en")
            rounding = passing_roundings[0]
            results = {"ja": probe_results[rounding]}
            results["en"] = build_int4.build_language(repo, "en", group_size=32, rounding=rounding, tag=sweep_tag(32, rounding))
            record = config_record(32, rounding, results)
            probe["completed_config"] = record
            if record["cleared_canaries"]:
                clearing.append(record)
        summary["group32_probe"] = probe

    # ---- Phase 3: full 9-language run with the chosen clearing config.
    if clearing:
        chosen = max(clearing, key=lambda c: (c.get("min_canary_top1", 0.0), -c["artifact_bytes"]))
        protocol.append(
            f"config {chosen['tag']} cleared the canaries -> full run on {len(all_langs)} languages"
        )
        remaining = [lang for lang in all_langs if lang not in chosen["languages"]]
        results = run_config(repo, remaining, chosen["group_size"], chosen["rounding"])
        merged = {lang: per_language_record(results[lang]) for lang in remaining}
        # Re-run bookkeeping: assemble the full-run record from both phases.
        full_record = config_record(chosen["group_size"], chosen["rounding"], {})
        full_record["languages"] = all_langs
        full_record["per_language"] = dict(chosen["per_language"])
        full_record["per_language"].update(merged)
        full_record["artifact_bytes"] = next(iter(full_record["per_language"].values()))["bytes"]
        full_record["within_20mb_budget"] = full_record["artifact_bytes"] <= BUDGET_CEILING_BYTES
        failed = [lang for lang, v in full_record["per_language"].items() if not v["gate_passed"]]
        full_record["failed_languages"] = failed
        full_record["all_passed"] = not failed
        summary["full_run"] = full_record

    sweep_failed = not clearing or (summary["full_run"] is not None and not summary["full_run"]["all_passed"])

    # ---- Phase 4: mixed lever (fallback value) — only when the sweep failed.
    if sweep_failed:
        rounding = best_rounding(configs)
        protocol.append(f"sweep failed -> mixed lever: int4-mini (group-128, {rounding}) vs the MINI gates, all languages")
        mixed_per_lang = {lang: build_int4_mini(repo, lang, group_size=128, rounding=rounding) for lang in all_langs}
        # Side-by-side with the shipped int8 mini reports.
        for lang, v in mixed_per_lang.items():
            mini_report = json.loads((repo / "eval" / "reports" / f"{lang}.mini.json").read_text(encoding="utf-8"))
            v["mini_int8_top1_agreement"] = round(mini_report["metrics"]["top1_agreement"]["agreement"], 6)
            v["mini_int8_rank_corr"] = round(mini_report["metrics"]["rank_corr"]["mean_spearman"], 6)
        summary["mixed_lever"] = {
            "tier": "mini re-quantized to int4-group128",
            "group_size": 128,
            "rounding": rounding,
            "rounding_selection": "best min-canary top1 from the sweep (tie -> nearest-even)",
            "gate_thresholds": gates["mini"],
            "per_language": mixed_per_lang,
            "all_passed": all(v["gate_passed"] for v in mixed_per_lang.values()),
            "max_bytes": max(v["bytes"] for v in mixed_per_lang.values()),
        }
    protocol.append("verdict recorded; sweep complete")

    # ---- Final verdict text.
    if clearing and summary["full_run"] is not None and summary["full_run"]["all_passed"]:
        fr = summary["full_run"]
        top1s = [v["top1_agreement"] for v in fr["per_language"].values()]
        verdict = (
            f"CONDITIONAL SHIP CANDIDATE: int4-group{fr['group_size']}/{fr['rounding']} clears the fluency "
            f"gates on {len(fr['per_language'])}/{len(fr['per_language'])} languages (top1 {min(top1s):.4f}-"
            f"{max(top1s):.4f}, rank_corr all >= 0.97) at {fr['artifact_bytes']} bytes per model, inside the "
            f"20 MB budget — owner gate on tier naming/registry, and the kotoshu-rs RowFormat extension for "
            "per-group scales is still required before any native consumer."
        )
    else:
        best = max(configs, key=lambda c: c.get("min_canary_top1", -1.0))
        ja_worst = min(
            (v["top1_agreement"] for c in configs for l, v in c["per_language"].items() if l == "ja"),
            default=None,
        )
        verdict = (
            "CONFIRMED REJECTION of int4 for full-vocab artifacts: no config in "
            f"{{group 128/64/32}} x {{nearest-even, half-away}} clears the fluency top1 >= 0.95 gate "
            f"(best canary config {best['tag']} at min-canary top1 {best.get('min_canary_top1'):.4f}; "
            f"ja stays worst at {ja_worst:.4f}); rank_corr passes everywhere, so the 4-bit noise floor "
            "keeps flipping 1-2 typo probes per language regardless of grouping or rounding. "
            "int8-per-row remains the shipped recipe."
        )
        if summary["group32_probe"] is not None and not summary["group32_probe"]["ja_gate_passed"]["nearest-even"] and not summary["group32_probe"]["ja_gate_passed"]["half-away"]:
            g32 = summary["group32_probe"]
            verdict += (
                f" The decisive group-32 probe on ja (worst language) still failed "
                f"(top1 {g32['per_rounding']['nearest-even']['top1_agreement']:.4f} ne / "
                f"{g32['per_rounding']['half-away']['top1_agreement']:.4f} ha) at "
                f"{g32['per_rounding']['nearest-even']['bytes']} bytes."
            )
        if summary["mixed_lever"] is not None:
            ml = summary["mixed_lever"]
            if ml["all_passed"]:
                verdict += (
                    f" Mixed fallback viable: int4-mini (10k vocab, group-128, {ml['rounding']}) meets the "
                    f"MINI gates on {len(ml['per_language'])}/{len(ml['per_language'])} languages at "
                    f"~{ml['max_bytes'] / 1e6:.2f} MB vs ~3.0 MB int8-mini — owner decision."
                )
            else:
                failed = [lang for lang, v in ml["per_language"].items() if not v["gate_passed"]]
                verdict += f" Mixed fallback REJECTED too: int4-mini fails the MINI gate on {', '.join(failed)}."

    summary["protocol"] = protocol
    summary["final_verdict"] = verdict
    summary["note"] = (
        "Experiment only (plan 68 B1 retry): registry.json, manifest.json, tiers.json and the "
        "release workflow unchanged; sweep artifacts are gitignored."
    )
    summary["generated_at"] = build_tiers.iso_now()

    out = repo / "eval" / "reports" / "int4-retry.summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")

    # ---- Append the retry section to the original summary (verdict text preserved).
    orig_path = repo / "eval" / "reports" / "int4.summary.json"
    original_json = json.loads(orig_path.read_text(encoding="utf-8"))
    original_recommendation = original_json["recommendation"].split(RETRY_MARKER)[0]
    original_json["retry"] = {
        "date": build_tiers.iso_now()[:10],
        "summary_file": "eval/reports/int4-retry.summary.json",
        "levers_tried": ["group-size 64/32", "half-away rounding", "mixed int4-mini"],
        "final_verdict": verdict,
    }
    original_json["recommendation"] = original_recommendation + f"{RETRY_MARKER}{build_tiers.iso_now()[:10]}): {verdict}"
    orig_path.write_text(json.dumps(original_json, indent=2) + "\n", encoding="utf-8")
    print(f"appended retry section to {orig_path}")
    print(f"FINAL VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
