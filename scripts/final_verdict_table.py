#!/usr/bin/env python3
"""Generate the wave-2 verdict table from the frozen reports.

Reads eval/reports/suggest-benchmark-{lang}-wave2.json for all 16
languages, detects the report generation via the field-lane
fingerprints (the dictionary-gated splits changed the field lanes too —
a pre-gate file carries the old field numbers and is flagged STALE),
and writes eval/reports/verdict-table.md.

    python3 scripts/final_verdict_table.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LANGS = ["en", "de", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt",
         "ru", "vi", "zh-Hans-CN", "zh-Hant-TW", "zh-Hant-HK", "ar"]

# Pre-gate field-lane nonword top-1 fingerprints (the reports written
# before models PR #8's dictionary-gated splits). A report whose field
# number still matches its fingerprint predates the clean splits.
PRE_GATE_FIELD = {
    "en": 0.855, "de": 0.870, "es": 0.820, "it": 0.940, "nl": 0.846,
    "pt": 0.809, "ru": 0.864, "ko": 0.613,
}
# zh-Hans-CN / zh-Hant-TW are deliberately absent: their splits were
# never regenerated (0 dictionary-valid typos), so old and new field
# numbers are identical and the fingerprint cannot discriminate — the
# volume-mtime audit is the authority for those two.


def field_top1(report):
    ss = report["engines"].get("symspell", {}).get("nonword", {}).get("top1") or 0
    hs = report["engines"].get("hunspell", {}).get("nonword", {}).get("top1") or 0
    return max(ss, hs)


def realword_top1(report):
    ss = report["engines"].get("symspell", {}).get("realword", {}).get("top1") or 0
    hs = report["engines"].get("hunspell", {}).get("realword", {}).get("top1") or 0
    k = report["engines"]["kotoshu"]["realword"].get("top1") or 0
    return k, max(ss, hs)


def main():
    rows, pending, stale = [], [], []
    for lang in LANGS:
        path = REPO / f"eval/reports/suggest-benchmark-{lang}-wave2.json"
        if not path.exists():
            pending.append(lang)
            continue
        r = json.loads(path.read_text())
        ks = r["engines"]["kotoshu"]
        nw, rw = ks["nonword"], ks["realword"]
        f1 = field_top1(r)
        kr, fr = realword_top1(r)
        if lang in PRE_GATE_FIELD and abs(f1 - PRE_GATE_FIELD[lang]) < 0.0005:
            stale.append(lang)
        rw_cell = "—" if rw.get("n", 0) == 0 else f"{kr*100:.1f} vs {fr*100:.1f}"
        gate = "WIN" if (nw["top1"] >= f1 and kr >= fr) else "MISS"
        rows.append((lang, nw["top1"], nw["top3"], nw["top5"], f1, rw_cell, gate))

    lines = [
        "# Wave-2 verdict table (final)",
        "",
        "Nonword/realword exact-match top-1 over the frozen dictionary-gated",
        "splits; field = best of the symspell/hunspell lanes on the same",
        "published frequency lists. Gate: kotoshu ≥ every field lane on both",
        "classes. Every number traces to a committed",
        "`eval/reports/suggest-benchmark-{lang}-wave2.json`.",
        "",
        "| lang | kotoshu t1 | t3 | t5 | field t1 | rw t1 (k vs field) | gate |",
        "|---|---|---|---|---|---|---|",
    ]
    wins = 0
    for lang, k1, k3, k5, f1, rw_cell, gate in rows:
        wins += gate == "WIN"
        lines.append(f"| {lang} | {k1*100:.2f} | {k3*100:.2f} | {k5*100:.2f} "
                     f"| {f1*100:.2f} | {rw_cell} | {gate} |")
    lines += ["", f"**{wins}/{len(rows)} WIN"
              + ("**" if not pending and not stale else
                 f"** — {len(pending)} pending, {len(stale)} stale"
                 + (f" ({', '.join(pending + stale)})" if (pending or stale) else ""))]
    out = REPO / "eval/reports/verdict-table.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({wins}/{len(rows)} WIN, {len(pending)} pending, {len(stale)} stale)")


if __name__ == "__main__":
    main()
