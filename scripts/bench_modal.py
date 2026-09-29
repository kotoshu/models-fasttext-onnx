#!/usr/bin/env python3
"""Run the wave-2 suggester benches on Modal (serial-local was the wall-time
bottleneck; accuracy is unaffected by isolated concurrent containers).

    modal run scripts/bench_modal.py

Phases:
  1. vi probe — must reproduce the frozen vi field lanes exactly
     (symspell 0.6515/0.0, hunspell 0.0/0.0) before anything fans out.
  2. fan out ko, zh-Hant-HK, ar, pl, fr, ja concurrently; write each
     report into eval/reports/ as it lands.

Pins chosen to match the machine that produced the frozen wave-2 numbers:
Ruby 3.4.8, symspellpy 6.10.0, wordfreq 3.1.1, numpy 2.2.6, hunspell 1.7
(bookworm), gem deps thor 1.5.0 / rubyzip 2.4.1 / lutaml-model 0.8.61.
Gem source: kotoshu main 9db84511 (vi raw scoring, PR #235).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

STAGE = Path("/tmp/kotoshu-modal-stage")
LANGS = ["ko", "zh-Hant-HK", "ar", "pl", "fr", "ja"]
LOCAL_REPO = Path(__file__).resolve().parents[1]

image = (
    modal.Image.from_registry("ruby:3.4.8-slim", add_python="3.11")
    .apt_install("hunspell", "build-essential", "pkg-config", "ca-certificates",
                 "curl", "clang", "libclang-dev")
    .pip_install("numpy==2.2.6", "symspellpy==6.10.0", "wordfreq==3.1.1")
    .run_commands(
        "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
        "gem install thor:1.5.0 rubyzip:2.4.1 lutaml-model:0.8.61 rb_sys:0.9.130 --no-document",
    )
    .add_local_dir(str(STAGE / "kotoshu"), "/root/src/kotoshu/kotoshu", copy=True)
    .add_local_dir(str(STAGE / "repo"), "/root/repo", copy=True)
    .add_local_file(str(STAGE / "bench_kotoshu.rb"), "/tmp/bench_kotoshu.rb", copy=True)
    .add_local_dir(str(STAGE / "bench-dicts"), "/tmp/bench-dicts", copy=True)
    .add_local_dir(str(STAGE / "cache"), "/root/.cache/kotoshu", copy=True)
    .run_commands(
        'export PATH="$HOME/.cargo/bin:$PATH" && '
        "cd /root/src/kotoshu/kotoshu/ext/kotoshu_native && "
        "ruby extconf.rb >/tmp/ext.log 2>&1 && make -j4 >>/tmp/ext.log 2>&1 && "
        "cp kotoshu_native.so ../../lib/kotoshu/ && "
        "cd /root/src/kotoshu/kotoshu && "
        'ruby -e \'$LOAD_PATH.unshift "lib"; require "kotoshu"; '
        'abort "native ext failed to load" unless Kotoshu::Native.available?\' '
        "|| (cat /tmp/ext.log; exit 1)"
    )
)

app = modal.App("kotoshu-bench-lanes", image=image)

# Reports persist server-side: a lane's report lands in the volume the
# moment its container exits, regardless of whether the machine that
# spawned the run is still up (2026-09-26: a Mac reboot killed the
# driver mid-fan-out and lost every in-flight lane).
volume = modal.Volume.from_name("kotoshu-bench-reports", create_if_missing=True)


@app.function(cpu=1, memory=8192, timeout=86_400, volumes={"/vol": volume})
def run_bench(lang: str):
    report_path = Path(f"/root/repo/eval/reports/suggest-benchmark-{lang}-wave2.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, KOTOSHU_BENCH_TIMEOUT="86400")
    proc = subprocess.run(
        ["python", "scripts/benchmark_suggesters.py", "--lang", lang, "--split2"],
        cwd="/root/repo", capture_output=True, text=True, env=env,
    )
    report = report_path.read_text() if report_path.exists() else None
    if report:
        Path(f"/vol/suggest-benchmark-{lang}-wave2.json").write_text(report)
    else:
        # A reportless lane must never hide (two campaigns lost hours to
        # silent reportless exits) — persist the failure for collect.
        Path(f"/vol/FAILED-{lang}.txt").write_text(
            f"rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}")
    return {
        "lang": lang,
        "rc": proc.returncode,
        "log": proc.stdout[-4000:] + proc.stderr[-2000:],
        "report": report,
    }


def write_report(result):
    text = result["report"]
    if not text:
        print(f"[{result['lang']}] NO REPORT rc={result['rc']}\n{result['log']}")
        return None
    out = LOCAL_REPO / f"eval/reports/suggest-benchmark-{result['lang']}-wave2.json"
    out.write_text(text)
    for line in result["log"].splitlines():
        if line.startswith(("hunspell:", "symspell:", "kotoshu:")):
            print(f"[{result['lang']}] {line}")
    return json.loads(text)


@app.function(cpu=1, memory=8192, timeout=86_400, volumes={"/vol": volume})
def translit_capability(lang: str):
    """Capability-class measurement: the {lang}.suggest3-translit.json
    split (romanization typo -> native word). Field lanes have no
    cross-script capability — they are measured anyway for the record
    and are expected to score ~0. The kotoshu lane is the only
    meaningful number."""
    import json as _json
    pairs = _json.loads(Path(f"/root/repo/eval/realword/{lang}.suggest3-translit.json").read_text())["pairs"]
    words = [p["typo"] for p in pairs]
    env = dict(os.environ, BENCH_LANG=lang, KOTOSHU_BENCH_TIMEOUT="86400")
    out = {}
    # kotoshu lane (the gem HTTP-equivalent: the ruby lane script)
    proc = subprocess.run(
        ["ruby", "/tmp/bench_kotoshu.rb"],
        input="\n".join(words) + "\n", capture_output=True, text=True,
        cwd="/root/src/kotoshu/kotoshu", env=env)
    mapping = {}
    for line in proc.stdout.splitlines():
        try:
            mapping.update(_json.loads(line))
        except _json.JSONDecodeError:
            pass
    top1 = sum(1 for p in pairs if (mapping.get(p["typo"]) or [None])[0]
               and mapping[p["typo"]][0].lower() == p["correction"].lower())
    top5 = sum(1 for p in pairs if p["correction"].lower() in
               [x.lower() for x in (mapping.get(p["typo"]) or [])[:5]])
    # field lanes for the record (expected ~0: no cross-script capability)
    field = {}
    try:
        from symspellpy import SymSpell, Verbosity
        import tempfile
        cache = Path("/root/.cache/kotoshu/frequency-lists") / lang / "frequency.json"
        data = _json.loads(cache.read_text())
        n = len(data["full_list"])
        fh = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False)
        for e in data["full_list"]:
            fh.write(f"{e['word']}\t{n - e['rank'] + 1}\n")
        fh.close()
        sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        sym.load_dictionary(fh.name, term_index=0, count_index=1, separator="\t", encoding="utf-8")
        f_top1 = f_top5 = 0
        for p in pairs:
            got = [s.term for s in sym.lookup(p["typo"], Verbosity.TOP, 2)[:8]]
            if got and got[0].lower() == p["correction"].lower():
                f_top1 += 1
            if p["correction"].lower() in [x.lower() for x in got[:5]]:
                f_top5 += 1
        field["symspell"] = {"top1": round(f_top1 / len(pairs), 4),
                             "top5": round(f_top5 / len(pairs), 4), "n": len(pairs)}
    except Exception as e:
        field["symspell"] = {"error": str(e)[:120]}
    report = {
        "spec": "kotoshu.translit-capability/v1", "language": lang,
        "class": "translit (romanization typo -> native word)",
        "engines": {
            "kotoshu": {"top1": round(top1 / len(pairs), 4),
                        "top5": round(top5 / len(pairs), 4), "n": len(pairs)},
            **field}}
    out_path = Path(f"/vol/suggest-benchmark-{lang}-translit-capability.json")
    out_path.write_text(_json.dumps(report, indent=1) + "\n")
    return report


@app.local_entrypoint()
def spawn_translit(langs: str = "ar"):
    """Fire-and-forget the translit capability measurement."""
    handles = [translit_capability.spawn(lang) for lang in langs.split(",")]
    print("spawned capability:", [h.object_id for h in handles])


@app.local_entrypoint()
def spawn(langs: str = ""):
    """Fire-and-forget: launch lanes server-side; safe to close the
    terminal immediately. Reports persist in the kotoshu-bench-reports
    volume; collect them later with `collect`. Pass a comma list to
    re-run specific languages, e.g. --langs zh-Hant-HK,ja."""
    targets = (langs.split(",") if langs else ["vi"] + LANGS)
    handles = [run_bench.spawn(lang) for lang in targets]
    print("spawned:", [h.object_id for h in handles])


@app.function(cpu=1, memory=8192, timeout=3_600, volumes={"/vol": volume})
def diag(lang: str, max_pairs: int = 30):
    """Run a small sample end-to-end and return everything the harness
    printed — names the exact in-container failure for lanes that exit
    without a report."""
    env = dict(os.environ, KOTOSHU_BENCH_TIMEOUT="3600")
    proc = subprocess.run(
        ["python", "scripts/benchmark_suggesters.py", "--lang", lang,
         "--split2", "--max", str(max_pairs)],
        cwd="/root/repo", capture_output=True, text=True, env=env,
    )
    return {"lang": lang, "rc": proc.returncode,
            "out": proc.stdout[-6000:], "err": proc.stderr[-6000:]}


@app.local_entrypoint()
def diagnose(langs: str = "pl,fr"):
    for lang in langs.split(","):
        result = diag.remote(lang)
        print(f"== {lang} rc={result['rc']}")
        print(result["out"][-1500:])
        print(result["err"][-2500:])


@app.local_entrypoint()
def collect():
    """Pull whatever reports have landed in the volume into eval/reports/."""
    vol = modal.Volume.from_name("kotoshu-bench-reports")
    names = [e.path for e in vol.listdir("/") if e.path.endswith(".json")]
    for name in sorted(names):
        base = Path(name).name.removesuffix(".json")
        out = LOCAL_REPO / f"eval/reports/{base}.json"
        with open(out, "wb") as fh:
            for chunk in vol.read_file(name):
                fh.write(chunk)
        report = json.loads(out.read_text())
        ks = report["engines"]["kotoshu"]
        if "nonword" in ks:
            print(f"{base}: kotoshu nonword top1={ks['nonword']['top1']}")
        else:
            print(f"{base}: kotoshu top1={ks.get('top1')}")
        # cleanup: the collect also fetches stray FAILED markers
    for marker in (LOCAL_REPO / "eval/reports").glob("FAILED-*.txt"):
        marker.unlink()


@app.local_entrypoint()
def main():
    frozen = json.loads(
        (LOCAL_REPO / "eval/reports/suggest-benchmark-vi-wave2.json").read_text()
    )

    print("phase 1: vi host-consistency probe", flush=True)
    vi = write_report(run_bench.remote("vi"))
    if vi is None:
        sys.exit("vi produced no report — not fanning out")
    for lane in ("hunspell", "symspell"):
        if vi["engines"][lane] != frozen["engines"][lane]:
            sys.exit(
                f"host drift on {lane}: modal={vi['engines'][lane]} "
                f"frozen={frozen['engines'][lane]} — not fanning out"
            )
    print("vi field lanes reproduce frozen numbers — host consistent", flush=True)

    print(f"phase 2: fanning out {LANGS}", flush=True)
    for result in run_bench.map(LANGS):
        write_report(result)
    print("all lanes complete", flush=True)
