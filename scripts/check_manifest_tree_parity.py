#!/usr/bin/env python3
"""Tree-parity check for the surgical manifest (TODO.deploy/9).

generate_manifest.rb is tree-derived; since the full tiers became
release-only (TODO.deploy/2) the committed manifest is maintained
surgically and regeneration alone would DROP those entries. The drift
invariant therefore becomes: for every artifact present in the tree,
the committed manifest must carry the identical sha256/size (regenerated
entries are a subset of committed ones, matching exactly). Extra
committed entries are the release-only artifacts and are legal.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

committed = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
subprocess.run(["ruby", "scripts/generate_manifest.rb"], check=True,
               cwd=ROOT, capture_output=True)
regenerated = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
(ROOT / "manifest.json").write_text(json.dumps(committed, indent=2) + "\n",
                                    encoding="utf-8")

errors = []
for key, entry in sorted(regenerated["resources"].items()):
    other = committed["resources"].get(key)
    if other is None:
        errors.append(f"{key}: present in the tree but missing from the committed manifest")
    elif (other.get("sha256"), other.get("size")) != (entry.get("sha256"), entry.get("size")):
        errors.append(f"{key}: tree artifact drifted from the committed manifest "
                      f"(committed {other.get('sha256', '?')[:12]}, tree {entry.get('sha256', '?')[:12]})")

n_extra = len(committed["resources"]) - len(regenerated["resources"])
if errors:
    for e in errors:
        print(f"[FAIL] {e}")
    sys.exit(f"manifest tree-parity failed: {len(errors)} error(s)")
print(f"manifest tree-parity OK: {len(regenerated['resources'])} tree artifacts match; "
      f"{n_extra} release-only entries exempt")
