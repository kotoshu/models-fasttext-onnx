#!/usr/bin/env python3
"""Load-verify every .onnx under models/ and write docs/inventory.json.

For each model: onnx.load, an onnxruntime CPU session, vocabulary
cross-check against the sibling vocab.json, and a 3-index gather
round-trip against the vanilla (full) or dequantized (tier) matrix.
Exits nonzero if any model fails any check.
"""

from __future__ import annotations

import gc
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

REPO = Path(__file__).resolve().parent.parent
SPOT_TOL = 1e-4
SPOT_INDICES = 3


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def constant_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            arr = numpy_helper.to_array(attr.t)
            for name in filter(None, (attr.t.name, *node.output)):
                arrays[name] = arr
    return arrays


def tier_from_filename(path: Path, lang: str) -> str:
    rest = path.stem[len(f"fasttext.{lang}") :]
    return rest.lstrip(".") or "full"


def verify_model(path: Path, lang: str, tier: str) -> dict:
    record: dict = {
        "language": lang,
        "tier": tier,
        "file": str(path.relative_to(REPO)),
        "dims": None,
        "vocab_size": None,
        "quantization": None,
        "bytes": path.stat().st_size,
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "vocab_json_ok": False,
        "ort_load_ok": False,
        "spot_inference_ok": False,
        "error": None,
    }
    try:
        model = onnx.load(str(path))
        props = {p.key: p.value for p in model.metadata_props}
        arrays = constant_arrays(model)

        if "q_embeddings" in arrays and "row_scale" in arrays:
            q = arrays["q_embeddings"]
            scale = arrays["row_scale"]
            if q.ndim != 2 or scale.shape != (q.shape[0],):
                raise ValueError(f"bad tier tensor shapes q={q.shape} scale={scale.shape}")
            matrix = q.astype(np.float32) * scale.astype(np.float32)[:, None]
            record["quantization"] = props.get("quantization", "int8-per-row")
        else:
            matrix = arrays.get("word_embeddings", arrays.get("embeddings_matrix"))
            if matrix is None:
                raise KeyError("no embedding matrix constant found")
            if matrix.ndim != 2:
                raise ValueError(f"embedding matrix not 2-D: {matrix.shape}")
            matrix = matrix.astype(np.float32)

        rows, dims = matrix.shape
        record["dims"] = int(dims)
        record["vocab_size"] = int(rows)

        meta_vocab = props.get("vocabulary_size")
        meta_dims = props.get("embedding_dimension")
        if meta_vocab is None or int(meta_vocab) != rows:
            raise ValueError(f"metadata vocabulary_size {meta_vocab!r} != matrix rows {rows}")
        if meta_dims is None or int(meta_dims) != dims:
            raise ValueError(f"metadata embedding_dimension {meta_dims!r} != matrix dims {dims}")
        meta_tier = props.get("tier")
        if meta_tier is not None and meta_tier != tier:
            raise ValueError(f"metadata tier {meta_tier!r} != filename tier {tier!r}")

        vocab_path = path.with_suffix(".vocab.json")
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        word_to_idx = vocab["word_to_idx"]
        indices = word_to_idx.values()
        record["vocab_json_ok"] = (
            vocab["vocab_size"] == rows
            and len(word_to_idx) == rows
            and max(indices) < rows
            and min(indices) >= 0
        )
        if not record["vocab_json_ok"]:
            raise ValueError(
                f"vocab json disagrees with matrix rows: vocab_size={vocab['vocab_size']} "
                f"entries={len(word_to_idx)} rows={rows}"
            )
        del vocab, word_to_idx, matrix
        gc.collect()

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        record["ort_load_ok"] = True

        if "q_embeddings" in arrays and "row_scale" in arrays:
            matrix = arrays["q_embeddings"].astype(np.float32) * arrays["row_scale"].astype(np.float32)[:, None]
        else:
            matrix = arrays.get("word_embeddings", arrays["embeddings_matrix"]).astype(np.float32)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        worst = 0.0
        for i in sorted({0, rows // 2, rows - 1}):
            out = session.run([output_name], {input_name: np.array([i], dtype=np.int64)})[0]
            if out.shape != (dims,):
                raise ValueError(f"spot inference index {i}: output shape {out.shape} != ({dims},)")
            worst = max(worst, float(np.abs(out.astype(np.float32) - matrix[i]).max()))
        if worst >= SPOT_TOL:
            raise ValueError(f"spot inference max abs diff {worst} >= tol {SPOT_TOL}")
        record["spot_inference_ok"] = True
        record["spot_indices"] = sorted({0, rows // 2, rows - 1})
        record["spot_max_abs_diff"] = worst
    except Exception as exc:  # noqa: BLE001 - record and continue with other models
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main() -> int:
    records = []
    for path in sorted((REPO / "models").glob("*/*.onnx")):
        lang = path.parent.name
        tier = tier_from_filename(path, lang)
        record = verify_model(path, lang, tier)
        records.append(record)
        status = "ok" if record["error"] is None else "FAIL"
        line = (
            f"{lang}/{tier} {record['quantization'] or 'fp32'} "
            f"vocab={record['vocab_size']} dims={record['dims']} "
            f"bytes={record['bytes']} sha256={record['sha256'][:12]} "
            f"ort={'ok' if record['ort_load_ok'] else 'FAIL'} "
            f"infer={'ok' if record['spot_inference_ok'] else 'FAIL'} "
            f"vocab={'ok' if record['vocab_json_ok'] else 'FAIL'}"
        )
        print(f"{line} [{status}]" + (f" {record['error']}" if record["error"] else ""))

    records.sort(key=lambda r: (r["language"], r["tier"]))
    by_tier: dict[str, int] = {}
    for r in records:
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
    failures = [r for r in records if r["error"] is not None]
    inventory = {
        "generated_at": iso_now(),
        "counts": {
            "languages": len({r["language"] for r in records}),
            "models": len(records),
            "by_tier": {t: by_tier[t] for t in sorted(by_tier)},
            "total_bytes": sum(r["bytes"] for r in records),
            "failed": len(failures),
        },
        "models": records,
    }
    out = REPO / "docs" / "inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")

    c = inventory["counts"]
    print(
        f"{c['models']} models / {c['languages']} languages / {c['total_bytes']} bytes "
        f"({', '.join(f'{t}={n}' for t, n in c['by_tier'].items())}); "
        f"failed={c['failed']}; wrote {out.relative_to(REPO)}"
    )
    if failures:
        for r in failures:
            print(f"FAILED {r['file']}: {r['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
