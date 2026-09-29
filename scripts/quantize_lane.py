#!/usr/bin/env python3
"""S6: per-lane ONNX quantization + latency measurement (TODO.sota/6).

Quantizes a model artifact to int8 (dynamic weights) and measures
per-lane latency budgets with onnxruntime CPU threads matching the
deployment shape. Emits a lane record for the model manifest.

    python scripts/quantize_lane.py --model /path/model.onnx --name slate-en
"""
import argparse
import json
import time
from pathlib import Path

from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic


def measure(path, feeds, runs=200):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    # warmup
    for _ in range(20):
        sess.run(None, feeds)
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, feeds)
    return (time.perf_counter() - t0) / runs * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--feeds", default=None, help="JSON file with input feeds")
    args = ap.parse_args()

    src = Path(args.model)
    dst = src.with_suffix(".int8.onnx")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8,
                     per_channel=True, quant_format=QuantFormat.QOperator
                     ) if False else quantize_dynamic(str(src), str(dst))

    feeds = json.loads(Path(args.feeds).read_text()) if args.feeds else None
    record = {"name": args.name, "fp32_bytes": src.stat().st_size,
              "int8_bytes": dst.stat().st_size,
              "int8_ir": None}
    import onnx
    m = onnx.load(str(dst))
    record["int8_ir"] = m.ir_version
    if feeds:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
        dtypes = {i.name: i.type for i in sess.get_inputs()}
        def dtype_for(t):
            return np.int64 if t == "tensor(int64)" else np.float32
        np_feeds = {k: np.asarray(v, dtype=dtype_for(dtypes.get(k, "tensor(float)")))
                    for k, v in feeds.items()}
        record["fp32_ms"] = round(measure(src, np_feeds), 3)
        record["int8_ms"] = round(measure(dst, np_feeds), 3)
    out = src.with_suffix(".lane.json")
    out.write_text(json.dumps(record, indent=1))
    print(json.dumps(record))


if __name__ == "__main__":
    main()
