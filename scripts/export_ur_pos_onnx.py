"""Export the ur POS pipeline slot to ONNX (the train→gate→ONNX tail).

Builds a plain Gemm graph from the npz the trainer wrote: three one-hot
input segments (prev/cur/next word index) → Gather per segment → concat
→ Gemm(weights) → Argmax. Placeholder-class by design (see
scripts/train_ur_pos.py); the 100K gold swaps in the production
fasttext under the same artifact contract.

Usage: python3 scripts/export_ur_pos_onnx.py --model models/ur
"""
import argparse
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="models/ur directory")
    a = ap.parse_args()

    npz = np.load(os.path.join(a.model, "fasttext.ur.pos.npz"))
    w = npz["weights"]
    n_words = int(npz["n_words"])
    vocab = json.load(open(os.path.join(a.model, "fasttext.ur.pos.vocab.json")))
    labels = vocab["labels"]

    idx_in = helper.make_tensor_value_info("indices", TensorProto.INT64, [3])
    scores_out = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [len(labels)])

    seg_nodes, seg_outputs = [], []
    for k, name in enumerate(("prev", "cur", "next")):
        gather_out = f"{name}_onehot"
        seg_nodes.append(helper.make_node(
            "Gather", ["eye_table", "indices"], [gather_out],
            axis=0, name=f"gather_{name}"))
        seg_outputs.append(gather_out)
    concat_out = "features"
    seg_nodes.append(helper.make_node(
        "Concat", seg_outputs, [concat_out], axis=1, name="concat_segments"))
    seg_nodes.append(helper.make_node(
        "Gemm", [concat_out, "weights"], ["scores_transposed"],
        alpha=1.0, beta=0.0, transB=0, name="linear"))
    seg_nodes.append(helper.make_node(
        "Transpose", ["scores_transposed"], ["scores"], perm=[1, 0], name="transpose"))

    eye = np.eye(n_words, dtype=np.float32)
    graph = helper.make_graph(
        seg_nodes, "ur_pos_slot",
        [idx_in], [scores_out],
        initializer=[
            numpy_helper.from_array(eye, "eye_table"),
            numpy_helper.from_array(w, "weights"),
        ])
    graph.doc_string = ("ur POS pipeline slot — SAMPLE-SCALE placeholder; "
                        "accuracy gate deferred until the 100K gold arrives")
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    out_path = os.path.join(a.model, "fasttext.ur.pos.onnx")
    onnx.save(model, out_path)
    onnx.checker.check_model(onnx.load(out_path))
    json.dump({"labels": labels, "n_words": n_words, "opset": 13,
               "input": "indices: [prev_word_idx, cur_word_idx, next_word_idx] (int64)",
               "note": "SAMPLE-SCALE placeholder pipeline"},
              open(os.path.join(a.model, "fasttext.ur.pos.onnx.meta.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes) — checker passed")


if __name__ == "__main__":
    main()
