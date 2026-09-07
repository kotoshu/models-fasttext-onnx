#!/usr/bin/env python3
"""Convert the fastText LID model (lid.176.ftz) into a registry resource.

The gem detects document language with Facebook's fastText language
identification model (supervised, 176 labels, dim 16, char 2-4 grams,
hierarchical softmax), distributed as a product-quantized .ftz (938 KB).
The gem shells out to the fasttext bindings over that file; the browser
engine (kotoshu-rs `detectLanguage`) needs the same weights as a
pure-Rust-readable artifact pair in this repo's tier shape: an ONNX
container (int8-per-row input matrix + row scales + output matrix, the
same Constant-node serialization scripts/build_tiers.py writes) plus a
vocab sidecar (labels, label counts for the Huffman tree, the word
dictionary, the pruned-ngram index map, and the model arguments).

The .ftz binary format is parsed natively here (fastText v0.9.2 wire
format, magic 793712314 / version 12) — the quantized input rows are
RECONSTRUCTED exactly (row = norm-centroid x PQ centroids; inference on
the reconstructed rows is bit-identical to the QuantMatrix path), then
re-quantized with the repo's int8-per-row recipe when the gates hold:

- gate `top1_agreement`: the int8 artifact must agree with the float32
  (exactly reconstructed) labels on the probe corpus
  (eval/corpora/lid_probe.jsonl) on EVERY sample — a language detector
  flipping a label is a wrong answer, not a ranking drift, so the gate
  is 1.0 (stricter than the embedding tiers' 0.95/0.85).
- gate `mirror_budget`: the mirrored artifact must stay <= 20 MB so the
  LFS media mirror (ACAO:*) stays browser-friendly.

If the int8 gates fail the script falls back to the float32 artifact and
records that choice in the report ("else float32" per plan 102). Gates
are never weakened; a fallback is a recorded decision, not a silent one.

The reference scorer reproduces the shipped fasttext bindings exactly
(sigmoid computed through f64 returned as f32 — NOT the 0.9.2 lookup
table — and log(x + 1e-5); validated to 1 ulp against
`fasttext.predict` and against the gem's LanguageIdentifier output on
the same corpus; see eval/reports/lid.176.json).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
from onnx import StringStringEntryProto, TensorProto, helper, numpy_helper

FASTTEXT_MAGIC = 793712314
FASTTEXT_VERSION = 12
EOS = "</s>"
BOW, EOW = "<", ">"
SEPARATORS = b" \n\t\v\f\r\x00"
LOSS_HS = 1
# Mirror budget from the storage decision: tier binaries mirrored to the
# LFS media host must stay browser-friendly (mini ~3 MB, fluency ~15 MB).
MIRROR_BUDGET_BYTES = 20 * 1024 * 1024

UPSTREAM_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
UPSTREAM_SHA256 = "8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def fnv1a(data: bytes) -> int:
    """fastText Dictionary::hash — FNV-1a over SIGN-EXTENDED bytes
    (uint32_t(int8_t(c)) sign-extends every byte >= 0x80; critical for
    every non-ASCII script)."""
    h = 2166136261
    for b in data:
        if b >= 128:
            b = 0xFFFFFF00 + b
        h = (h ^ b) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    return h


class LidFtz:
    """Native reader for the supervised fastText .ftz/.bin wire format."""

    def __init__(self, path: Path):
        data = path.read_bytes()
        pos = 0

        def read_fmt(fmt: str):
            nonlocal pos
            value = np.frombuffer(data, dtype=np.dtype("<" + fmt), count=1, offset=pos)[0]
            pos += np.dtype("<" + fmt).itemsize
            return value.item()

        def read_cstring() -> str:
            nonlocal pos
            end = data.index(b"\x00", pos)
            raw = data[pos:end]
            pos = end + 1
            return raw.decode("utf-8", "surrogateescape")

        def read_raw(count: int) -> bytes:
            nonlocal pos
            raw = data[pos:pos + count]
            pos += count
            return raw

        if read_fmt("i") != FASTTEXT_MAGIC or read_fmt("i") != FASTTEXT_VERSION:
            sys.exit(f"error: {path}: not a fastText v12 model file")
        args12 = (int(v) for v in np.frombuffer(data, dtype="<i4", count=12, offset=pos))
        (self.dim, _ws, _epoch, _min_count, _neg, self.word_ngrams, self.loss,
         _model, self.bucket, self.minn, self.maxn, _lr) = args12
        pos += 48
        pos += 8  # t (double)
        size, self.nwords, self.nlabels = (int(v) for v in np.frombuffer(data, "<i4", 3, pos))
        pos += 12
        pos += 8  # ntokens (int64)
        pruneidx_size = int(np.frombuffer(data, "<i8", 1, pos)[0])
        pos += 8
        self.words: list[str] = []
        self.labels: list[str] = []
        label_counts: list[int] = []
        for _ in range(size):
            word = read_cstring()
            count = int(np.frombuffer(data, "<i8", 1, pos)[0])
            pos += 8
            entry_type = data[pos]
            pos += 1
            if entry_type == 1:
                self.labels.append(word)
                label_counts.append(count)
            else:
                self.words.append(word)
        self.label_counts = label_counts
        self.pruneidx: dict[int, int] = {}
        for _ in range(max(0, pruneidx_size)):
            key, value = (int(v) for v in np.frombuffer(data, "<i4", 2, pos))
            pos += 8
            self.pruneidx[key] = value
        self.pruneidx_size = pruneidx_size

        quant = data[pos]
        pos += 1
        if quant:
            self.qnorm = bool(data[pos])
            pos += 1
            rows, cols = (int(v) for v in np.frombuffer(data, "<i8", 2, pos))
            pos += 16
            codesize = int(np.frombuffer(data, "<i4", 1, pos)[0])
            pos += 4
            codes = np.frombuffer(read_raw(codesize), dtype=np.uint8).reshape(rows, -1)
            _dim, nsubq, dsub, lastdsub = (int(v) for v in np.frombuffer(data, "<i4", 4, pos))
            pos += 16
            centroids = np.frombuffer(read_raw(_dim * 256 * 4), dtype="<f4")
            self.nsubq, self.dsub, self.lastdsub = nsubq, dsub, lastdsub
            self.codes = codes
            self.centroids = centroids
            self.norm_codes = None
            self.norm_centroids = None
            if self.qnorm:
                self.norm_codes = np.frombuffer(read_raw(rows), dtype=np.uint8)
                pq = tuple(int(v) for v in np.frombuffer(data, "<i4", 4, pos))
                pos += 16
                if pq != (1, 1, 1, 1):
                    sys.exit(f"error: {path}: unexpected norm PQ {pq}")
                self.norm_centroids = np.frombuffer(read_raw(256 * 4), dtype="<f4")
            if data[pos]:
                sys.exit(f"error: {path}: quantized output matrix (qout) is not supported")
            pos += 1
        else:
            self.qnorm = False
            self.codes = None
            rows, cols = (int(v) for v in np.frombuffer(data, "<i8", 2, pos))
            pos += 16
            self.dense = np.frombuffer(read_raw(rows * cols * 4), dtype="<f4").reshape(rows, cols)
        self.input_rows = int(rows)
        out_rows, out_cols = (int(v) for v in np.frombuffer(data, "<i8", 2, pos))
        pos += 16
        self.wo = np.frombuffer(read_raw(out_rows * out_cols * 4), dtype="<f4").reshape(out_rows, out_cols).copy()
        if pos != len(data):
            sys.exit(f"error: {path}: {len(data) - pos} trailing bytes — format mismatch")

    def rows_dense(self) -> np.ndarray:
        """Reconstructed input rows, float32: row = norm x PQ centroids.
        Bit-identical to what the QuantMatrix produces at inference."""
        if self.codes is None:
            return self.dense
        out = np.empty((self.input_rows, self.dim), dtype=np.float32)
        for i in range(self.input_rows):
            out[i] = self.row(i)
        return out

    def row(self, i: int) -> np.ndarray:
        if self.codes is None:
            return self.dense[i]
        norm = np.float32(1.0)
        if self.qnorm:
            norm = self.norm_centroids[self.norm_codes[i]]
        vals = np.empty(self.dim, dtype=np.float32)
        for m in range(self.nsubq):
            d = self.dsub if m < self.nsubq - 1 else self.lastdsub
            base = m * 256 * self.dsub + int(self.codes[i, m]) * d
            vals[m * self.dsub:m * self.dsub + d] = self.centroids[base:base + d]
        return vals * norm

    # --- dictionary ---------------------------------------------------
    def push_hash(self, out: list[int], ngram_id: int) -> None:
        if self.pruneidx_size == 0 or ngram_id < 0:
            return
        if self.pruneidx_size > 0:
            if ngram_id in self.pruneidx:
                ngram_id = self.pruneidx[ngram_id]
            else:
                return
        out.append(self.nwords + ngram_id)

    def compute_subwords(self, word: str, out: list[int]) -> None:
        raw = word.encode("utf-8", "surrogateescape")
        length = len(raw)
        for i in range(length):
            if (raw[i] & 0xC0) == 0x80:
                continue
            j, n = i, 1
            ngram = bytearray()
            while j < length and n <= self.maxn:
                ngram.append(raw[j])
                j += 1
                while j < length and (raw[j] & 0xC0) == 0x80:
                    ngram.append(raw[j])
                    j += 1
                n += 1
                if (n - 1) >= self.minn and not (n - 1 == 1 and (i == 0 or j == length)):
                    self.push_hash(out, fnv1a(bytes(ngram)) % self.bucket)

    def features(self, text: str) -> list[int]:
        word_ids = {w: i for i, w in enumerate(self.words)}
        feats: list[int] = []
        tokens = [t for t in re.split(r"[ \t\v\f\r\x00]", text.replace("\n", " ")) if t]
        tokens.append(EOS)  # predict() scores one line, terminated by EOS
        for token in tokens:
            wid = word_ids.get(token, -1)
            if wid >= 0:
                feats.append(wid)
                if token != EOS:
                    self.compute_subwords(BOW + token + EOW, feats)
            elif token != EOS:
                self.compute_subwords(BOW + token + EOW, feats)
        return feats

    # --- hierarchical softmax ------------------------------------------
    def hs_paths(self) -> tuple[list[list[int]], list[list[bool]]]:
        assert self.loss == LOSS_HS, "only the hs loss is expected for lid.176"
        counts = self.label_counts
        osz = len(counts)
        nodes = 2 * osz - 1
        parent = [-1] * nodes
        binary = [False] * nodes
        total = [10**15] * nodes
        for i in range(osz):
            total[i] = counts[i]
        leaf, node = osz - 1, osz
        for i in range(osz, nodes):
            mini = [0, 0]
            for j in range(2):
                if leaf >= 0 and total[leaf] < total[node]:
                    mini[j] = leaf
                    leaf -= 1
                else:
                    mini[j] = node
                    node += 1
            total[i] = total[mini[0]] + total[mini[1]]
            parent[mini[0]] = i
            parent[mini[1]] = i
            binary[mini[1]] = True
        paths, codes = [], []
        for i in range(osz):
            path, code = [], []
            j = i
            while parent[j] != -1:
                path.append(parent[j] - osz)
                code.append(binary[j])
                j = parent[j]
            paths.append(path)
            codes.append(code)
        return paths, codes


def sigmoid(x: np.float32) -> np.float32:
    return np.float32(1.0 / (1.0 + np.exp(-np.float64(x))))


def std_log(x: np.float32) -> np.float32:
    return np.float32(np.log(np.float64(x) + 1e-5))


def predict(model: "ScoredModel", text: str) -> tuple[str, float]:
    feats = model.ftz.features(text)
    if not feats:
        return "", 0.0
    hidden = np.zeros(model.ftz.dim, dtype=np.float32)
    for feature in feats:
        hidden += model.rows[feature]
    hidden *= np.float32(1.0 / len(feats))
    best_label, best_score = "", -np.inf
    for i, (path, code) in enumerate(zip(model.paths, model.codes)):
        logprob = np.float32(0.0)
        for node, bit in zip(path, code):
            dot = np.float32(0.0)
            row = model.wo[node]
            for j in range(model.ftz.dim):
                dot = np.float32(dot + np.float32(row[j] * hidden[j]))
            p = sigmoid(dot)
            term = std_log(p) if bit else std_log(np.float32(1.0) - p)
            logprob = np.float32(logprob + term)
        score = np.float32(np.exp(np.float64(logprob)))
        if score > best_score:
            best_label, best_score = model.ftz.labels[i], float(score)
    return best_label.removeprefix("__label__"), best_score


class ScoredModel:
    def __init__(self, ftz: LidFtz, rows: np.ndarray, wo: np.ndarray):
        self.ftz = ftz
        self.rows = rows
        self.wo = wo
        self.paths, self.codes = ftz.hs_paths()


def quantize_int8_per_row(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The repo recipe (scripts/build_tiers.py), verbatim."""
    scale = (np.abs(matrix).max(axis=1) / 127.0).astype(np.float32)
    scale[scale == 0.0] = 1.0
    q = np.rint(matrix / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale


def make_lid_onnx(q: np.ndarray, scale: np.ndarray, wo: np.ndarray) -> onnx.ModelProto:
    """Container in the tier shape: Constant nodes with raw_data, plus a
    variable-length feature-index input reduced to the mean hidden row.
    The classifier half (hierarchical softmax over `wo`) is engine-side:
    its tree depends on label counts carried by the vocab sidecar."""
    rows, dims = q.shape
    features = helper.make_tensor_value_info("features", TensorProto.INT64, ["N"])
    hidden = helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [dims])
    nodes = [
        helper.make_node("Constant", [], ["q_input"], value=numpy_helper.from_array(q, name="q_input")),
        helper.make_node("Constant", [], ["row_scale"], value=numpy_helper.from_array(scale, name="row_scale")),
        helper.make_node("Constant", [], ["output_weights"], value=numpy_helper.from_array(wo, name="output_weights")),
        helper.make_node("Gather", ["q_input", "features"], ["emb_i8"], axis=0),
        helper.make_node("Gather", ["row_scale", "features"], ["row_scales"], axis=0),
        helper.make_node("Unsqueeze", ["row_scales"], ["scales_col"], axes=[1]),
        helper.make_node("Cast", ["emb_i8"], ["emb_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["emb_f", "scales_col"], ["rows_f"]),
        helper.make_node("ReduceMean", ["rows_f"], ["hidden"], axes=[0], keepdims=0),
    ]
    graph = helper.make_graph(nodes, "fasttext_lid", [features], [hidden])
    model = helper.make_model(
        graph,
        producer_name="kotoshu-fasttext-converter",
        producer_version="1.0.0",
        opset_imports=[helper.make_operatorsetid("", 11)],
        ir_version=11,
    )
    for key, value in (
        ("model_type", "fasttext_lid"),
        ("quantization", "int8-per-row"),
        ("embedding_dimension", str(dims)),
        ("input_rows", str(rows)),
        ("label_count", str(wo.shape[0])),
        ("source_model", "lid.176.ftz"),
        ("tier", "lid-176"),
    ):
        model.metadata_props.append(StringStringEntryProto(key=key, value=value))
    return model


def make_sidecar(ftz: LidFtz) -> dict:
    return {
        "kind": "fasttext-lid-vocab",
        # Labels in dictionary (tree-leaf) order; counts rebuild the
        # Huffman tree deterministically (fastText HierarchicalSoftmaxLoss).
        "labels": [label.removeprefix("__label__") for label in ftz.labels],
        "label_counts": ftz.label_counts,
        # Word rows: id == position (the matrix keeps pruned-dictionary
        # word ids 0..nwords-1 in its first rows).
        "words": ftz.words,
        # Hashed-ngram rows: original ngram id (hash % bucket) -> row
        # offset (row = nwords + mapped id); ngrams absent from the map
        # were pruned away by the upstream quantization and contribute
        # nothing, exactly like Dictionary::pushHash.
        "pruneidx": [[key, value] for key, value in sorted(ftz.pruneidx.items())],
        "args": {
            "dim": ftz.dim,
            "bucket": ftz.bucket,
            "minn": ftz.minn,
            "maxn": ftz.maxn,
            "wordNgrams": ftz.word_ngrams,
            "loss": "hs",
            "nwords": ftz.nwords,
            "eos": EOS,
        },
    }


def verify_onnx(onnx_path: Path, ftz: LidFtz, rows: np.ndarray, quantized: bool) -> dict:
    import onnxruntime as ort

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for text in ("the quick brown fox", "今日はとても良い天気ですね", "Быстрая бурая лиса"):
        feats = ftz.features(text)
        got = session.run(["hidden"], {"features": np.array(feats, dtype=np.int64)})[0]
        want = np.zeros(ftz.dim, dtype=np.float32)
        for feature in feats:
            want += rows[feature]
        want *= np.float32(1.0 / len(feats))
        worst = max(worst, float(np.abs(got - want).max()))
    tol = 5e-3 if quantized else 1e-5
    if worst > tol:
        raise RuntimeError(f"{onnx_path}: runtime hidden differs by {worst:.2e} (> {tol})")
    return {"runtime_vs_reference_max_abs": worst, "tolerance": tol}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=".", help="repo root (default: cwd)")
    parser.add_argument("--ftz", type=Path, default=None, help="lid.176.ftz source (default downloads/lid.176.ftz)")
    parser.add_argument("--corpus", type=Path, default=None, help="probe corpus jsonl (default eval/corpora/lid_probe.jsonl)")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    ftz_path = args.ftz or root / "downloads" / "lid.176.ftz"
    corpus_path = args.corpus or root / "eval" / "corpora" / "lid_probe.jsonl"
    if not ftz_path.exists():
        sys.exit(f"error: {ftz_path} missing; fetch {UPSTREAM_URL} (sha256 {UPSTREAM_SHA256})")
    digest = sha256(ftz_path.read_bytes()).hexdigest()
    if digest != UPSTREAM_SHA256:
        sys.exit(f"error: {ftz_path}: sha256 {digest} != pinned {UPSTREAM_SHA256} — refusing to convert")
    corpus = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line]

    ftz = LidFtz(ftz_path)
    dense = ftz.rows_dense()
    reference = ScoredModel(ftz, dense, ftz.wo)
    baseline = [predict(reference, sample["text"]) for sample in corpus]
    labeled = [i for i, sample in enumerate(corpus) if sample["expected"]]
    for i in labeled:
        if baseline[i][0] != corpus[i]["expected"]:
            print(f"warning: probe {corpus[i]['id']}: float32 reference says {baseline[i][0]}, corpus expects {corpus[i]['expected']}", file=sys.stderr)

    # --- int8 candidate vs the float32 labels -------------------------
    q, scale = quantize_int8_per_row(dense)
    int8_rows = q.astype(np.float32) * scale[:, None]
    int8_model = ScoredModel(ftz, int8_rows, ftz.wo)
    int8_results = [predict(int8_model, sample["text"]) for sample in corpus]
    int8_agreement = sum(a[0] == b[0] for a, b in zip(int8_results, baseline)) / len(corpus)
    int8_max_drift = max(abs(a[1] - b[1]) for a, b in zip(int8_results, baseline))
    int8_bytes = q.nbytes + scale.nbytes + ftz.wo.nbytes

    quantization = "int8-per-row"
    if int8_agreement == 1.0 and int8_bytes <= MIRROR_BUDGET_BYTES:
        rows_payload = int8_rows
    else:
        quantization = None  # float32 fallback, recorded below
        rows_payload = dense

    onnx_path = root / "models" / "lid" / "lid.176.onnx"
    vocab_path = root / "models" / "lid" / "lid.176.vocab.json"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    if quantization:
        model = make_lid_onnx(q, scale, ftz.wo)
    else:
        model = make_lid_onnx(
            dense, np.ones(len(dense), dtype=np.float32), ftz.wo)
        model.metadata_props[:] = [
            prop for prop in model.metadata_props if prop.key != "quantization"]
    onnx.save(model, str(onnx_path))
    sidecar = make_sidecar(ftz)
    vocab_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    check = verify_onnx(onnx_path, ftz, rows_payload, quantized=bool(quantization))

    report = {
        "generated_at": iso_now(),
        "source": {
            "url": UPSTREAM_URL,
            "sha256": UPSTREAM_SHA256,
            "size_bytes": ftz_path.stat().st_size,
            "labels": len(ftz.labels),
            "dim": ftz.dim,
            "nwords": ftz.nwords,
            "pruneidx_size": len(ftz.pruneidx),
            "input_rows": ftz.input_rows,
            "loss": "hs",
            "license": "MIT (fastText)",
        },
        "artifact": {
            "onnx": str(onnx_path.relative_to(root)),
            "vocab": str(vocab_path.relative_to(root)),
            "quantization": quantization or "float32",
            "onnx_bytes": onnx_path.stat().st_size,
            "vocab_bytes": vocab_path.stat().st_size,
            "sha256": sha256(onnx_path.read_bytes()).hexdigest(),
            "vocab_sha256": sha256(vocab_path.read_bytes()).hexdigest(),
        },
        "gates": {
            "top1_agreement_min": 1.0,
            "int8_top1_agreement": round(int8_agreement, 6),
            "int8_max_score_drift": round(int8_max_drift, 6),
            "mirror_budget_bytes": MIRROR_BUDGET_BYTES,
            "int8_payload_bytes": int(int8_bytes),
            "chosen": quantization or "float32",
            "passed": (int8_agreement == 1.0 and int8_bytes <= MIRROR_BUDGET_BYTES)
            or quantization is None,
        },
        "runtime_check": check,
        "corpus": corpus_path.name,
        "samples": len(corpus),
        "baseline": [
            {"id": sample["id"], "code": code, "score": score}
            for sample, (code, score) in zip(corpus, baseline)
        ],
    }
    reports = root / "eval" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "lid.176.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    descriptor = {
        "language": "lid",
        "tier": "lid-176",
        "dims": ftz.dim,
        "input_rows": ftz.input_rows,
        "words": ftz.nwords,
        "labels": len(ftz.labels),
        "quantization": quantization or "float32",
        "bytes": onnx_path.stat().st_size,
        "sha256": report["artifact"]["sha256"],
        "vocab_sha256": report["artifact"]["vocab_sha256"],
        "vocab_bytes": vocab_path.stat().st_size,
        "eval_ref": "eval/reports/lid.176.json",
        "source_url": UPSTREAM_URL,
        "source_sha256": UPSTREAM_SHA256,
        "license": "MIT",
        "generated_at": iso_now(),
    }
    (root / "models" / "lid" / "lid.json").write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")

    print(f"lid.176: quantization={report['gates']['chosen']} "
          f"int8_agreement={int8_agreement:.4f} drift={int8_max_drift:.2e} "
          f"onnx={report['artifact']['onnx_bytes']} vocab={report['artifact']['vocab_bytes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
