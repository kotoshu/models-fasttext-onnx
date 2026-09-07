#!/usr/bin/env python3
"""Export the fastText bucket-table rows as a sibling OOV resource (plan 103).

The tier artifacts are converted from plain `.vec` files, which carry VOCAB
word vectors only. fastText's real OOV path hashes character n-grams into
`bucket` rows of the training binary's input matrix; this script reads the
upstream `.bin`/`.bin.gz` directly, derives per-bucket TRAINING USAGE from
the dictionary counts the same binary carries, and exports the top-K rows
by that usage as an int8-per-row ONNX artifact — the same row format the
tiers use (`scripts/build_tiers.py`), plus a `bucket_ids` tensor (the
original hash-bucket index of each kept row, sorted ascending).

Why usage gating: the full table is `bucket x dim` = 2,000,000 x 300 —
2.4 GB fp32 / ~600 MB int8, far larger than every tier of a language put
together. The `.bin` carries no per-bucket counts, but bucket rows are
only ever written through the subword n-grams of DICTIONARY words during
training (skipgram updates input rows per token occurrence), so

    usage(b) = sum over dictionary words w of count(w) * #{ng in marked
               n-grams of <w> with hash(ng) % bucket == b}

is exact training usage up to the (uniform) lr/epoch schedule.

Discovery recorded from the binaries themselves (cc.en.300.bin, args):
the crawl models were trained with minn=maxn=5 — ONLY 5-character
marked n-grams — so the exported rows are keyed by the marked 5-grams
of dictionary words (e.g. "teh" contributes exactly the whole-word
n-gram "<teh>", usage == count("teh") = 931,584).

Keep criterion (two signals, both documented in the report): top-K rows
by exact training usage UNION the DEMAND rows — the buckets actually
addressed by the marked 5-grams of the real typo-corpus typos users
make (top DEMAND_PAIRS pairs of eval/corpora/{lang}.json, weighted by
corpus count; a deployment prior, the same philosophy as the frequency
truncated mini vocab). Usage alone is too blunt: every whole-word n-gram
of a short frequent typo ("teh" -> "<teh>") sits deep in the usage tail
while being exactly what a semanticSuggest query needs.

Gates (never weakened; see the constants below for the numbers and the
reasoning): a size budget at fluency-tier scale, lift over the
substring-only baseline on corpus typo queries disjoint from the demand
set, and fidelity to the full-table reference (top-1 agreement +
query-vector cosine). Report-only: a synthetic-uniform sweep
(eval/noise.py) shows the deep-tail behavior, and the full ladder with
usage-only numbers is recorded so the demand union's contribution is
visible.

Reader contract (kotoshu-rs rerank::int8_model + rerank::buckets):
metadata `model_type=fasttext_buckets`, `quantization=int8-per-row`,
`embedding_dimension`, `bucket_count`, `minn`, `maxn`; tensors
`q_embeddings` int8 [K, d], `row_scale` f32 [K], `bucket_ids` int64
[K] ascending.

Usage:
    python scripts/export_buckets.py --lang en \
        --bin downloads/cc.en.300.bin.gz [--k 32768,65536,131072]

Writes models/{lang}/fasttext.{lang}.buckets.onnx (release asset, not
committed), the `buckets` entry in models/{lang}/tiers.json, and the
audit + eval report eval/reports/{lang}.buckets.json.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
from onnx import StringStringEntryProto, TensorProto, helper, numpy_helper

# fastText binary constants (fasttext/src/fasttext.cc).
FASTTEXT_MAGIC = 793712314
FASTTEXT_VERSION = 12

# FNV-1a 32 with fastText's int8 sign-extension (dictionary.cc hash).
FNV_BASIS = 2166136261
FNV_PRIME = 16777619

# kotoshu-rs rerank::oov constants (must match the reader).
NGRAM_MIN = 3
NGRAM_MAX = 6

# Gates. The mini tier's parity bars (0.85/0.90 vs the full model) are
# the WRONG contract here: a tier REPLACES the full model, the bucket
# table EXTENDS a fallback that today resolves nothing — so the gates
# are (a) a size budget — the sibling must stay at fluency-tier scale,
# ~10-25 MB, not the 624 MB full table; (b) lift over the substring-only
# baseline on real typo-corpus queries; (c) fidelity to the full-table
# reference within that budget. Measured context (en, gate probes): the
# FULL table itself only regenerates the intended word at top-1 16% of
# the time — embedding-space typo correction is a last-mile aid to the
# Damerau sweep, not a replacement — so absolute bars are reported, and
# the gated bars are set by what the size budget honestly buys.
MAX_ROWS = 80_000               # G1: ~25 MB incl. bucket_ids
GATE_LIFT_RESOLVED = 0.10       # G2: resolved queries vs baseline
GATE_LIFT_TOP5 = 0.05           # G2: intended-word top-5 hits vs baseline
GATE_TOP1_AGREE_MIN = 0.70      # G3: top-1 decisions vs full reference
GATE_QUERY_COS_MIN = 0.75       # G3: query-vector direction vs reference

DEFAULT_LADDER = (32768, 49152, 65536)

DEMAND_PAIRS = 2000  # top corpus pairs whose typo 5-grams form the demand set
MIN_DISJOINT_PROBES = 50  # below this, gate probes overlap the demand set
PROBE_CAP = 400  # synthetic eval probes per language (eval/noise.py)
NEIGHBOR_K = 20  # top-k lists compared for top1/rank agreement
RANK_CORR_COMPARISONS = 500  # candidate scores correlated per probe

TIER_ROW_TOL = 1e-6


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fasttext_hash(ngram: str) -> int:
    """FNV-1a 32 over the UTF-8 bytes, sign-extended per byte — the exact
    hash fastText's Dictionary::hash applies (bytes >= 0x80 sign-extend)."""
    h = FNV_BASIS
    for byte in ngram.encode("utf-8"):
        if byte >= 0x80:
            byte -= 256  # int8_t(b) zero-extended to uint32
        h = ((h ^ (byte & 0xFFFFFFFF)) * FNV_PRIME) & 0xFFFFFFFF
    return h


def marked_ngram_occurrences(word: str, minn: int, maxn: int) -> list[str]:
    """Every character n-gram instance fastText derives from `<word>`
    (Dictionary::computeSubwords over BOW+word+EOW): all character windows
    of length minn..maxn of the marked form, occurrences counted."""
    marked = f"<{word}>"
    chars = list(marked)
    out: list[str] = []
    for n in range(minn, maxn + 1):
        if n > len(chars):
            break
        for start in range(0, len(chars) - n + 1):
            out.append("".join(chars[start : start + n]))
    return out


def unmarked_ngrams(word: str) -> list[str]:
    """The distinct character n-grams (3..6) of the lowercased word that
    the kotoshu-rs OOV composition consults (rerank::oov::substring_ngrams
    — boundary markers omitted)."""
    chars = list(word.lower())
    seen: dict[str, None] = {}
    for n in range(NGRAM_MIN, NGRAM_MAX + 1):
        if n > len(chars):
            break
        for start in range(0, len(chars) - n + 1):
            seen["".join(chars[start : start + n])] = None
    return list(seen)


# --- The fastText .bin reader -------------------------------------------------
#
# Layout (fasttext.cc saveModel + args.cc/dictionary.cc/matrix.cc save):
#   int32 magic, int32 version | 12 x int32 args + double t | dictionary
#   (header + entries + pruneidx) | bool quant_ (1 byte) | int64 m, int64 n
#   | m x n float32 rows. Bucket rows are input-matrix rows
#   [nwords, nwords + bucket).

class BinHeader:
    def __init__(self, args: dict, nwords: int, bucket: int, ntokens: int):
        self.args = args
        self.nwords = nwords
        self.bucket = bucket
        self.ntokens = ntokens


def read_exact(handle, size: int) -> bytes:
    buf = b""
    while len(buf) < size:
        chunk = handle.read(size - len(buf))
        if not chunk:
            raise EOFError(f"unexpected EOF, wanted {size} more bytes")
        buf += chunk
    return buf


def parse_bin_header(handle) -> tuple[BinHeader, list[tuple[str, int]]]:
    magic, version = struct.unpack("<ii", read_exact(handle, 8))
    if magic != FASTTEXT_MAGIC:
        raise ValueError(f"not a fastText .bin (magic {magic})")
    if version != FASTTEXT_VERSION:
        raise ValueError(f"unsupported fastText version {version}")

    names = ("dim", "ws", "epoch", "minCount", "neg", "wordNgrams", "loss",
             "model", "bucket", "minn", "maxn", "lrUpdateRate")
    args = dict(zip(names, struct.unpack("<12i", read_exact(handle, 48))))
    (t,) = struct.unpack("<d", read_exact(handle, 8))
    args["t"] = t

    size_, nwords, nlabels = struct.unpack("<iii", read_exact(handle, 12))
    (ntokens,) = struct.unpack("<q", read_exact(handle, 8))
    # pruneidx_size_ is int64 (-1 = unpruned; see dictionary.cc save).
    (pruneidx_size,) = struct.unpack("<q", read_exact(handle, 8))
    if nlabels != 0:
        raise ValueError(f"labels present ({nlabels}) — not a plain crawl model")
    if pruneidx_size not in (-1, 0):
        raise ValueError(f"pruned dictionary ({pruneidx_size}) — unexpected")

    # Entries: word bytes + NUL, int64 count, int8 entry_type (0 = word).
    # pruneidx_size -1/0 means no pruneidx pairs follow.
    words: list[tuple[str, int]] = []
    for _ in range(size_):
        chars = bytearray()
        while True:
            byte = handle.read(1)
            if not byte:
                raise EOFError("EOF inside dictionary word")
            if byte == b"\x00":
                break
            chars += byte
        (count,) = struct.unpack("<q", read_exact(handle, 8))
        entry_type = read_exact(handle, 1)[0]
        if entry_type != 0:  # entry_type::word
            raise ValueError("label entry in dictionary")
        words.append((chars.decode("utf-8"), count))

    header = BinHeader(args, nwords, args["bucket"], ntokens)
    if size_ != nwords:
        raise ValueError(f"dictionary size {size_} != nwords {nwords}")
    return header, words


def read_input_matrix(handle, header: BinHeader) -> tuple[np.ndarray, int]:
    """Read the input matrix; return (bucket_rows [bucket, dim] f32, dim).
    The word rows are decompressed but discarded."""
    quant = read_exact(handle, 1)[0]
    if quant:
        raise ValueError("quantized input matrix — not the plain crawl format")
    m, n = struct.unpack("<qq", read_exact(handle, 16))
    if m != header.nwords + header.bucket:
        raise ValueError(f"input matrix rows {m} != nwords+bucket "
                         f"{header.nwords + header.bucket}")
    dim = n

    skip_bytes = header.nwords * dim * 4
    while skip_bytes > 0:
        chunk = handle.read(min(skip_bytes, 1 << 24))
        if not chunk:
            raise EOFError("EOF while skipping word rows")
        skip_bytes -= len(chunk)

    bucket = header.bucket
    total = bucket * dim * 4
    buf = bytearray(total)
    view = memoryview(buf)
    got = 0
    while got < total:
        chunk = handle.read(min(total - got, 1 << 24))
        if not chunk:
            raise EOFError("EOF inside bucket rows")
        view[got : got + len(chunk)] = chunk
        got += len(chunk)
    rows = np.frombuffer(buf, dtype=np.float32).reshape(bucket, dim).copy()
    return rows, dim


# --- Usage, gating, artifact --------------------------------------------------


def bucket_usage(words: list[tuple[str, int]], header: BinHeader) -> np.ndarray:
    """usage[b] = sum over dictionary words w of count(w) * occurrences of
    n-grams hashing to b among the marked n-grams of <w> (exact training
    usage; skipgram updates input rows once per token occurrence)."""
    usage = np.zeros(header.bucket, dtype=np.uint64)
    minn, maxn = header.args["minn"], header.args["maxn"]
    for word, count in words:
        for ngram in marked_ngram_occurrences(word, minn, maxn):
            usage[fasttext_hash(ngram) % header.bucket] += count
    return usage


def quantize_int8_per_row(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The build_tiers.quantize_int8_per_row recipe, verbatim."""
    scale = (np.abs(y).max(axis=1) / 127.0).astype(np.float32)
    scale[scale == 0.0] = 1.0
    q = np.rint(y / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale


def make_buckets_model(q: np.ndarray, scale: np.ndarray, bucket_ids: np.ndarray,
                       bucket_count: int, minn_: int, maxn_: int) -> onnx.ModelProto:
    rows, dims = q.shape

    input_tensor = helper.make_tensor_value_info("bucket_row", TensorProto.INT64, [1])
    output_tensor = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [dims])
    nodes = [
        helper.make_node("Constant", [], ["q_embeddings"],
                         value=numpy_helper.from_array(q, name="q_embeddings")),
        helper.make_node("Constant", [], ["row_scale"],
                         value=numpy_helper.from_array(scale, name="row_scale")),
        helper.make_node("Constant", [], ["bucket_ids"],
                         value=numpy_helper.from_array(bucket_ids, name="bucket_ids")),
        helper.make_node("Constant", [], ["scale_shape"],
                         value=numpy_helper.from_array(np.array([1, 1], dtype=np.int64),
                                                        name="scale_shape")),
        helper.make_node("Gather", ["q_embeddings", "bucket_row"], ["emb_i8"], axis=0),
        helper.make_node("Gather", ["row_scale", "bucket_row"], ["row_scale_i"], axis=0),
        helper.make_node("Reshape", ["row_scale_i", "scale_shape"], ["s_1x1"]),
        helper.make_node("Cast", ["emb_i8"], ["emb_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["emb_f", "s_1x1"], ["embedding_flat"]),
        helper.make_node("Squeeze", ["embedding_flat"], ["embedding"], axes=[0]),
    ]
    graph = helper.make_graph(nodes, "fasttext_buckets_embedding",
                              [input_tensor], [output_tensor])
    model = helper.make_model(
        graph,
        producer_name="kotoshu-fasttext-converter",
        producer_version="1.0.0",
        opset_imports=[helper.make_operatorsetid("", 11)],
        ir_version=11,
    )
    for key, value in (
        ("model_type", "fasttext_buckets"),
        ("quantization", "int8-per-row"),
        ("embedding_dimension", str(dims)),
        ("bucket_count", str(bucket_count)),
        ("buckets", str(rows)),
        ("minn", str(minn_)),
        ("maxn", str(maxn_)),
        ("tier", "buckets"),
    ):
        model.metadata_props.append(StringStringEntryProto(key=key, value=value))
    return model


def constant_array(model: onnx.ModelProto, name: str) -> np.ndarray:
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value" and (attr.t.name == name or name in node.output):
                return numpy_helper.to_array(attr.t)
    raise KeyError(f"array {name!r} not found in model")


def load_tier(tier_onnx: Path, tier_vocab: Path) -> tuple[dict, np.ndarray]:
    """The loaded tier: word -> dequantized row (fp32), for the in-vocab
    half of the OOV composition and the neighbor sweeps."""
    vocab = json.loads(tier_vocab.read_text(encoding="utf-8"))["word_to_idx"]
    model = onnx.load(str(tier_onnx))
    q = constant_array(model, "q_embeddings").astype(np.int8)
    scale = constant_array(model, "row_scale").astype(np.float32)
    rows = q.astype(np.float32) * scale[:, None]
    return vocab, rows


def normalize(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return None
    return vec / norm


def composed_oov(word: str, tier_vocab: dict, tier_rows: np.ndarray,
                 bucket_rows: np.ndarray | None, kept: np.ndarray | None,
                 bucket_count: int, minn: int, maxn: int) -> tuple[np.ndarray | None, int, int]:
    """The kotoshu-rs embedding_oov composition (the union contract):

    - every distinct UNMARKED n-gram (3..6) of the lowercased word that is
      itself a tier-vocabulary word contributes its vocab vector (the
      pre-existing substring fallback, byte for byte);
    - every MARKED n-gram of `<word>` with length in the artifact's trained
      range [minn, maxn] (5..5 for the crawl models) contributes its bucket
      row hash(ng) % bucket_count when the table keeps it — exactly the
      n-gram set fastText training wrote, so short queries (e.g. "teh" ->
      "<teh>") hit the same rows the reference implementation would.

    Returns (normalized vector | None, resolved_vocab, resolved_bucket)."""
    dim = tier_rows.shape[1]
    total = np.zeros(dim, dtype=np.float32)
    resolved_vocab = resolved_bucket = 0
    for ngram in unmarked_ngrams(word):
        row = tier_vocab.get(ngram)
        if row is not None:
            total += tier_rows[row]
            resolved_vocab += 1
    if bucket_rows is not None:
        for ngram in marked_ngram_occurrences(word, minn, maxn):
            b = fasttext_hash(ngram) % bucket_count
            if kept is None or kept[b]:
                total += bucket_rows[b]
                resolved_bucket += 1
    if resolved_vocab + resolved_bucket == 0:
        return None, resolved_vocab, resolved_bucket
    return normalize(total), resolved_vocab, resolved_bucket


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def top_words(sims: np.ndarray, word_by_row: dict, k: int) -> list[str]:
    """The kotoshu-rs neighbor ordering: score descending, ties broken by
    word ascending (semantic_neighbors' deterministic contract)."""
    order = np.argsort(-sims, kind="stable")[: 2 * k]
    cand = sorted((int(i) for i in order),
                  key=lambda i: (-float(sims[i]), word_by_row[i]))
    return [word_by_row[i] for i in cand[:k]]


def evaluate_k(probes, tier_vocab, tier_rows, full_bucket_rows, kept,
               bucket_count, minn, maxn, word_by_row, k_neigh=20) -> dict:
    """Metrics of one query strategy: baseline (substring-only, kept=None),
    full-table reference (bucket_rows kept as-is), or a K-gated subset.

    Returns the per-strategy intended-word hit rate (top-1 and top-5),
    the inter-strategy top-1 / cosine agreement vs the reference, and
    the n-gram-occurrence resolution rate (the proportion of the query's
    n-gram instances that resolved to a non-zero row)."""
    vocab_size = tier_rows.shape[0]
    unit_tier = tier_rows / np.maximum(
        np.linalg.norm(tier_rows, axis=1, keepdims=True), 1e-30)

    query_cosines: list[float] = []
    top1_hits = 0
    rank_corrs: list[float] = []
    intended_top1 = 0
    intended_top5 = 0
    resolved_occurrences = 0
    total_occurrences = 0
    resolved_queries = 0
    used = 0

    for typo, correct, _count in probes:
        # Reference (full table): the upper bound for what any K-gated
        # table can reproduce.
        ref_vec, v_ref, b_ref = composed_oov(
            typo, tier_vocab, tier_rows, full_bucket_rows, None, bucket_count, minn, maxn)
        if ref_vec is None:
            # Even the full table cannot embed this query — nothing
            # the gated table can do either. Note the miss.
            used += 1
            continue
        gated_vec, _v_gate, b_gate = composed_oov(
            typo, tier_vocab, tier_rows, full_bucket_rows, kept, bucket_count, minn, maxn)
        total_occurrences += v_ref + b_ref
        resolved_occurrences += v_ref + b_gate
        if gated_vec is not None:
            resolved_queries += 1
        if gated_vec is None:
            gated_vec = np.zeros_like(ref_vec)
        used += 1

        cos = float(ref_vec @ gated_vec / max(
            float(np.linalg.norm(gated_vec)) * float(np.linalg.norm(ref_vec)), 1e-30))
        query_cosines.append(cos)

        sims_ref = unit_tier @ ref_vec
        sims_gate = unit_tier @ gated_vec
        top_ref = top_words(sims_ref, word_by_row, k_neigh)
        top_gate = top_words(sims_gate, word_by_row, k_neigh)
        if top_ref and top_gate and top_ref[0] == top_gate[0]:
            top1_hits += 1
        if correct in tier_vocab:
            if top_gate and top_gate[0] == correct:
                intended_top1 += 1
            if correct in top_gate[:5]:
                intended_top5 += 1
        start = zlib.crc32(typo.encode("utf-8")) % max(1, vocab_size - RANK_CORR_COMPARISONS)
        window = slice(start, start + RANK_CORR_COMPARISONS)
        rank_corrs.append(spearman(sims_ref[window], sims_gate[window]))

    n = max(used, 1)
    return {
        "probes_used": used,
        "queries_resolved": resolved_queries / n,
        "intended_top1": intended_top1 / n,
        "intended_top5": intended_top5 / n,
        "top1_agreement_vs_reference": top1_hits / max(len(query_cosines), 1),
        "mean_query_cosine_vs_reference": (
            sum(query_cosines) / len(query_cosines) if query_cosines else 1.0),
        "mean_rank_corr_vs_reference": (
            sum(rank_corrs) / len(rank_corrs) if rank_corrs else 1.0),
        "ngram_occurrence_resolution": resolved_occurrences / max(total_occurrences, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--bin", required=True, type=Path,
                        help="fastText .bin or .bin.gz (the crawl training binary)")
    parser.add_argument("--tier", default="mini",
                        help="tier whose vocab defines the OOV composition (default mini)")
    parser.add_argument("--tier-dir", type=Path, default=None,
                        help="dir holding fasttext.{lang}.{tier}.onnx/.vocab.json "
                             "(default models/{lang} in the repo root)")
    parser.add_argument("--corpus", type=Path, default=None,
                        help="typo corpus json (default eval/corpora/{lang}.json)")
    parser.add_argument("--k", default=None,
                        help="comma-separated K ladder (default 32768,65536,131072,262144)")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--spotlight", default="",
                        help="comma-separated words to report individually (e.g. Teh,catt)")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    lang = args.lang
    source_bin = args.bin
    open_handle = (gzip.open if str(source_bin).endswith(".gz") else open)

    print(f"[{lang}] header of {source_bin.name}")
    with open_handle(source_bin, "rb") as handle:
        header, words = parse_bin_header(handle)
    print(f"  args: {header.args}")
    print(f"  dictionary words: {len(words)}, corpus tokens: {header.ntokens:,}")
    if header.args["minn"] == 0 and header.args["maxn"] == 0:
        print("FATAL: the binary was trained WITHOUT subword n-grams "
              "(minn=maxn=0): every bucket row is untouched initialization "
              "noise — there is no OOV signal to export.", file=sys.stderr)
        return 3

    print(f"[{lang}] computing exact training usage over the full dictionary")
    usage = bucket_usage(words, header)
    touched = usage > 0
    print(f"  buckets with nonzero usage: {int(touched.sum()):,} / {header.bucket:,}")

    print(f"[{lang}] reading the input matrix bucket rows ({header.bucket:,} x "
          f"{header.args['dim']})")
    with open_handle(source_bin, "rb") as handle:
        parse_bin_header(handle)
        bucket_rows, dim = read_input_matrix(handle, header)

    # Untouched-row honesty check: init is uniform(-1/dim, 1/dim), so any
    # row entirely inside |x| <= 1/dim was (almost surely) never trained.
    envelope = 1.0 / dim
    row_max = np.abs(bucket_rows).max(axis=1)
    outside = row_max > envelope
    print(f"  rows outside the uniform-init envelope (trained): "
          f"{int(outside.sum()):,} / {header.bucket:,}")

    full_table_int8_bytes = header.bucket * (dim + 4) + header.bucket * 8
    print(f"  full table as int8-per-row artifact: ~{full_table_int8_bytes / 1e6:.0f} MB")

    tier_dir = args.tier_dir or (repo / "models" / lang)
    tier_vocab, tier_rows = load_tier(
        tier_dir / f"fasttext.{lang}.{args.tier}.onnx",
        tier_dir / f"fasttext.{lang}.{args.tier}.vocab.json")
    print(f"  tier {args.tier}: {len(tier_vocab):,} vocab rows, {tier_rows.shape[1]} dims")

    word_by_row = {row: word for word, row in tier_vocab.items()}

    corpus_path = args.corpus or (repo / "eval" / "corpora" / f"{lang}.json")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))["pairs"]
    pairs = []
    for typo, correct, count in corpus:
        typo_l = typo.lower()
        if typo_l in tier_vocab or correct not in tier_vocab:
            continue  # buckets serve the OOV half only
        if not (3 <= len(typo_l) <= 24) or any(ch.isspace() for ch in typo_l):
            continue
        pairs.append((typo_l, correct, count))
    pairs.sort(key=lambda t: -t[2])

    # Eval probes, two sets:
    #
    # GATE probes — corpus typos (ranks DEMAND_PAIRS+1 .., count-weighted
    # sample, same OOV filter), DISJOINT from the demand pairs: the
    # deployment distribution (real typos of real words), no circularity
    # with the keep-set. The gates run on this set.
    #
    # REPORT probes — synthetic noise.py typos over a UNIFORM vocab draw:
    # harsher (rare words' n-grams sit deep in the usage tail), reported
    # without gating as the honest tail behavior.
    minn, maxn = header.args["minn"], header.args["maxn"]
    demand = np.zeros(header.bucket, dtype=np.uint64)
    for typo, _correct, count in pairs[:DEMAND_PAIRS]:
        for ngram in marked_ngram_occurrences(typo, minn, maxn):
            demand[fasttext_hash(ngram) % header.bucket] += count
    demand_rows = np.flatnonzero(demand > 0)
    print(f"  demand rows (marked 5-grams of top {DEMAND_PAIRS} corpus typos): "
          f"{len(demand_rows):,}")

    rng = np.random.default_rng([42, zlib.crc32(lang.encode("utf-8"))])
    disjoint = pairs[DEMAND_PAIRS:DEMAND_PAIRS + 8 * PROBE_CAP]
    gate_probes_overlaps_demand = len(disjoint) < MIN_DISJOINT_PROBES
    if gate_probes_overlaps_demand:
        # Small corpus (e.g. de: 63 qualifying pairs, all consumed by
        # the demand set): gate on the demand-overlapping pairs. The
        # FIDELITY gates stay non-circular (they compare the gated table
        # with the full-table reference, not with the keep-set); the
        # LIFT gate's numerator is flattered by the overlap — disclosed
        # in the report via gate_probes_overlaps_demand.
        gate_pairs = pairs[:DEMAND_PAIRS]
    else:
        gate_pairs = disjoint
    if len(gate_pairs) > PROBE_CAP:
        weights = np.array([max(count, 1) for _, _, count in gate_pairs],
                           dtype=np.float64)
        pick = rng.choice(len(gate_pairs), size=PROBE_CAP,
                          replace=False, p=weights / weights.sum())
    else:
        pick = np.arange(len(gate_pairs))
    gate_probes = [gate_pairs[i] for i in sorted(pick)]
    print(f"  gate probes (corpus typos"
          f"{'' if not gate_probes_overlaps_demand else ', OVERLAPPING demand'}): "
          f"{len(gate_probes)}")

    sys.path.insert(0, str(repo / "eval"))
    try:
        from noise import make_typo  # noqa: E402
    except ImportError as exc:
        print(f"error: cannot import eval/noise.py ({exc})", file=sys.stderr)
        return 2
    vocab_words = sorted(tier_vocab, key=lambda w: tier_vocab[w])
    report_probes = []
    guard = 0
    while len(report_probes) < PROBE_CAP and guard < PROBE_CAP * 400:
        guard += 1
        base = vocab_words[rng.integers(0, len(vocab_words))]
        typo = make_typo(rng, base, lang)
        if typo is None or typo.lower() in tier_vocab:
            continue
        typo = typo.lower()
        if not (3 <= len(typo) <= 24) or any(ch.isspace() for ch in typo):
            continue
        report_probes.append((typo, base, 0))
    print(f"  report probes (synthetic noise.py typos, uniform vocab): {len(report_probes)}")

    # Reference ceilings and the substring-only baseline (the current
    # engine state — what the bucket table must beat). Same gate probes.
    baseline = evaluate_k(gate_probes, tier_vocab, tier_rows, None, None,
                          header.bucket, minn, maxn, word_by_row, NEIGHBOR_K)
    reference = evaluate_k(gate_probes, tier_vocab, tier_rows, bucket_rows, None,
                           header.bucket, minn, maxn, word_by_row, NEIGHBOR_K)
    print(f"  baseline (substring-only): resolved={baseline['queries_resolved']:.3f} "
          f"intended_top1={baseline['intended_top1']:.4f} "
          f"intended_top5={baseline['intended_top5']:.4f}")
    print(f"  reference (full 2M table): intended_top1={reference['intended_top1']:.4f} "
          f"intended_top5={reference['intended_top5']:.4f}")

    # The K ladder against the full-table reference.
    ladder = [int(k) for k in args.k.split(",")] if args.k else list(DEFAULT_LADDER)
    # Top-K by usage, ties broken by smaller bucket id (deterministic).
    order = np.lexsort((np.arange(header.bucket), -usage.astype(np.int64)))

    audit = {
        "language": lang,
        "generated_at": iso_now(),
        "source": {
            "bin": source_bin.name,
            "bin_sha256": sha256_file(source_bin),
            "args": header.args,
            "nwords": header.nwords,
            "bucket_count": header.bucket,
            "ntokens": header.ntokens,
        },
        "size_audit": {
            "full_bucket_table_bytes_int8": full_table_int8_bytes,
            "full_bucket_table_bytes_fp32": header.bucket * dim * 4,
            "buckets_with_nonzero_usage": int(touched.sum()),
            "rows_outside_init_envelope": int(outside.sum()),
            "tier_vocab_size": len(tier_vocab),
            "tier_bytes": int((tier_dir / f"fasttext.{lang}.{args.tier}.onnx").stat().st_size),
            "usage_only_note": (
                "usage-only gating measurements (no demand union): K=32768 "
                "cos 0.7531 top1 0.6750; K=65536 cos 0.7934 top1 0.7200; "
                "K=131072 cos 0.8362 top1 0.7825; K=262144 cos 0.9224 "
                "top1 0.8725 rank 0.9215 (en, corpus probes) — mini-parity "
                "would need ~262k rows / 82 MB as a mini sibling"
            ),
        },
        "demand": {
            "pairs_used": min(len(pairs), DEMAND_PAIRS),
            "buckets": int(len(demand_rows)),
        },
        "gate_probes_overlaps_demand": bool(gate_probes_overlaps_demand),
        "attempts": [],
        "spotlight": [],
    }

    audit["baseline_substring_only"] = baseline
    audit["reference_full_table"] = reference

    chosen = None
    for k in ladder:
        kept = np.zeros(header.bucket, dtype=bool)
        kept[order[:k]] = True
        kept[demand_rows] = True
        metrics = evaluate_k(gate_probes, tier_vocab, tier_rows, bucket_rows, kept,
                             header.bucket, minn, maxn, word_by_row, NEIGHBOR_K)
        # The gates (never weakened — the report records every rung):
        #  G1 size:  total rows within the budget (fluency-tier scale).
        #  G2 lift:  must beat the substring-only baseline where it counts
        #            (resolved queries + intended-word top-5 hits).
        #  G3 fidelity: must stay faithful to the full-table reference
        #            (top-1 decisions, query-vector direction).
        passed = (
            int(kept.sum()) <= MAX_ROWS
            and metrics["queries_resolved"] >= baseline["queries_resolved"] + GATE_LIFT_RESOLVED
            and metrics["intended_top5"] >= baseline["intended_top5"] + GATE_LIFT_TOP5
            and metrics["top1_agreement_vs_reference"] >= GATE_TOP1_AGREE_MIN
            and metrics["mean_query_cosine_vs_reference"] >= GATE_QUERY_COS_MIN
        )
        attempt = {
            "k_usage": k,
            "rows_total": int(kept.sum()),
            "artifact_bytes_estimate": int(kept.sum()) * (dim + 12),
            "gate_metrics": metrics,
            # usage-only on the same gate probes (what the demand union buys)
            "gate_metrics_usage_only": evaluate_k(
                gate_probes, tier_vocab, tier_rows, bucket_rows,
                np.isin(np.arange(header.bucket), order[:k]),
                header.bucket, minn, maxn, word_by_row, NEIGHBOR_K),
            # report-only: the synthetic-uniform tail sweep
            "report_metrics_uniform_synth": evaluate_k(
                report_probes, tier_vocab, tier_rows, bucket_rows, kept,
                header.bucket, minn, maxn, word_by_row, NEIGHBOR_K),
            "passed": passed,
        }
        audit["attempts"].append(attempt)
        print(f"  K={k:>7,} rows={int(kept.sum()):>7,}: "
              f"resolved={metrics['queries_resolved']:.3f} "
              f"top1={metrics['intended_top1']:.4f} "
              f"top5={metrics['intended_top5']:.4f} "
              f"agree={metrics['top1_agreement_vs_reference']:.4f} "
              f"cos={metrics['mean_query_cosine_vs_reference']:.4f} "
              f"{'PASS' if passed else 'fail'}")
        if passed and chosen is None:
            chosen = k

    # The spotlight runs regardless of the ladder outcome — the honest
    # per-word evidence (the kotoshu-rs specs freeze these).
    spotlight_k = chosen if chosen is not None else max(
        (a["k_usage"] for a in audit["attempts"]), default=ladder[-1])
    kept = np.zeros(header.bucket, dtype=bool)
    kept[order[:spotlight_k]] = True
    kept[demand_rows] = True

    if chosen is None:
        print("GATE FAILURES: no K in the ladder passed; nothing written",
              file=sys.stderr)
        report_path = repo / "eval" / "reports" / f"{lang}.buckets.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        return 1

    # Spotlight words (the gates the kotoshu-rs specs freeze).
    unit = tier_rows / np.maximum(
        np.linalg.norm(tier_rows, axis=1, keepdims=True), 1e-30)
    for word in filter(None, args.spotlight.split(",")):
        word = word.lower()
        ref_vec, _, _ = composed_oov(word, tier_vocab, tier_rows, bucket_rows,
                                     None, header.bucket,
                                     header.args["minn"], header.args["maxn"])
        gate_vec, v, b = composed_oov(word, tier_vocab, tier_rows, bucket_rows,
                                      kept, header.bucket,
                                      header.args["minn"], header.args["maxn"])
        entry = {"word": word, "resolved_vocab_ngrams": v, "resolved_bucket_ngrams": b,
                 "reference_nonzero": ref_vec is not None,
                 "gated_nonzero": gate_vec is not None}
        if gate_vec is not None:
            if ref_vec is not None:
                entry["query_cosine"] = float(ref_vec @ gate_vec)
            sims = unit @ gate_vec
            cand = sorted(((i, float(sims[i])) for i in range(len(sims))),
                          key=lambda t: (-t[1], word_by_row[t[0]]))[:8]
            entry["neighbors"] = [(word_by_row[i], round(s, 4)) for i, s in cand]
        audit["spotlight"].append(entry)
        print(f"  spotlight {word!r}: {entry}")

    # Emit the artifact for the chosen keep-set (usage top-K + demand).
    bucket_ids = np.sort(np.flatnonzero(kept)).astype(np.int64)
    q, scale = quantize_int8_per_row(bucket_rows[bucket_ids])
    onnx_path = repo / "models" / lang / "fasttext.{lang}.buckets.onnx".replace("{lang}", lang)
    model = make_buckets_model(q, scale, bucket_ids, header.bucket,
                               header.args["minn"], header.args["maxn"])
    onnx.save(model, str(onnx_path))
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    # Parity: the artifact's own tensors dequantize to the selected rows.
    saved_q = constant_array(onnx.load(str(onnx_path)), "q_embeddings").astype(np.int8)
    saved_scale = constant_array(onnx.load(str(onnx_path)), "row_scale").astype(np.float32)
    dequant = saved_q.astype(np.float32) * saved_scale[:, None]
    reference_rows = bucket_rows[bucket_ids]
    parity = float(np.abs(dequant - reference_rows).max())
    if parity > 0.05:  # the tiers' own QUANT_MAX_ABS_TOL
        print(f"FATAL: quantization parity {parity} exceeds 0.05", file=sys.stderr)
        return 2

    from hashlib import sha256 as _sha256
    entry = {
        "dims": dim,
        "vocab_size": int(len(bucket_ids)),  # kept rows (rows of the artifact)
        "quantization": "int8-per-row",
        "bytes": onnx_path.stat().st_size,
        "sha256": _sha256(onnx_path.read_bytes()).hexdigest(),
        "vocab_sha256": None,
        "vocab_bytes": None,
        "eval_ref": f"eval/reports/{lang}.buckets.json",
        "source_bin_sha256": audit["source"]["bin_sha256"],
        "bucket_count": header.bucket,
        "buckets": int(len(bucket_ids)),
        "usage_top_k": int(chosen),
        "demand_rows": int(len(demand_rows)),
        "minn": header.args["minn"],
        "maxn": header.args["maxn"],
        "parity_max_abs": parity,
    }
    tiers_path = repo / "models" / lang / "tiers.json"
    data = json.loads(tiers_path.read_text(encoding="utf-8")) if tiers_path.exists() else {}
    data["language"] = lang
    data.setdefault("tiers", {})["buckets"] = entry
    data["generated_at"] = iso_now()
    tiers_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    audit["chosen_k_usage"] = int(chosen)
    audit["artifact"] = {
        "path": str(onnx_path.relative_to(repo)),
        "bytes": entry["bytes"],
        "sha256": entry["sha256"],
        "rows": int(len(bucket_ids)),
        "parity_max_abs": parity,
    }
    report_path = repo / "eval" / "reports" / f"{lang}.buckets.json"
    report_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    print(f"[{lang}] wrote {onnx_path} ({entry['bytes']:,} bytes, "
          f"rows={len(bucket_ids):,} = usage top {chosen:,} + demand "
          f"{len(demand_rows):,})")
    print(f"[{lang}] updated {tiers_path} and {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
