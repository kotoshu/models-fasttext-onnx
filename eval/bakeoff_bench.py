#!/usr/bin/env python3
"""Plan-111 embedding bakeoff benchmark (report-only, no gates).

Scores successor candidates against the fastText tiers on the SAME
real-typo corpus protocol as eval/corpus_bench.py: pool construction,
ranking rule, tie handling and hit metrics are identical (pool builder
mirrored from corpus_bench/corpus loader; corpus_bench.py itself is
untouched). Per language (en de ru es by default):

- baseline A: the committed eval/reports/corpus.{lang}.json numbers for
  full/fluency/mini, imported verbatim, never recomputed here;
- candidate B: fastText full retrieval (top-100 cosine slate around the
  typo) + ms-marco-MiniLM-L-6-v2 int8 cross-encoder rerank over
  (context, candidate) pairs, where the context is the corpus source
  sentence containing the typo (extracted from the vendored raw corpus,
  first occurrence in file order, deterministic). A no-context ablation
  (query = the typo word alone) separates context value from rerank
  value. Hit metric: top-1/5/20 of the human correction within the
  slate after rerank; the correction outside the slate is a miss for
  every k (the product can never surface it), slate recall reported
  separately;
- candidate C: purpose-trained char-BiGRU typo bi-encoder
  (scripts/train_typo_biencoder.py). Ranks the correction among the
  whole full fastText vocabulary by cosine — the exact corpus_bench
  rule — but only on pool pairs with NO training-repo occurrence under
  any language label (strict repo-level split; exclusions counted);
- candidate D: ModernBERT-base int8 as a zero-shot bi-encoder (word ->
  mean-pooled L2-normalized embedding), same whole-vocab ranking rule.
  ModernBERT-small was never published (recorded in the manifest);
  base is 149.9 MB int8 — over the 30 MB budget, measured anyway to
  price the class.

Latency: batch-1 per-suggest on the bench hardware (median/p90 over
pool queries), retrieval through the ranked list, amortized per-language
vocabulary embeddings excluded (their build cost is printed).

Output: eval/reports/bakeoff.{lang}.json + eval/reports/bakeoff.summary.json
with a pre-declared decision rule and ship/reject verdicts (verdict prose
is finalized in eval/reports/bakeoff-v1.md from these numbers).

Usage:
  python3 eval/bakeoff_bench.py --repo-root . [--lang en de ru es]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

SEED = 42
MAX_PAIRS_DEFAULT = 2000
HIT_KS = (1, 5, 20)
LANGS_DEFAULT = ("en", "de", "ru", "es")
RERANK_SLATE = 100  # fastText retrieval slate size for candidate B
CE_MAX_LENGTH = 96  # cross-encoder total sequence budget (context + candidate)
LATENCY_QUERIES = 200

SIZE_CAP_MB = 30.0      # added int8 bytes budget (plan: 15-30 MB class)
LATENCY_CAP_MS = 100.0  # per-suggest median on the bench hardware
WINS_NEEDED = 3         # languages (of 4) beating fluency top-5
LANGS_TOTAL = 4


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------- pool


def load_pool(repo: Path, run_eval, lang: str, max_pairs: int) -> dict:
    """Identical pool construction to corpus_bench.bench_language."""
    model_dir = repo / "models" / lang
    full = run_eval.load_full_model(
        model_dir / f"fasttext.{lang}.onnx", model_dir / f"fasttext.{lang}.vocab.json"
    )
    corpus = json.loads((repo / "eval" / "corpora" / f"{lang}.json").read_text(encoding="utf-8"))
    pool = [
        (t, c)
        for t, c, _count in corpus["pairs"]
        if t != c and t in full.word_to_idx and c in full.word_to_idx
    ]
    in_full = len(pool)
    if len(pool) > max_pairs:
        rng = np.random.default_rng([SEED, zlib.crc32(lang.encode("utf-8"))])
        pool = [pool[i] for i in rng.permutation(len(pool))[:max_pairs]]
    words = [None] * full.vocab_size
    for w, i in full.word_to_idx.items():
        words[i] = w
    full.words = words
    return {
        "model": full,
        "pool": pool,
        "corpus": dict(corpus["corpus"], extraction=corpus["extraction"]),
        "pairs": {
            "extracted_unique": corpus["n_pairs_unique"],
            "both_in_full_vocab": in_full,
            "sampled_for_eval": len(pool),
            "sampling": (
                f"first {max_pairs} of default_rng([{SEED}, crc32(language)]) permutation"
                if in_full > max_pairs
                else "all pairs (below cap)"
            ),
        },
    }


def load_baselines(repo: Path, lang: str) -> dict:
    report = json.loads((repo / "eval" / "reports" / f"corpus.{lang}.json").read_text(encoding="utf-8"))
    return {
        tier: {
            "vocab_size": t["vocab_size"],
            "pairs_evaluated": t["pairs_evaluated"],
            "top1": t["top1"],
            "top5": t["top5"],
            "top20": t["top20"],
        }
        for tier, t in report["tiers"].items()
    }


# ---------------------------------------------------------------- contexts (B)


def extract_contexts(repo: Path, lang: str, wanted: set) -> dict:
    """First (file order) source sentence per (typo, correction) from the raw corpus."""
    cache = repo / "eval" / "corpora" / f"contexts.{lang}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return {tuple(k.split("\x00", 1)): v for k, v in data.items()}
    sys.path.insert(0, str(repo / "scripts"))
    from fetch_corpus import LANG_MAP, _word_pair

    out: dict[tuple[str, str], str] = {}
    corpus_path = repo / "eval" / "corpora" / "github-typo-corpus.v1.0.0.jsonl.gz"
    with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
        for line in f:
            if len(out) == len(wanted):
                break
            obj = json.loads(line)
            for edit in obj.get("edits", []):
                src = edit.get("src") or {}
                if LANG_MAP.get(src.get("lang", "")) != lang:
                    continue
                tgt = edit.get("tgt") or {}
                src_toks = src.get("text", "").split()
                tgt_toks = tgt.get("text", "").split()
                if not src_toks or len(src_toks) != len(tgt_toks):
                    continue
                diffs = [(a, b) for a, b in zip(src_toks, tgt_toks) if a != b]
                if len(diffs) != 1:
                    continue
                pair = _word_pair(*diffs[0])
                if pair is not None and pair in wanted and pair not in out:
                    out[pair] = src.get("text", "")
    cache.write_text(
        json.dumps({"\x00".join(k): v for k, v in out.items()}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


# ---------------------------------------------------------------- encoders


class CharEncoder:
    """Candidate C int8 ONNX char bi-encoder."""

    MAX_WORD_LEN = 24

    def __init__(self, cand_dir: Path) -> None:
        self.char_to_idx = json.loads((cand_dir / "char_to_idx.json").read_text(encoding="utf-8"))
        self.unk = len(self.char_to_idx) + 1
        self.sess = ort.InferenceSession(str(cand_dir / "model.int8.onnx"), providers=["CPUExecutionProvider"])

    def ids_for(self, words: list[str]) -> np.ndarray:
        ids = np.zeros((len(words), self.MAX_WORD_LEN), dtype=np.int64)
        for i, w in enumerate(words):
            for j, ch in enumerate(w[: self.MAX_WORD_LEN]):
                ids[i, j] = self.char_to_idx.get(ch, self.unk)
        return ids

    def encode(self, words: list[str], batch: int = 4096) -> np.ndarray:
        chunks = []
        for lo in range(0, len(words), batch):
            chunks.append(self.sess.run(["embedding"], {"char_ids": self.ids_for(words[lo : lo + batch])})[0])
        return np.vstack(chunks).astype(np.float32)


class TransformerEncoder:
    """Candidate D int8 ONNX bi-encoder (ModernBERT)."""

    def __init__(self, cand_dir: Path, max_length: int = 16, batch: int = 256) -> None:
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(str(cand_dir))
        self.sess = ort.InferenceSession(str(cand_dir / "model.int8.onnx"), providers=["CPUExecutionProvider"])
        self.max_length = max_length
        self.batch = batch
        self.feed_names = [i.name for i in self.sess.get_inputs()]

    def encode(self, words: list[str]) -> np.ndarray:
        chunks = []
        for lo in range(0, len(words), self.batch):
            chunk = words[lo : lo + self.batch]
            enc = self.tok(
                chunk, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="np"
            )
            feeds = {n: enc[n] for n in self.feed_names}
            chunks.append(self.sess.run(["embedding"], feeds)[0])
        return np.vstack(chunks).astype(np.float32)

    def unk_fraction(self, words: list[str]) -> float:
        """Fraction of words with NO known subword (fully [UNK] embeddings)."""
        vocab = self.tok.get_vocab()
        unk = self.tok.unk_token_id
        n_unk = 0
        for w in words:
            toks = self.tok.tokenize(w)
            if not toks or all(vocab.get(t) == unk for t in toks):
                n_unk += 1
        return n_unk / max(1, len(words))


def _normalized(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


# ---------------------------------------------------------------- scorers


def biencoder_bench(normalized: np.ndarray, word_to_idx: dict, pool: list) -> dict:
    """corpus_bench ranking rule over a candidate embedding matrix."""
    hits = {k: 0 for k in HIT_KS}
    ranks: list[int] = []
    typo_oov = 0
    corr_oov = 0
    for typo, correction in pool:
        t = word_to_idx.get(typo)
        c = word_to_idx.get(correction)
        if t is None:
            typo_oov += 1
            continue
        if c is None:
            corr_oov += 1
            continue
        sims = normalized @ normalized[t]
        sims[t] = -np.inf
        s = sims[c]
        rank0 = int(np.count_nonzero(sims > s))
        ranks.append(rank0 + 1)
        for k in HIT_KS:
            if rank0 < k:
                hits[k] += 1
    n = len(ranks)
    return {
        "pairs_evaluated": n,
        "pairs_typo_oov": typo_oov,
        "pairs_correction_oov": corr_oov,
        "top1": hits[1] / n if n else None,
        "top5": hits[5] / n if n else None,
        "top20": hits[20] / n if n else None,
        "mean_rank": float(np.mean(ranks)) if n else None,
        "median_rank": float(np.median(ranks)) if n else None,
        "ranking": (
            "rank of the correction among the full fastText vocabulary by cosine to "
            "the typo under the candidate encoder, excluding the typo itself (ties "
            "counted optimistically) — identical rule to corpus_bench"
        ),
    }


def rerank_bench(full, pool, contexts, ce_sess, ce_tok, use_context: bool, slate: int, block: int = 16) -> dict:
    """Candidate B: fastText top-slate retrieval + cross-encoder rerank.

    The correction outside the retrieval slate is a miss for every k (the
    product pipeline can never surface it); slate recall is reported
    separately as the retriever ceiling.
    """
    feed_names = [i.name for i in ce_sess.get_inputs()]
    hits = {k: 0 for k in HIT_KS}
    ranks: list[int] = []
    slate_hits = 0
    no_context = 0
    scored = 0

    for lo in range(0, len(pool), block):
        chunk = pool[lo : lo + block]
        queries: list[str] = []
        cands_per_q: list[list[str]] = []
        for typo, correction in chunk:
            if use_context and (typo, correction) not in contexts:
                no_context += 1
                continue
            t = full.word_to_idx[typo]
            sims = full.normalized @ full.normalized[t]
            sims[t] = -np.inf
            top = np.argpartition(-sims, slate - 1)[:slate]
            cands = [full.words[int(j)] for j in top]
            if correction in cands:
                slate_hits += 1
                cands_per_q.append(cands)
            else:
                cands_per_q.append(None)  # miss for every k; nothing to score
            queries.append(contexts[(typo, correction)] if use_context else typo)

        # flatten (only slates that contain their correction get scored)
        flat_q: list[str] = []
        flat_c: list[str] = []
        spans: list[tuple[int, int] | None] = []
        for q, cands in zip(queries, cands_per_q):
            if cands is None:
                spans.append(None)
                continue
            start = len(flat_q)
            flat_q.extend([q] * len(cands))
            flat_c.extend(cands)
            spans.append((start, len(flat_q)))

        scores = np.zeros(len(flat_q), dtype=np.float32)
        bs = 128
        for b in range(0, len(flat_q), bs):
            enc = ce_tok(
                flat_q[b : b + bs],
                flat_c[b : b + bs],
                padding="max_length",
                max_length=CE_MAX_LENGTH,
                truncation="only_first",
                return_tensors="np",
            )
            feeds = {n: enc[n] for n in feed_names}
            scores[b : b + bs] = ce_sess.run(["logits"], feeds)[0].astype(np.float32).ravel()

        # score each kept pair against its slate scores (kept_pairs aligns with
        # cands_per_q/spans: pairs skipped for missing context were never enqueued)
        kept_pairs = [p for p in chunk if (not use_context) or (p in contexts)]
        for (typo, correction), cands, span in zip(kept_pairs, cands_per_q, spans):
            scored += 1
            if span is None:
                ranks.append(slate + 1)
                continue
            sc = scores[span[0] : span[1]]
            corr_idx = cands.index(correction)
            rank0 = int(np.count_nonzero(sc > sc[corr_idx]))
            ranks.append(rank0 + 1)
            for k in HIT_KS:
                if rank0 < k:
                    hits[k] += 1

    n = scored
    assert n == len(ranks), f"{n} != {len(ranks)}"
    return {
        "pairs_evaluated": n,
        "slate_size": slate,
        "slate_recall": slate_hits / n if n else None,
        "no_context_pairs_skipped": no_context,
        "top1": hits[1] / n if n else None,
        "top5": hits[5] / n if n else None,
        "top20": hits[20] / n if n else None,
        "mean_rank": float(np.mean(ranks)) if n else None,
        "median_rank": float(np.median(ranks)) if n else None,
        "ranking": (
            f"rank of the correction within the {slate}-word fastText-cosine slate "
            "after cross-encoder rerank (ties counted optimistically); a correction "
            f"outside the slate is a miss for every k (rank {slate + 1})"
        ),
    }


def latency_bench(full, ce_sess, ce_tok, char_enc, mb_enc, c_nrm, d_nrm, pairs, contexts) -> dict:
    """Batch-1 per-suggest latency: from typo string to a ranked list.

    Amortized per-language vocabulary embeddings (C/D) are precomputed and
    passed in; their build cost is printed by the caller, not counted here.
    """
    res: dict[str, dict] = {}

    def stats(samples: list[float]) -> dict:
        arr = np.asarray(samples)
        return {"n": len(arr), "median_ms": float(np.median(arr)), "p90_ms": float(np.percentile(arr, 90))}

    probe = pairs[:LATENCY_QUERIES]
    ce_feeds = [i.name for i in ce_sess.get_inputs()]
    d_feeds = mb_enc.feed_names

    # A: fastText full-tier retrieval (cosine over the whole vocab + top-20)
    samples = []
    for typo, _c in probe:
        t = full.word_to_idx.get(typo)
        if t is None:
            continue
        t0 = time.perf_counter()
        sims = full.normalized @ full.normalized[t]
        np.argpartition(-sims, 20)[:20]
        samples.append((time.perf_counter() - t0) * 1e3)
    res["a_full_fasttext"] = stats(samples)

    # B: retrieval + cross-encoder rerank of the slate (one batched CE call)
    samples = []
    for typo, correction in probe:
        t = full.word_to_idx.get(typo)
        if t is None:
            continue
        ctx = contexts.get((typo, correction), typo)
        t0 = time.perf_counter()
        sims = full.normalized @ full.normalized[t]
        sims[t] = -np.inf
        top = np.argpartition(-sims, RERANK_SLATE - 1)[:RERANK_SLATE]
        cands = [full.words[int(j)] for j in top]
        enc = ce_tok(
            [ctx] * len(cands),
            cands,
            padding="max_length",
            max_length=CE_MAX_LENGTH,
            truncation="only_first",
            return_tensors="np",
        )
        feeds = {n: enc[n] for n in ce_feeds}
        ce_sess.run(["logits"], feeds)
        samples.append((time.perf_counter() - t0) * 1e3)
    res["b_rerank_slate100"] = stats(samples)

    # C: char-encode the typo (batch 1) + cosine over the precomputed vocab matrix
    samples = []
    for typo, _c in probe:
        t0 = time.perf_counter()
        q = char_enc.sess.run(["embedding"], {"char_ids": char_enc.ids_for([typo])})[0]
        sims = c_nrm @ (q[0] / max(float(np.linalg.norm(q[0])), 1e-9))
        np.argpartition(-sims, 20)[:20]
        samples.append((time.perf_counter() - t0) * 1e3)
    res["c_biencoder_query_plus_matvec"] = stats(samples)

    # D: transformer-encode the typo (batch 1) + cosine over the precomputed matrix
    samples = []
    for typo, _c in probe:
        enc = mb_enc.tok([typo], padding="max_length", max_length=mb_enc.max_length, truncation=True, return_tensors="np")
        feeds = {n: enc[n] for n in d_feeds}
        t0 = time.perf_counter()
        q = mb_enc.sess.run(["embedding"], feeds)[0]
        sims = d_nrm @ (q[0] / max(float(np.linalg.norm(q[0])), 1e-9))
        np.argpartition(-sims, 20)[:20]
        samples.append((time.perf_counter() - t0) * 1e3)
    res["d_modernbert_query_plus_matvec"] = stats(samples)

    res["note"] = (
        "per-language vocabulary embedding matrices are amortized (built once per "
        "language; build cost printed in the bench log), so C/D latency is the "
        "incremental per-suggest cost"
    )
    return res


# ---------------------------------------------------------------- main


def cached_vocab_matrix(cache_path: Path, encode_fn) -> np.ndarray:
    """Cache the per-language vocabulary embedding matrix (gitignored dir).

    The ModernBERT pass over a 100k-word vocabulary takes ~20 minutes per
    language on the bench hardware; the cache keeps reruns and partial
    failures cheap. Cache key is the artifact directory name + language;
    stale entries are impossible because artifacts are rebuilt by
    scripts/prepare_bakeoff_candidates.py / train_typo_biencoder.py, which
    recreate the whole candidate directory (new sizes/sha256 land in the
    receipt and this cache path moves with it).
    """
    if cache_path.exists():
        return np.load(cache_path)
    mat = encode_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, mat)
    return mat


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan-111 bakeoff benchmark, report-only")
    parser.add_argument("--lang", nargs="+", default=list(LANGS_DEFAULT))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS_DEFAULT)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo / "eval"))
    import run_eval  # noqa: E402
    from transformers import AutoTokenizer

    manifest = json.loads((repo / "eval" / "candidates" / "manifest.json").read_text(encoding="utf-8"))

    ce_sess = ort.InferenceSession(
        str(repo / "eval" / "candidates" / "b_rerank" / "model.int8.onnx"), providers=["CPUExecutionProvider"]
    )
    ce_tok = AutoTokenizer.from_pretrained(str(repo / "eval" / "candidates" / "b_rerank"))
    char_enc = CharEncoder(repo / "eval" / "candidates" / "c_typo")
    eval_ok_pairs = {
        tuple(p)
        for p in json.loads(
            (repo / "eval" / "candidates" / "c_typo" / "eval_ok_pairs.json").read_text(encoding="utf-8")
        )
    }
    mb_enc = TransformerEncoder(repo / "eval" / "candidates" / "d_modernbert_base")

    reports_dir = repo / "eval" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_langs: dict[str, dict] = {}

    for lang in args.lang:
        print(f"== {lang}")
        pool_data = load_pool(repo, run_eval, lang, args.max_pairs)
        full = pool_data["model"]
        pool = pool_data["pool"]
        baselines = load_baselines(repo, lang)
        words = full.words
        contexts = extract_contexts(repo, lang, set(pool))

        # candidate B (with-context + no-context ablation)
        print(f"  B rerank (ctx) on {len(pool)} pairs, slate {RERANK_SLATE}")
        b_ctx = rerank_bench(full, pool, contexts, ce_sess, ce_tok, use_context=True, slate=RERANK_SLATE)
        print(f"  B rerank (no ctx)")
        b_noctx = rerank_bench(full, pool, contexts, ce_sess, ce_tok, use_context=False, slate=RERANK_SLATE)

        # candidate C (embed vocab once; rank on the repo-clean subset)
        print(f"  C embedding {full.vocab_size} vocab words")
        t0 = time.perf_counter()
        c_mat = cached_vocab_matrix(
            repo / "eval" / "candidates" / "cache" / f"c_typo.vocab.{lang}.npy",
            lambda: char_enc.encode(words),
        )
        c_nrm = _normalized(c_mat)
        print(f"    vocab embedding {time.perf_counter() - t0:.1f}s")
        pool_clean = [p for p in pool if p in eval_ok_pairs]
        c_res = biencoder_bench(c_nrm, full.word_to_idx, pool_clean)
        c_res["pairs_excluded_train_tainted"] = len(pool) - len(pool_clean)
        # apples-to-apples comparator: the fastText FULL tier (same 100k ranking
        # universe C uses) restricted to the identical repo-clean subset — the
        # clean subset is distributionally skewed toward rare single-repo pairs,
        # so the unrestricted corpus baseline is NOT a fair comparator for C
        c_res["baseline_full_on_same_subset"] = biencoder_bench(full.normalized, full.word_to_idx, pool_clean)

        # candidate D (embed vocab once; rank on the whole pool)
        print(f"  D embedding {full.vocab_size} vocab words (ModernBERT-base int8)")
        t0 = time.perf_counter()
        d_mat = cached_vocab_matrix(
            repo / "eval" / "candidates" / "cache" / f"d_modernbert_base.vocab.{lang}.npy",
            lambda: mb_enc.encode(words),
        )
        d_nrm = _normalized(d_mat)
        print(f"    vocab embedding {time.perf_counter() - t0:.1f}s")
        d_res = biencoder_bench(d_nrm, full.word_to_idx, pool)
        d_res["vocab_unk_fraction_first20k"] = mb_enc.unk_fraction(words[:20000])

        lat = latency_bench(full, ce_sess, ce_tok, char_enc, mb_enc, c_nrm, d_nrm, pool, contexts)

        report = {
            "language": lang,
            "protocol": (
                "pool construction and ranking rule identical to eval/corpus_bench.py "
                "(see eval/reports/corpus.{lang}.json); candidates additionally use the "
                "raw corpus for contexts (B) and the repo-level train/eval split (C)"
            ),
            "corpus": pool_data["corpus"],
            "pairs": pool_data["pairs"],
            "baselines_fasttext": baselines,
            "baseline_note": "full/fluency/mini numbers imported verbatim from the committed corpus report",
            "candidate_b_rerank": {
                "with_context": b_ctx,
                "no_context_ablation": b_noctx,
                "model": manifest["b_rerank"],
            },
            "candidate_c_biencoder": {
                **c_res,
                "model": manifest["c_typo"],
                "protocol_note": (
                    "evaluated only on pool pairs with no occurrence in any training "
                    "repo under any language label (strict repo-level split)"
                ),
            },
            "candidate_d_modernbert": {
                **d_res,
                "model": manifest["d_modernbert_base"],
                "modernbert_small": manifest.get("d_modernbert_small"),
            },
            "latency_ms_per_suggest": lat,
            "determinism": {"seed": SEED, "rng": "np.random.default_rng([seed, crc32(language)])"},
            "report_only": True,
            "generated_at": iso_now(),
        }
        out = reports_dir / f"bakeoff.{lang}.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        flu = baselines["fluency"]
        c_top5 = c_res["top5"]
        c_base_top5 = c_res["baseline_full_on_same_subset"]["top5"]
        line = (
            f"{lang}: pool={len(pool)} | A fluency top5={flu['top5']:.3f}"
            f" | B ctx top5={b_ctx['top5']:.3f} (noctx {b_noctx['top5']:.3f}, slate_rec {b_ctx['slate_recall']:.3f})"
            f" | C top5={c_top5:.3f} vs full-on-clean {c_base_top5:.3f} (n={c_res['pairs_evaluated']}, excluded {c_res['pairs_excluded_train_tainted']})"
            f" | D top5={d_res['top5']:.3f} (unk {d_res['vocab_unk_fraction_first20k']:.2f})"
            f" | lat ms: A {lat['a_full_fasttext']['median_ms']:.1f} B {lat['b_rerank_slate100']['median_ms']:.0f}"
            f" C {lat['c_biencoder_query_plus_matvec']['median_ms']:.2f} D {lat['d_modernbert_query_plus_matvec']['median_ms']:.1f}"
        )
        print(line + f" -> {out.relative_to(repo)}")

        summary_langs[lang] = {
            "pool": len(pool),
            "fluency_top5": flu["top5"],
            "b_top5_ctx": b_ctx["top5"],
            "b_top5_noctx": b_noctx["top5"],
            "b_slate_recall": b_ctx["slate_recall"],
            "c_top5": c_top5,
            "c_full_baseline_on_same_subset_top5": c_base_top5,
            "c_n": c_res["pairs_evaluated"],
            "c_excluded": c_res["pairs_excluded_train_tainted"],
            "d_top5": d_res["top5"],
            "latency_median_ms": {
                "a_full": lat["a_full_fasttext"]["median_ms"],
                "b_rerank": lat["b_rerank_slate100"]["median_ms"],
                "c_biencoder": lat["c_biencoder_query_plus_matvec"]["median_ms"],
                "d_modernbert": lat["d_modernbert_query_plus_matvec"]["median_ms"],
            },
        }

    summary = {
        "plan": "111 embedding bakeoff",
        "decision_rule": {
            "rule": (
                f"a candidate SHIPs only if it beats the fluency tier corpus top-5 on "
                f">= {WINS_NEEDED} of {LANGS_TOTAL} languages (en de ru es) AND adds "
                f"<= {SIZE_CAP_MB:g} MB int8 AND keeps median per-suggest latency "
                f"<= {LATENCY_CAP_MS:g} ms on the bench hardware"
            ),
            "reject_on_data_is_complete": "a data-backed rejection closes the candidate, like the int4 ladder",
        },
        "candidates": {
            "b_rerank": {
                "int8_mb": manifest["b_rerank"]["int8_mb"],
                "within_size_cap": manifest["b_rerank"]["int8_bytes"] <= SIZE_CAP_MB * 1e6,
            },
            "c_typo_biencoder": {
                "int8_mb": manifest["c_typo"]["int8_mb"],
                "within_size_cap": manifest["c_typo"]["int8_bytes"] <= SIZE_CAP_MB * 1e6,
            },
            "d_modernbert_base": {
                "int8_mb": manifest["d_modernbert_base"]["int8_mb"],
                "within_size_cap": manifest["d_modernbert_base"]["int8_bytes"] <= SIZE_CAP_MB * 1e6,
            },
            "d_modernbert_small": {"status": manifest["d_modernbert_small"]["status"]},
        },
        "languages": summary_langs,
        "hardware": "Apple M1 Max (arm64), CPUExecutionProvider, onnxruntime 1.23.2",
        "generated_at": iso_now(),
    }
    out = reports_dir / "bakeoff.summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
