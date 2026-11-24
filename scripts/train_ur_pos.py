"""Train the ur POS pipeline slot at SAMPLE SCALE (pipeline validation only).

The interscript CLE17T015 order is in flight; the 1,955-token human-
verified sample exists only to prove the slot runs end-to-end:

  train (this script) -> gate (scripts/gate_ur_pos.py) -> ONNX export

The model inside is deliberately placeholder-class: a ridge-regularized
linear classifier over one-hot (prev, cur, next) word features — small
enough to train deterministically on the sample in seconds and to export
as a plain ONNX Gemm graph. NO accuracy claim is made or gated; the
accuracy gate stays deferred until the 100K gold arrives, at which point
this placeholder is swapped for the production fasttext trainer under
the same artifact contract (models/ur/fasttext.ur.pos.*).

Inputs:  the pk-cle-urdu sample (pos-sample/extracted/*.txt, word/TAG)
Outputs: models/ur/fasttext.ur.pos.npz      (weights)
         models/ur/fasttext.ur.pos.vocab.json
Usage:   python3 scripts/train_ur_pos.py \
           --sample ~/src/interscript/pk-cle-urdu/pos-sample/extracted \
           --out models/ur
"""
import argparse
import glob
import json
import os

import numpy as np

HOLDOUT_FRACTION = 0.15
RIDGE = 1.0
SEED = 20261124


def read_sentences(sample_dir):
    sentences = []
    for path in sorted(glob.glob(os.path.join(sample_dir, "*.txt"))):
        for line in open(path, encoding="utf-8-sig"):
            toks = []
            for tok in line.strip().split():
                if "/" not in tok:
                    continue
                word, tag = tok.rsplit("/", 1)
                if word:
                    toks.append((word, tag))
            if toks:
                sentences.append(toks)
    return sentences


def featurize(sentences, word_index):
    rows, labels = [], []
    for sent in sentences:
        words = [w for w, _ in sent]
        tags = [t for _, t in sent]
        for i, tag in enumerate(tags):
            cur = word_index.get(words[i], 0)
            prev = word_index.get(words[i - 1], 0) if i else 0
            nxt = word_index.get(words[i + 1], 0) if i + 1 < len(words) else 0
            rows.append((prev, cur, nxt))
            labels.append(tag)
    return rows, labels


def one_hot(rows, n_words):
    # three one-hot blocks (prev, cur, next)
    x = np.zeros((len(rows), 3 * n_words), dtype=np.float32)
    for r, (prev, cur, nxt) in enumerate(rows):
        x[r, prev] = 1.0
        x[r, n_words + cur] = 1.0
        x[r, 2 * n_words + nxt] = 1.0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sentences = read_sentences(a.sample)
    cut = int(len(sentences) * (1 - HOLDOUT_FRACTION))
    train_sents, gate_sents = sentences[:cut], sentences[cut:]

    words = sorted({w for s in sentences for w, _ in s})
    labels = sorted({t for s in sentences for _, t in s})
    word_index = {w: i + 1 for i, w in enumerate(words)}  # 0 = <unk>
    label_index = {t: i for i, t in enumerate(labels)}
    n_words = len(words) + 1

    train_rows, train_labels = featurize(train_sents, word_index)
    x = one_hot(train_rows, n_words)
    y = np.array([label_index[t] for t in train_labels])

    # ridge closed form: (X'X + λI)^-1 X'Y_onehot
    eye = np.eye(x.shape[1], dtype=np.float32) * RIDGE
    y_oh = np.eye(len(labels), dtype=np.float32)[y]
    w = np.linalg.solve(x.T @ x + eye, x.T @ y_oh).astype(np.float32)

    # mechanical gate: held-out token accuracy (PIPELINE check, not an
    # accuracy bar — sample scale proves the slot runs, nothing else)
    gate_rows, gate_labels = featurize(gate_sents, word_index)
    xg = one_hot(gate_rows, n_words)
    pred = xg @ w
    hit = sum(1 for r, t in enumerate(gate_labels)
              if labels[int(np.argmax(pred[r]))] == t)
    gate = {"heldout_tokens": len(gate_labels), "correct": hit,
            "accuracy": round(hit / max(len(gate_labels), 1), 4),
            "note": "PIPELINE VALIDATION ONLY — accuracy gate deferred "
                    "until the 100K gold arrives"}

    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "fasttext.ur.pos.npz"),
                        weights=w, n_words=np.int64(n_words))
    vocab = {"_provenance": "ur POS pipeline slot, SAMPLE-SCALE placeholder "
                            "(pk-cle-urdu 2026-11-20 sample); the 100K gold "
                            "replaces this model",
             "words": words, "unk_index": 0, "labels": labels}
    json.dump(vocab, open(os.path.join(a.out, "fasttext.ur.pos.vocab.json"), "w"),
              ensure_ascii=False, indent=1)
    json.dump(gate, open(os.path.join(a.out, "fasttext.ur.pos.gate.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"train sentences: {len(train_sents)} gate: {len(gate_sents)}")
    print(f"vocab: {len(words)} words, {len(labels)} tags")
    print(f"gate: {gate['correct']}/{gate['heldout_tokens']} "
          f"= {gate['accuracy']} (pipeline validation only)")


if __name__ == "__main__":
    main()
