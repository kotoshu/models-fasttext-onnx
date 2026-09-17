#!/usr/bin/env python3
"""Modal.com integration - serverless fastText substrate training (plan 21).

The variant-model trainings (zh-Hans-CN / zh-Hant-TW / zh-Hant-HK / ja
and the fleet-audit retrains) run here instead of the laptop: fastText
is CPU-bound, and Modal gives it 16+ dedicated cores (owner directive
2026-09-17: "use modal to run not on local computer"). A 500 MB corpus
that trains in ~60 local minutes completes in ~15-20 minutes at ~$0.10.

Recipe per the house modal pattern (ml-models/src/gpu/modal_train.py):
debian_slim, corpus mounted at run time (image rebuilds only when the
deps change), results written back through a Modal Volume so a retry
resumes instead of restarting.

Usage (detached; poll with `modal app list` / the volume):
  modal run scripts/modal_train_fasttext.py \
      --corpus eval/corpus/zh-hans-quality.txt \
      --name zh-hans-cn-quality \
      --minn 1 --maxn 3 --bucket 2000000 --epoch 5

Then pull the artifacts:
  modal volume get kotoshu-models zh-hans-cn-quality/ .
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fasttext-wheel==0.9.2")
)

VOLUME = modal.Volume.from_name("kotoshu-models", create_if_missing=True)

stub = modal.App("kotoshu-fasttext-train", image=IMAGE)

CORPUS_MOUNT = "/root/corpus.txt"


@stub.function(
    cpu=16.0,
    memory=32768,
    timeout=6 * 60 * 60,
    volumes={"/root/artifacts": VOLUME},
)
def train(corpus_path: str, name: str, minn: int, maxn: int,
          bucket: int, epoch: int, dim: int = 300) -> dict:
    import time
    import fasttext

    out_dir = Path("/root/artifacts") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{name}.bin"
    vec_path = out_dir / f"{name}.vec"

    if bin_path.exists() and vec_path.exists():
        # resume: a retry after a network blip reuses finished work
        words_done = vec_path.read_text(encoding="utf-8").splitlines()
        return {"status": "resumed", "vocab": len(words_done) - 1,
                "bin": str(bin_path)}

    t0 = time.time()
    model = fasttext.train_unsupervised(
        corpus_path if corpus_path.startswith("/") else CORPUS_MOUNT,
        model="skipgram", dim=dim, minn=minn, maxn=maxn, bucket=bucket,
        minCount=5, epoch=epoch, thread=16, ws=5, lr=0.05,
    )
    train_min = (time.time() - t0) / 60
    model.save_model(str(bin_path))
    words = model.get_words()
    with vec_path.open("w", encoding="utf-8") as fh:
        fh.write(f"{len(words)} {dim}\n")
        for w in words:
            v = model.get_word_vector(w)
            fh.write(w + " " + " ".join(f"{x:.5f}" for x in v) + "\n")
    VOLUME.commit()

    return {
        "status": "trained",
        "minutes": round(train_min, 1),
        "vocab": len(words),
        "bin": str(bin_path),
        "vec": str(vec_path),
        "nn_sample": model.get_nearest_neighbors("\u6570", k=5),
    }


@stub.local_entrypoint()
def main(corpus: str, name: str, minn: int = 1, maxn: int = 3,
         bucket: int = 2_000_000, epoch: int = 5, dim: int = 300):
    corpus_path = Path(corpus)
    if not corpus_path.is_absolute():
        corpus_path = REPO_ROOT / corpus_path
    print(f"uploading {corpus_path.name} "
          f"({corpus_path.stat().st_size / 2**20:.0f} MB) to Modal...")
    import time
    remote_corpus = f"{name}/corpus-{int(corpus_path.stat().st_size)}.txt"
    with VOLUME.batch_upload() as batch:
        batch.put_file(corpus_path, remote_corpus)
    print(f"corpus uploaded to volume as {remote_corpus}")
    t0 = time.time()
    result = train.remote(f"/root/artifacts/{remote_corpus}", name,
                          minn, maxn, bucket, epoch, dim)
    print(result, f"(wall {(time.time() - t0) / 60:.1f} min incl. upload)")
