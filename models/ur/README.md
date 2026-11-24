# models/ur — POS pipeline slot (SAMPLE-SCALE placeholder)

The interscript CLE17T015 POS order is in flight; this slot proves the
**train → gate → ONNX** pipeline end-to-end against the held 1,955-token
human-verified sample (pk-cle-urdu delivery 2026-11-20: 6 genre files,
64 sentences, 31 tags).

| artifact | purpose |
|---|---|
| `fasttext.ur.pos.npz` | trainer weights (ridge over one-hot prev/cur/next word features) |
| `fasttext.ur.pos.vocab.json` | word index + tag labels + provenance header |
| `fasttext.ur.pos.gate.json` | mechanical held-out check — **PIPELINE VALIDATION ONLY**, not an accuracy gate |
| `fasttext.ur.pos.onnx` | Gemm-graph export (onnx checker passed) |
| `fasttext.ur.pos.onnx.meta.json` | input contract + labels |

## Pipeline

```
python3 scripts/train_ur_pos.py \
  --sample ~/src/interscript/pk-cle-urdu/pos-sample/extracted --out models/ur
python3 scripts/export_ur_pos_onnx.py --model models/ur
```

## What this slot is NOT

The model inside is deliberately **placeholder-class** (linear over
one-hot context windows, trained in seconds, deterministic). No accuracy
claim is made; `fasttext.ur.pos.gate.json` exists so the pipeline has a
mechanical check to run, not as a quality bar. **The accuracy gate stays
deferred until the 100K human-reviewed gold arrives**, at which point
the production fasttext trainer swaps in under this same artifact
contract and the registry entry (`kotoshu://models/ur/pos`) is generated
from the real model.

## Provenance

Sample: CLE POS Sample Data (checksummed at
`~/src/interscript/pk-cle-urdu/pos-sample/20261120-Sample-Data.rar.sha256`),
CLE-Urdu-POS-Tagset per the L14-1250 paper; tag→reduced mapping lives in
the kotoshu gem (`Grammar::UrPosTagger::REDUCED`).
