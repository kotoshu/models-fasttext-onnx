# Plan 24: the Urdu POS tagger — CLE tagset adoption, gated ONNX artifact

## Status: proposed — blocked on owner decisions (CLE order + supersession answer); nothing acquired, trained, or shipped

Three owner gates stand before execution: (1) CLE's answer on whether
CLE17T015 (IOB, POS + chunk) supersedes CLE12T006's POS layer, and the
USD 250-store-vs-500-quoted fee delta; (2) the store order itself —
commercial licensing rides the executed CLE-Ribose agreement
(2026-10-29), so the order is a Schedule update the owner signs; (3)
confirmation that training runs in the private compliance environment
(the pk-cle-urdu pipeline guard: no public repo, workflow, or AI-
assisted session may ingest CLE text). The tagset SCHEMA itself is a
published LREC 2014 paper, free to adopt with citation — that half is
unblocked and is the first execution step. The consumer-side work is a
separate plan in the gem's own TODO.impl counter (see Consumers).

## Problem

Urdu has zero kotoshu artifacts: the registry (v1.8.0) carries no `ur`
resource of any kind (fastText publishes cc.ur vectors —
`scripts/test_ur.sh` exists — but no ur tier ever shipped), and the
grammar pipeline cannot serve the language: `Grammar::Checker` and
`PosSequenceMatcher#tag_tokens` hardcode the rule-based English
`PosTagger`, whose word lists and suffix tables are English-only. Free
tagged Urdu effectively does not exist (EMILLE/UXIL carry older,
different tagsets), and CLE is the only source of a current,
linguist-verified tagset plus gold: the LREC 2014 tagset (Ahm et al.,
12 categories, ~35 tags — public paper, adoptable with citation) and
the 100K-word human-reviewed gold (CLE12T006; CLE17T015 adds an IOB
chunk layer). CLE's own 2014 TreeTagger reached 96.8% on this corpus,
so a 100K-gold linear tagger has a measured bar to clear — and the
grammar rules the ecosystem needs for Urdu want exactly that class
tagger, not a per-language re-derivation of the English suffix hack.

## What

- **Tagset adoption + citation (unblocked, do first)**:
  `docs/urdu-pos-tagset.md` records the 12-category / ~35-tag schema
  transcribed from the paper (Ahm et al., LREC 2014,
  https://aclanthology.org/L14-1250 — held at
  pk-cle-urdu/pos-sample/CLE-Urdu-POS-Tagset-L14-1250.pdf) with the
  citation, plus the CLE-tag-to-Kotoshu mapping table grounded on the
  31 tags observed in the verified sample. Schema and mapping are
  public (published tagset); README License/Citation sections gain the
  L14-1250 bibtex beside the fastText one. The tagged DATA never
  becomes public — see the compliance bullet.
- **Data acquisition (owner-gated)**: first choice is CLE17T015 (2017,
  same 348-file/100K structure, every word POS + IOB chunk
  NP/VP/PP/PSP, USD 250 store fee) — one order covers POS and chunks
  and supersedes the 2012 POS-only corpus IF CLE confirms the layer is
  current; fall back to CLE12T006 (store fee USD 250 published vs USD
  500 quoted — anchor the reply to the published fee or ask for the
  delta rationale). The order is a Schedule update under the executed
  agreement. On arrival, sentence-level validation against the held
  verified sample (pk-cle-urdu/pos-sample: 1,955 tokens, 31 tags, all
  valid, traced into the licensed CLE12T003 text) is the acceptance
  check that the ordered POS layer matches what CLE certified.
- **Training slot**: `scripts/build_pos_tagger.py` follows the
  `build_lid.py` precedent — a supervised fastText classifier (labels
  = tags, not languages) serialized as an int8-per-row ONNX container
  plus sidecar JSON (word dictionary, pruned n-gram map, label list,
  model args). Self-contained by design: it carries its own
  char-ngram embedding space, so the missing `ur` embedding tier is
  NOT a dependency. Training runs in the private environment over the
  CLE gold only; this public repo receives the gated artifact, the
  eval report (aggregate numbers only), and the citation.
- **Gates** (all pass before registry promotion):
  - token accuracy >= 96.8% on a held-out gold split — the CLE 2014
    TreeTagger baseline on the same corpus; below it the linear rung
    escalates per the house ladder instead of shipping a weaker
    artifact;
  - int8-vs-fp32 label agreement = 1.0 on the verified sample (a POS
    flip is a wrong answer, not ranking drift — the lid gate,
    stricter than the embedding tiers' 0.95/0.85), with the
    recorded else-float32 fallback if the quantized artifact fails;
  - clean-text budget: zero grammar-rule firings caused solely by a
    POS mis-tag on the verified-clean sample sentences;
  - artifact floor: ONNX ir_version 10 per the 2026-09-19 law
    (first bytes `08 0a`), <= 20 MB mirror budget (expected
    sub-2 MB), shipped as a release asset + registry entry, zero
    Git LFS;
  - compliance gate, before ANY release: Clause 16(e) extraction and
    membership-inference testing passed with records retained (2
    years), Schedule D attribution in the model card and NOTICE
    (NOTICE-TEMPLATE.md wording + the POS corpus catalog number),
    compliance certificate covering the release.
- **Compliance envelope** (binding, from pk-cle-urdu/COMPLIANCE.md):
  CLE text lives only in the private access-controlled repo; the
  pipeline guard forbids ingestion by any public repo, workflow, or
  release channel, and forbids submitting corpus text to third-party
  AI services (including AI coding assistants) — sample and gold
  processing is local scripts only. Permitted derivative: trained
  weights that pass 16(e). Prohibited: the annotated gold itself and
  derived datasets including train/dev/test splits — the split stays
  in the private repo; only aggregate accuracy numbers flow here.
  Anything not on Schedule C's permitted list (e.g. a chunk head
  beyond the trained tagger) needs written CLE clearance under
  Clause 16(d) before it ships.
- **Not in scope**: the 1M-word human re-review (USD 5 / 1,000 words
  single-annotator quote; the IAA full-dataset-vs-sample-percentage
  question) — this plan ships exactly the ordered 100K gold, nothing
  more; no corpus re-annotation or coverage expansion; no chunk-parser
  training beyond consuming the IOB layer as data; no commercial
  redistribution of CLE data; and no `ur` embedding tier (a separate
  owner decision under the plans 06/12-13 machinery).

## Consumers

The rule engine's expectation (verified in the gem): rules are YAML
`pattern:` slots (`pos:` tags, `word:`, `regex:`, `optional:`,
`skip:`, `negated:`, `exception:`) matched by
`PosSequenceMatcher#pos_match?` over the reduced Kotoshu tagset with
hierarchical groups (:VERB, :NOUN, :PRON cover their subtypes); POS
reaches the engine exclusively through a tagger call, today hardcoded
`PosTagger.new` in two places (`Grammar::Checker`,
`PosSequenceMatcher#tag_tokens`). A gem TODO.impl plan (its own
counter) lands alongside: pluggable per-language tagger selection
(rule-based en stays the default; the ONNX ur tagger attaches through
`Grammar::Checker`'s existing `language:` seam), a CLE mapper in the
`Loaders::PosMapper` family (one mapper per tagset family — the
existing mapper stays LanguageTool/Penn-only), and
`grammar/rules/ur/*.yml` for the Urdu agreement and postposition
classes. The IOB chunk layer has NO consumer today — no chunk concept
exists in the pattern DSL; consuming it (phrase slots) is a follow-on
engine decision, not this plan.
