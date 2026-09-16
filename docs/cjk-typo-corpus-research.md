# CJK typo-corpus research — sources, probe results, and the script-separation requirement

Status: research executed 2026-09-16 (plan 18 follow-up; owner chose
the same-space route). Probes ran on this machine; every number below
is reproducible from the downloaded samples (gitignored, local).

## Why

Three campaign needs starve on CJK typo data: (1) the bucket gates
(zh gated on 3 probes vs en's 400); (2) real-word detection eval
(plan 15: zh had 22 real-word pairs vs en's 12,648); (3) any future
KTM1 matrix for CJK. The owner directed the same-space route (one
segmented zh model serving tiers AND buckets) and — mid-research —
added a hard requirement: **separate Traditional (TW/HK) and
Simplified models.**

## The 2025/2026 research map (arXiv)

The comprehensive survey (arXiv 2502.11508, 2025-02) Table 2 is the
canonical dataset census:

| Dataset | Sentences | Real errors? | Source |
|---|---|---|---|
| SIGHAN13 (Wu 2013) | 1,700 | yes | Chinese beginners |
| SIGHAN14 (Yu 2014) | 4,499 | yes | Chinese beginners |
| SIGHAN15 (Tseng 2015) | 3,439 | yes | Chinese beginners |
| Wang271K (Wang 2018, D18-1273) | 271,329 | NO — synthetic | newspapers |
| MCSC (Jiang 2022) | 196,495 | yes | medical website |
| ECSpell (Lv 2023) | 8,188 | partial | exams/web/docs |
| LEMON (Wu 2023) | 22,252 | yes | daily writing |
| CSCD-NS (Hu 2024) | 40,000 | yes | Weibo |
| AlipaySEQ (Wang 2024) | 15,522 | yes | search queries |

2025/2026 SOTA papers: CEC-Zero (2512.23971 / 2505.09082) generates
training data with LLMs + self-rewards; RAIR (2504.18938) retrieval;
AxBERT (2503.02255) associative networks; multimodal CSC (2504.07661);
CL2GEC (2509.13672) literature-domain benchmark; a 2026 layered error
taxonomy (2609.02153). Trend: synthetic/LLM-generated training data
evaluated on the small REAL sets — consistent with our house rule
(synthetic allowed for training, REAL corpora for gates).

## Verified HuggingFace candidates

- `twnlp/csc_data` (MIT; aggregate of W271K 279,816 + Medical[MCSC]
  39,303 + Lemon 22,259 + ECSpell 6,688 + CSCD 35,001; train.txt +
  SIGHAN15 + ECSpell domain tests). Format: `wrong<TAB>right` full
  sentences WITH context.
- `twnlp/lang8_hsk` (MIT; 1,568,885 parallel learner sentences,
  Lang-8 + HSK). Real learner corrections, sentence pairs.
- `twnlp/cgc_data` (MIT; CGED + FCGEC + MuCGEC + NACGEC) —
  grammar-class (CGEC), not spelling-class.
- `Macropodus/csc_clean_wang271k` (cleaned eval; upstream
  shibing624/CSC; provenance chain to D18-1273).
- `huanranhu-ruc/MCSC` — 2026 multimodal model outputs, not a corpus.

## Probe results (our extractor, fetch_corpus._word_pair)

| Corpus | lines | pairs extracted | unique | real-word class (both in zh tier vocab) |
|---|---|---|---|---|
| GitHub typo corpus zh (current) | 339 edits | 22 | 22 | 22 |
| twnlp/csc_data train.txt | 381,900 | 19,486 | 19,213 | 751 unique / 1,005 instances |
| twnlp/lang8_hsk train.para | 1,568,885 | 12,737 | 10,774 | 163 unique / 338 instances |

Reading: csc_data solves the GATE starvation (19k unique pairs; the
top homophone classes are exactly the real-word detection targets:
在/再, 那/哪, 做/作, 的/地, 象/像, 收/受, 惟/唯). Lang-8 is
grammar-class data (word-choice corrections like 只是/只有,
预防/避免) — small typo-pair yield but genuinely real; useful later
for CGEC-class work and as real-word eval context. Wang271k's
synthetic share (~271k of 381k lines) must be kept OUT of gate
positions (house rule) but is fine for model training signal.

## Script separation (owner directive: TW/HK vs mainland models)

Measured script mix (fixed probe char-sets, first 200k lines):

| Source | Simplified-only | Traditional-only | mixed-script |
|---|---|---|---|
| wikimedia/wikipedia 20231101.zh shard 0 (segmented) | 27.8% | 22.2% | 7.5% |
| twnlp/csc_data train.txt | 85.0% | 0.0% | 0.0% |
| twnlp/lang8_hsk train.para | 77.6% | 0.1% | 6.4% |

Findings and design consequences:

1. **The currently-shipped zh tiers are script-mixed** — nearly 30%
   of the Wikipedia training source is Traditional. The directive
   retroactively exposes a live quality issue: the single mixed zh
   model serves neither variant optimally.
2. HF wikipedia has NO zh-Hant config — the Traditional half must be
   FILTERED from the mixed shard (real TW/HK-written text, ~22% of
   each shard) and/or derived via OpenCC (s2twp converts vocabulary
   too: 软件→軟體, 网络→網路).
3. zh-Hans pipeline: Wikipedia simp-only lines + csc_data (already
   85% simp, 0% trad) + lang8_hsk simp lines. Clean end to end.
4. zh-Hant pipeline: Wikipedia trad-only lines (real TW/HK text);
   typo data via (a) OpenCC s2twp conversion of csc_data (synthetic
   share stays synthetic), and/or (b) the ORIGINAL SIGHAN 2013/2014
   releases (Traditional/Big5 by origin — the twnlp copies are
   simplified-converted). Lang-8 trad-only lines are too few (0.1%).
5. **Registry code decision is the owner's**: the schema accepts
   ^[a-z]{2,3}$ and no two-letter code distinguishes the variants.
   Options: (i) widen the schema to BCP-47 subtags (zh-Hans,
   zh-Hant — cleanest, matches the owner's phrasing); (ii) zh stays
   Simplified + a second unofficial 2-letter code. Option (i)
   touches the validator, the gem's resolver, and pack keys — a
   small but real cross-repo arc.

## Three variants, not two scripts (owner: "HK and TW and CN use different concepts/vocabs!")

The separation requirement is REGIONAL, not merely graphical: HK writes
Traditional but shares MAINLAND vocabulary (HK 軟件 vs TW 軟體; HK
巴士/的士/冷氣/雪櫃/質素 vs TW 公車[actually TW also uses 巴士
regionally]/計程車/空調/冰箱/素質). OpenCC ships s2twp AND s2hk
separately for exactly this reason. Therefore THREE models:

| Variant | Script | Vocabulary | Corpus state |
|---|---|---|---|
| zh-Hans-CN | Simplified | mainland | CLEAN today (csc_data 85% simp; Wikipedia simp-only lines; lang8 simp lines) |
| zh-Hant-TW | Traditional | Taiwan | GOOD: CC-100 zh-Hant / Traditional C4 (`jed351/Traditional-Chinese-Common-Crawl-Filtered`, 898 shards 2013-2025, no explicit license tag - Common Crawl basis) + Wikipedia trad-only lines; TW share separable via TW-vocab markers |
| zh-Hant-HK | Traditional | Hong Kong | NO ready open corpus with a clean license tag on HF today (verified 2026-09-16) |

### Hong Kong Chinese corpus (verified findings)

- `Swithord/hong-kong-legco-hansard` (813,034 speech records,
  1985-2025) is the **ENGLISH** record — the scrape pipeline
  (github.com/Swithord/hong-kong-legco-transcript) parses the same
  LegCo source that carries the **Traditional Chinese original**
  (官方紀錄): re-run the scraper against the Chinese side = the
  large, license-clean (HK government open data) formal HK written
  corpus. The single best HK lead.
- `zetavg/CC-100-zh-Hant-merged` (36 shards) and
  `jed351/Traditional-Chinese-Common-Crawl-Filtered` (898 shards):
  large Traditional web text, MIXED TW+HK — the HK share is
  isolatable downstream via HK-vocabulary markers (巴士/的士/冷氣/
  雪櫃/質素/立法會/行政長官 density), which the variant split needs
  anyway.
- HKSAR press releases (news.gov.hk) + RTHK archives: open HK
  government Traditional text, scrapeable.
- Written CANTONESE (yue: 嘅/係/唔) is a DIFFERENT written language,
  not the zh-Hant-HK spellchecker target: resources exist
  (`nanyang-technological-university-singapore/hkcancor` CC-BY-4.0
  spoken; `jed351/cantonese-wikipedia`; `ziyou-li/cantonese_daily`)
  and are OUT OF SCOPE for this arc unless a yue product is wanted.
- Typo/eval data for HK: no HK-specific CSC set found; nearest real
  data = native SIGHAN 2013/2014 Traditional originals + Lang-8
  trad lines (0.1%, thin) + OpenCC s2hk conversion of csc_data for
  TRAINING signal only. A real HK learner corpus is an open gap —
  flagged as an acquisition item (HK student essay corpora exist
  academically, e.g. through CUHK, but are not openly downloadable).

### arXiv 2025/2026: the Hong Kong / Cantonese data papers (second pass)

- **arXiv 2503.03702** (2025-03): large-scale Cantonese corpus — 2B
  tokens mined from open corpora, HK-specific forums, Wikipedia,
  Common Crawl, with the filtering/dedup pipeline published. Written
  CANTANESE (yue) — out of scope for formal zh-Hant-HK, but the
  HK-source mining recipe is reusable.
- **arXiv 2507.11502** (2025-07, HKGAI-V1): HK sovereign LLM for the
  Cantonese/Mandarin/English environment — the region's data
  curation reference; not an open corpus itself.
- **arXiv 2606.06679** (2026-06, HKJudge): ~290k sentences / 6.5M
  tokens of HK court judgments, expert-annotated — REAL formal
  Traditional HK written Chinese (legal domain); a strong
  formal-register eval slice once released, license to verify.
- **arXiv 2509.20557** (2025-09, SiniticMTError): error-span/type
  annotations for MT into Mandarin + Cantonese — variant-confusion
  signal, MT domain.
- No Cantonese/CSC spelling-correction corpus paper exists (the
  "Cantonese spelling" query returned nothing) — the HK learner
  typo-eval gap is confirmed by absence, not just by my HF search.

PERMANENT RULE (owner, 2026-09-16): never ship script-mixed or
region-mixed CJK models again — the shipped script-mixed zh tiers
(22.2% Traditional training lines) are the violation this rule
forecloses; the variant models REPLACE them when plan 18a lands.

## The crawl-model zh bucket route is CLOSED (third confirmation)

The cc.zh.300 binary was re-gated with the REAL CSC corpus
(eval/corpora/zh-csc.json, 19,212 unique pairs) on 2026-09-16: the
demand computation found **1** usable marked-5-gram row and 1 gate
probe — because the CSC error class is single-character (那/哪, 在/再)
and `<char>` produces no 5-grams at minn=maxn=5. Evidence:
eval/reports/zh.buckets.csc-corpus.json. Three independent proofs now
agree (Phase-0 intended_top1 0.01; the cross-space sibling probe; the
real-corpus demand starvation): **no corpus can fix a model that
stores no rows the CJK error class can hit. The per-variant
same-space models are the only route to CJK buckets.**

## Recommendation

Adopt twnlp/csc_data (MIT) as the zh-Hans typo corpus — gates
unblocked at 19k unique pairs; keep Wang271k lines for training
signal only; hold out SIGHAN15 + real subsets for eval. Build zh-Hans-CN first (fully clean sources), zh-Hant-TW second
(CC-100 zh-Hant / Traditional C4 + TW-vocab separation + native SIGHAN
originals), zh-Hant-HK third (LegCo Chinese Hansard scrape +
CC-Traditional HK-share filtering). The registry-code fork (recommend (i) BCP-47) and
any Lang-8 licensing review (upstream Lang-8 terms vs the aggregate's
MIT marker — we ship statistics, not text) are owner calls before the
zh-Hant cut. The same-space model route (plan 18a) proceeds per
variant once the code decision lands.
