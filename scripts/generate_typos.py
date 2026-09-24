#!/usr/bin/env python3
"""S1: layout-grounded synthetic typo generator (TODO.sota/1, impl 1).

Error classes, per language:
  adjacent-sub      key-adjacent substitution (layout-distance-1, dual-layout aware)
  far-sub           non-adjacent substitution
  transposition     adjacent letter swap
  double-insert     doubled-letter insertion ("helo"->"hello")
  double-delete     doubled-letter deletion ("hello"->"helo")
  diacritic-omit    fold-equal char swap (o for ö — plan C9 fold)
  ime-confusion     pinyin homophone char swap (zh, from CC-CEDICT pinyin)
  realword-swap     valid neighbor at edit distance 1 (both words real)

Splits carry per-pair class tags; the realism report compares the
edit-distance distribution and class mix against the human splits.
"""
import argparse
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LAYOUTS_PATH = Path("/tmp/layout-grids.json")

FOLD_EXCEPTIONS = {"ß": "ss"}


def fold_word(word):
    s = str(word).lower()
    if not s:
        return s
    s = s.encode("utf-8", "replace").decode("utf-8")
    out = []
    for ch in s:
        if ch in FOLD_EXCEPTIONS:
            out.append(FOLD_EXCEPTIONS[ch])
            continue
        try:
            d = unicodedata.normalize("NFD", ch)
        except (ValueError, UnicodeError):
            out.append(ch)
            continue
        stripped = "".join(c for c in d if not unicodedata.combining(c))
        out.append(stripped if stripped else ch)
    return "".join(out)


def load_layout_for(lang, layouts):
    for l in layouts:
        if lang in l["language_codes"]:
            return l["key_positions"]
    base = lang.split("-")[0]
    for l in layouts:
        if base in l["language_codes"]:
            return l["key_positions"]
    for l in layouts:
        if l["name"] == "QWERTY":
            return l["key_positions"]
    return {}


def adjacency(positions):
    adj = defaultdict(list)
    items = list(positions.items())
    for (c1, p1) in items:
        for (c2, p2) in items:
            if c1 == c2:
                continue
            d = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
            if d == 1:
                adj[c1].append(c2)
    return adj


def load_words(lang):
    cache = Path.home() / ".cache/kotoshu/frequency-lists" / lang / "frequency.json"
    if not cache.exists():
        import urllib.request
        url = f"https://raw.githubusercontent.com/kotoshu/frequency-list-kelly/main/data/{lang}.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "kotoshu-s1/1.0"})
        cache.write_bytes(urllib.request.urlopen(req).read())
    data = json.loads(cache.read_text())
    words = [e["word"] for e in data.get("full_list", [])]
    ranks = {e["word"].lower(): e["rank"] for e in data.get("full_list", [])}
    return words, ranks


def load_jyutping_groups():
    """zh-Hant-HK IME class: char -> jyutping syllable from Unihan
    kCantonese (scripts/build_jyutping_table.py writes the table)."""
    path = Path("/tmp/cjk/jyutping.txt")
    if not path.exists():
        return {}
    by_char = defaultdict(set)
    by_syl = defaultdict(list)
    seen = defaultdict(set)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            ch, _, syl = line.strip().partition("\t")
            if ch and syl and ch not in seen[syl]:
                seen[syl].add(ch)
                by_char[ch].add(syl)
                by_syl[syl].append(ch)
    return by_char, by_syl


def load_pinyin_groups():
    """zh IME class: char -> pinyin syllable from CC-CEDICT."""
    path = Path("/tmp/cjk/cedict.txt")
    if not path.exists():
        return {}
    line_re = None
    import re
    line_re = re.compile(r"^(\S+)\s+(\S+)\s+\[([^\]]+)\]")
    groups = defaultdict(set)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            m = line_re.match(line.strip())
            if not m:
                continue
            syls = m.group(3).split()
            for tok in (m.group(1), m.group(2)):
                if len(tok) == len(syls):
                    for ch, syl in zip(tok, syls):
                        groups[ch.lower()].add(syl.split("5")[0].lower())
    by_syl = defaultdict(list)
    for ch, syls in groups.items():
        for s in syls:
            by_syl[s].append(ch)
    return groups, by_syl


class TypoGen:
    def __init__(self, lang, layouts, seed):
        self.lang = lang
        self.rng = random.Random(seed)
        self.words, self.ranks = load_words(lang)
        self.positions = load_layout_for(lang, layouts)
        self.adj = adjacency(self.positions)
        self.word_set = set(w.lower() for w in self.words)
        self.ime_char = {}
        self.ime_syls = {}
        if lang == "zh-Hant-HK":
            self.ime_char, self.ime_syls = load_jyutping_groups()
        elif lang.startswith("zh"):
            self.ime_char, self.ime_syls = load_pinyin_groups()
        # Frequent own-script chars for the realword pool (non-Latin
        # scripts never collide with the Latin fallback pool).
        head_chars = defaultdict(int)
        for w in self.words[:2000]:
            for ch in w:
                if not ch.isascii():
                    head_chars[ch] += 1
        self.own_pool = "".join(
            ch for ch, _ in sorted(head_chars.items(), key=lambda kv: -kv[1])[:80]
        )

    def _neighbors(self, ch):
        return self.adj.get(ch) or self.adj.get(fold_word(ch)) or []

    VIET_TONES = "\u0300\u0301\u0309\u0303\u0323"  # `  '  ?  ~  .

    def tone_mark_swap(self, w):
        """vi: rotate the tone mark on one precomposed vowel (same
        letters, wrong dấu — the telex/VIQR error class)."""
        idxs = [i for i, ch in enumerate(w) if ch.lower() in "aeiouyàáâãăằắẵầấẩǣèéêẽẹềếểễệìíĩịòóôõọồốỗộờớởỡùúũụừứửữỳýỹỵ"]
        if not idxs:
            return None
        import unicodedata as _ud
        i = self.rng.choice(idxs)
        ch = w[i]
        d = _ud.normalize("NFD", ch)
        tone = next((c for c in d if c in self.VIET_TONES), None)
        if tone is None:
            return None
        others = [t for t in self.VIET_TONES if t != tone]
        rep = _ud.normalize("NFC", d.replace(tone, self.rng.choice(others)))
        if rep == ch:
            return None
        return w[:i] + rep + w[i + 1:], "tone-mark"

    AR_SETS = ({"أ", "إ", "آ", "ا"}, {"ة", "ه"}, {"ى", "ي"})

    def hamza_swap(self, w):
        """ar: hamza / taa-marbuta / alif maqsura confusables."""
        for k, s in enumerate(self.AR_SETS):
            idxs = [i for i, ch in enumerate(w) if ch in s]
            if idxs:
                i = self.rng.choice(idxs)
                alts = [c for c in s if c != w[i]]
                return w[:i] + self.rng.choice(alts) + w[i + 1:], "hamza-swap"
        return None

    KO_JAMO_PAIRS = [("되", "돼"), ("안", "않"), ("개", "게"), ("애", "에"),
                     ("내", "네"), ("해", "헤"), ("배", "베"), ("새", "세"),
                     ("재", "제"), ("채", "체"), ("캐", "케"), ("태", "테"),
                     ("패", "페"), ("래", "레"), ("얘", "예"), ("쉐", "셰"),
                     ("외", "웨"), ("죄", "졔"), ("되", "데"), ("요", "용")]

    def jamo_swap(self, w):
        """ko: composed-syllable confusables (ㅐ/ㅔ, ㅚ/ㅞ class)."""
        for a, b in self.KO_JAMO_PAIRS:
            if a in w:
                i = w.index(a)
                return w[:i] + b + w[i + 1:], "jamo-confusion"
            if b in w:
                i = w.index(b)
                return w[:i] + a + w[i + 1:], "jamo-confusion"
        return None

    def language_class(self, w):
        if self.lang == "vi":
            return self.tone_mark_swap(w)
        if self.lang == "ar":
            return self.hamza_swap(w)
        if self.lang == "ko":
            return self.jamo_swap(w)
        return None

    def typo_of(self, word):
        """Return (typo, class) or None."""
        w = word
        if len(w) < 3 or not w.isalpha():
            return None
        # language-specific orthographic confusions first (15%),
        # then the generic layout chain
        if self.rng.random() < 0.15:
            res = self.language_class(w)
            if res:
                return res
        r = self.rng.random()
        # class mix: adjacent 30%, transposition 15%, double-insert 15%,
        # double-delete 10%, diacritic 10%, far-sub 10%, ime/realword rest
        idx = self.rng.randrange(len(w))
        if r < 0.30:
            nb = self._neighbors(w[idx].lower())
            if nb:
                rep = self.rng.choice(nb)
                if self.rng.random() < 0.5:
                    rep = rep.upper() if w[idx].isupper() else rep
                return w[:idx] + rep + w[idx + 1:], "adjacent-sub"
            fallback = ("abcdefghijklmnopqrstuvwxyz"
                        if w[idx].isascii() else (self.own_pool or "aeiounstr"))
            rep = self.rng.choice(fallback)
            return w[:idx] + rep + w[idx + 1:], "far-sub"
        if r < 0.45 and idx < len(w) - 1:
            if w[idx].lower() != w[idx + 1].lower():
                return w[:idx] + w[idx + 1] + w[idx] + w[idx + 2:], "transposition"
            return None
        if r < 0.60:
            return w[:idx + 1] + w[idx] + w[idx + 1:], "double-insert"
        if r < 0.70 and len(w) > 3:
            dd = [i for i in range(len(w) - 1) if w[i].lower() == w[i + 1].lower()]
            if dd:
                i = self.rng.choice(dd)
                return w[:i] + w[i + 1:], "double-delete"
            return None
        if r < 0.80:
            for _ in range(8):
                i = self.rng.randrange(len(w))
                import unicodedata as _ud2
                decomp = _ud2.normalize("NFD", w[i])
                if not any(_ud2.combining(c) for c in decomp):
                    return None
                f = fold_word(w[i])
                if f != w[i].lower():
                    plain = self.rng.choice(f)
                    rep = plain.upper() if w[i].isupper() else plain
                    return w[:i] + rep + w[i + 1:], "diacritic-omit"
            return None
        if r < 0.90 and self.ime_char:
            han = [c for c in w if c.lower() in self.ime_char]
            if han:
                ch = self.rng.choice(han)
                syl = self.rng.choice(list(self.ime_char[ch.lower()]))
                alts = [c for c in self.ime_syls.get(syl, []) if c != ch]
                if alts:
                    alt = self.rng.choice(alts)
                    i = w.lower().index(ch)
                    return w[:i] + alt + w[i + 1:], "ime-confusion"
            return None
        # far-sub fallback — own-script chars for non-Latin positions
        i = self.rng.randrange(len(w))
        if w[i].isascii():
            pool = "abcdefghijklmnopqrstuvwxyz"
        else:
            pool = self.own_pool or "aeiounstr"
        rep = self.rng.choice(pool)
        if rep == w[i].lower():
            return None
        return w[:i] + rep + w[i + 1:], "far-sub"

    def confusion_variants(self, w):
        """All single-confusable variants of w, deterministically
        (the constructive counterpart of language_class/ime swaps)."""
        out = []
        import unicodedata as _ud
        if self.lang == "vi":
            for i, ch in enumerate(w):
                d = _ud.normalize("NFD", ch)
                tone = next((c for c in d if c in self.VIET_TONES), None)
                if tone is None:
                    continue
                for t in self.VIET_TONES:
                    if t == tone:
                        continue
                    out.append((_ud.normalize("NFC", d.replace(tone, t)), "tone-mark"))
        elif self.lang == "ar":
            for i, ch in enumerate(w):
                for s in self.AR_SETS:
                    if ch in s:
                        for alt in s:
                            if alt != ch:
                                out.append((w[:i] + alt + w[i + 1:], "hamza-swap"))
        elif self.lang == "ko":
            for a, b in self.KO_JAMO_PAIRS:
                if a in w:
                    out.append((w.replace(a, b, 1), "jamo-confusion"))
                if b in w:
                    out.append((w.replace(b, a, 1), "jamo-confusion"))
        if self.ime_char:
            for i, ch in enumerate(w):
                syls = self.ime_char.get(ch.lower())
                if not syls:
                    continue
                for syl in syls:
                    for alt in self.ime_syls.get(syl, []):
                        if alt != ch:
                            out.append((w[:i] + alt + w[i + 1:], "ime-confusion"))
        return out

    def skeleton_realword_pairs(self, limit):
        """vi real-word pairs: words sharing a tone-stripped skeleton
        (chấu/chầu, hài/hại) — both sides real by construction, and the
        vowel-cluster changes the tone-rotation model cannot enumerate
        are covered."""
        import unicodedata as _ud
        groups = defaultdict(list)
        for w in self.words:
            if len(w) < 2:
                continue
            d = _ud.normalize("NFD", w)
            key = "".join(c for c in d if c not in self.VIET_TONES)
            groups[key].append(w)
        out, seen = [], set()
        for key, ws in groups.items():
            if len(out) >= limit:
                break
            if len(ws) < 2:
                continue
            for i in range(len(ws)):
                if len(out) >= limit:
                    break
                for j in range(i + 1, len(ws)):
                    a, b = ws[i], ws[j]
                    if a.lower() in seen:
                        continue
                    seen.add(a.lower())
                    out.append({"typo": a, "correction": b,
                                "weight": 1, "class": "tone-mark-realword"})
                    break
        return out

    def confusion_realword_pairs(self, limit):
        """Constructive real-word pairs: enumerate the confusion variants
        of each list word and keep the ones where BOTH sides are real
        words. Stable across RNG states (the rejection sampler exhausted
        its proposal space and collapsed to single digits)."""
        if self.lang == "vi":
            return self.skeleton_realword_pairs(limit)
        out, seen = [], set()
        for w in self.words:
            if len(out) >= limit:
                break
            if len(w) < 2:
                continue
            for cand, klass in self.confusion_variants(w):
                cl = cand.lower()
                if len(cand) < 2 or len(w) < 2:
                    continue
                if cl == w.lower() or cl in seen or cl not in self.word_set:
                    continue
                seen.add(cl)
                out.append({"typo": cand, "correction": w,
                            "weight": 1, "class": klass + "-realword"})
                break
        return out

    def confusion_realword(self, attempts=6000):
        """Real-word pairs from the confusion tables: swap a confusable
        and keep the pair when BOTH sides are real words (the actual
        real-word error class — 那/哪, the wrong-dấu spelling, 개/게,
        the hamza variants)."""
        for _ in range(attempts):
            # full list: both-sides-real is guaranteed by word_set
            # membership, so the frequency guard is not needed here
            w = self.rng.choice(self.words)
            if len(w) < 2:
                continue
            res = self.language_class(w)
            if res:
                cand, klass = res
                if cand.lower() in self.word_set:
                    return cand, w, klass + "-realword"
            if self.ime_char:
                han = [c for c in w if c.lower() in self.ime_char]
                if han:
                    ch = self.rng.choice(han)
                    syl = self.rng.choice(list(self.ime_char[ch.lower()]))
                    alts = [c for c in self.ime_syls.get(syl, []) if c != ch]
                    if alts:
                        alt = self.rng.choice(alts)
                        i = w.lower().index(ch)
                        cand = w[:i] + alt + w[i + 1:]
                        if cand.lower() in self.word_set:
                            return cand, w, "ime-confusion-realword"
        return None

    def realword_pair(self, attempts=4000):
        """Valid neighbor at distance 1, both words real. Any single
        substitution (not only adjacent-key) — valid-word collisions
        are rare in vowel-dense languages."""
        for _ in range(attempts):
            w = self.rng.choice(self.words[:max(1, len(self.words) // 2)])
            if len(w) < 4:
                continue
            i = self.rng.randrange(len(w))
            pool = "abcdefghijklmnopqrstuvwxyzàáâäèéêëìíîïòóôöùúûüñçãõåøæœ"
            if not w[i].isascii() and self.own_pool:
                pool = self.own_pool
            pool += w[i].lower()
            rep = self.rng.choice(pool)
            if rep == w[i].lower():
                continue
            cand = w[:i] + rep + w[i + 1:]
            if cand.lower() in self.word_set:
                return cand, w  # "typo"=valid word, correction=original
        return None

    def edit_distance(self, a, b):
        la, lb = len(a), len(b)
        prev = list(range(lb + 1))
        for i in range(1, la + 1):
            cur = [i]
            for j in range(1, lb + 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                               prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1)))
            prev = cur
        return prev[lb]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--nonword", type=int, default=2000)
    ap.add_argument("--realword", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "eval" / "realword"))
    args = ap.parse_args()

    layouts = json.loads(LAYOUTS_PATH.read_text())["layouts"]
    gen = TypoGen(args.lang, layouts, args.seed)

    pairs, seen = [], set()
    classes = Counter()
    dists = Counter()
    while len(pairs) < args.nonword:
        if not gen.words:
            break
        w = gen.rng.choice(gen.words)
        if gen.word_set and w.lower() not in gen.word_set:
            continue
        res = gen.typo_of(w)
        if not res:
            continue
        typo, klass = res
        if typo == w or typo.lower() in gen.word_set or typo.lower() in seen:
            continue
        seen.add(typo.lower())
        d = gen.edit_distance(typo.lower(), w.lower())
        if d == 0 or d > 2:
            continue
        pairs.append({"typo": typo, "correction": w, "weight": 1, "class": klass})
        classes[klass] += 1
        dists[d] += 1

    rw = list(gen.confusion_realword_pairs(args.realword))
    seen_rw = {p["typo"].lower() for p in rw}
    attempts = 0
    while len(rw) < args.realword and attempts < 40:
        attempts += 1
        res = gen.realword_pair()
        if not res:
            continue
        typo, truth = res
        if typo.lower() in seen_rw:
            continue
        seen_rw.add(typo.lower())
        rw.append({"typo": typo, "correction": truth, "weight": 1, "class": "realword-swap"})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for klass, rows in (("nonword", pairs), ("realword", rw)):
        payload = {"spec": "kotoshu.suggest-benchmark/v2-synthetic", "language": args.lang,
                   "klass": klass, "n": len(rows), "pairs": rows}
        path = out / f"{args.lang}.suggest2-{klass}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        print(f"wrote {path} n={len(rows)}")

    report = {"language": args.lang, "class_mix": dict(classes),
              "distance_dist": dict(dists), "nonword": len(pairs), "realword": len(rw)}
    rp = out / f"{args.lang}.suggest2-report.json"
    rp.write_text(json.dumps(report, indent=1))
    print("class mix:", dict(classes))


if __name__ == "__main__":
    main()
