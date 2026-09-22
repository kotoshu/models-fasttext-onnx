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
    return by_syl


class TypoGen:
    def __init__(self, lang, layouts, seed):
        self.lang = lang
        self.rng = random.Random(seed)
        self.words, self.ranks = load_words(lang)
        self.positions = load_layout_for(lang, layouts)
        self.adj = adjacency(self.positions)
        self.word_set = set(w.lower() for w in self.words)
        self.pinyin = load_pinyin_groups() if lang.startswith("zh") else {}

    def _neighbors(self, ch):
        return self.adj.get(ch) or self.adj.get(fold_word(ch)) or []

    def typo_of(self, word):
        """Return (typo, class) or None."""
        w = word
        if len(w) < 3 or not w.isalpha():
            return None
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
            rep = self.rng.choice("abcdefghijklmnopqrstuvwxyz")
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
                f = fold_word(w[i])
                if f != w[i].lower():
                    plain = self.rng.choice(f)
                    rep = plain.upper() if w[i].isupper() else plain
                    return w[:i] + rep + w[i + 1:], "diacritic-omit"
            return None
        if r < 0.90 and self.pinyin:
            han = [c for c in w if c.lower() in self.pinyin]
            if han:
                ch = self.rng.choice(han)
                syl = self.rng.choice(list(self.pinyin[ch.lower()]))
                alts = [c for c in self.pinyin.get(syl, []) if c != ch]
                if alts:
                    alt = self.rng.choice(alts)
                    i = w.lower().index(ch)
                    return w[:i] + alt + w[i + 1:], "ime-confusion"
            return None
        # far-sub fallback
        i = self.rng.randrange(len(w))
        pool = "abcdefghijklmnopqrstuvwxyz" if w[i].isascii() else "aeiounstr"
        rep = self.rng.choice(pool)
        if rep == w[i].lower():
            return None
        return w[:i] + rep + w[i + 1:], "far-sub"

    def realword_pair(self, attempts=4000):
        """Valid neighbor at distance 1, both words real. Any single
        substitution (not only adjacent-key) — valid-word collisions
        are rare in vowel-dense languages."""
        for _ in range(attempts):
            w = self.rng.choice(self.words[:max(1, len(self.words) // 2)])
            if len(w) < 4:
                continue
            i = self.rng.randrange(len(w))
            pool = "abcdefghijklmnopqrstuvwxyzàáâäèéêëìíîïòóôöùúûüñçãõåøæœ" + w[i].lower()
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

    rw = []
    attempts = 0
    while len(rw) < args.realword and attempts < 40:
        attempts += 1
        res = gen.realword_pair()
        if not res:
            continue
        typo, truth = res
        if typo.lower() in {p["typo"].lower() for p in rw}:
            continue
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
