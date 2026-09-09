"""Language pack builder tests (plan 113): framing, round-trip,
corruption detection, and the cross-implementation golden bytes.

The golden pack pins the KPK1 framing against the kotoshu-rs pack
reader (kotoshu/src/pack.rs in kotoshu-rs): both suites build a pack
from the SAME synthetic sections and assert the SAME whole-file
sha256 - the two implementations cannot drift apart silently.
"""

import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_packs import (  # noqa: E402
    MAGIC, TAG_AFF, TAG_BUCKETS, TAG_DIC, TAG_MODEL, TAG_VOCAB, build_pack,
    parse_pack, section_offsets,
)

# The synthetic sections both repos pin (models repo: this file;
# kotoshu-rs: kotoshu/src/pack.rs GOLDEN_SECTIONS).
GOLDEN_SECTIONS = {
    "aff": b"SET UTF-8\nTRY esianrtolcdugmphbyfvkwzESIANRTOLCDUGMPHBYFVKWZ\n",
    "dic": "3\nhello/M\nworld\nkotoshu\n".encode(),
    "model": bytes(range(64)) * 4,          # 256 deterministic bytes
    "vocab": b'{"vocab_size":3,"word_to_idx":{"hello":0,"kotoshu":1,"world":2}}',
    "buckets": bytes(((i * 7) % 256) for i in range(300)),
}
GOLDEN_SHA256 = "5f124baa3fcae30625abb0e3297479ae5d05dad411c97d08e0e83f106f09843b"
GOLDEN_SIZE = 898

TAG_OF = {"aff": TAG_AFF, "dic": TAG_DIC, "model": TAG_MODEL,
          "vocab": TAG_VOCAB, "buckets": TAG_BUCKETS}


class BuildPackTest(unittest.TestCase):
    def test_round_trip_all_sections(self):
        data = build_pack(GOLDEN_SECTIONS)
        parsed = parse_pack(data)
        for name, payload in GOLDEN_SECTIONS.items():
            self.assertEqual(parsed[TAG_OF[name]], payload, name)

    def test_round_trip_without_buckets(self):
        sections = dict(GOLDEN_SECTIONS, buckets=None)
        data = build_pack(sections)
        parsed = parse_pack(data)
        self.assertNotIn(TAG_BUCKETS, parsed)
        for name in ("aff", "dic", "model", "vocab"):
            self.assertEqual(parsed[TAG_OF[name]], sections[name])

    def test_golden_bytes_pinned(self):
        # The cross-repo framing contract: kotoshu-rs pack.rs builds the
        # same sections and must produce the same whole-file sha256.
        data = build_pack(GOLDEN_SECTIONS)
        self.assertEqual(len(data), GOLDEN_SIZE)
        self.assertEqual(hashlib.sha256(data).hexdigest(), GOLDEN_SHA256)

    def test_header_shape(self):
        data = build_pack(GOLDEN_SECTIONS)
        self.assertEqual(data[:4], MAGIC)
        self.assertEqual(int.from_bytes(data[4:8], "little"), 5)
        # First section: u32 LE length then tag byte, payload at 8+5.
        length = int.from_bytes(data[8:12], "little")
        self.assertEqual(data[12], TAG_AFF)
        self.assertEqual(length, len(GOLDEN_SECTIONS["aff"]))

    def test_offsets_match_framing(self):
        data = build_pack(GOLDEN_SECTIONS)
        offsets = section_offsets(data)
        for name, payload in GOLDEN_SECTIONS.items():
            offset, length = offsets[TAG_OF[name]]
            self.assertEqual(length, len(payload))
            self.assertEqual(data[offset : offset + length], payload)

    def test_empty_required_section_rejected(self):
        with self.assertRaises(ValueError):
            build_pack(dict(GOLDEN_SECTIONS, dic=b""))


class ParsePackTest(unittest.TestCase):
    def setUp(self):
        self.data = build_pack(GOLDEN_SECTIONS)

    def test_bad_magic(self):
        with self.assertRaises(ValueError):
            parse_pack(b"XXXX" + self.data[4:])

    def test_tampered_payload_fails_footer(self):
        data = bytearray(self.data)
        offsets = section_offsets(self.data)
        offset, _ = offsets[TAG_MODEL]
        data[offset] ^= 0xFF
        with self.assertRaises(ValueError):
            parse_pack(bytes(data))

    def test_tampered_footer_fails(self):
        data = bytearray(self.data)
        offsets = section_offsets(self.data)
        offset, length = offsets[TAG_DIC]
        footer = offset + length
        data[footer] ^= 0xFF
        with self.assertRaises(ValueError):
            parse_pack(bytes(data))

    def test_truncated_payload(self):
        with self.assertRaises(ValueError):
            parse_pack(self.data[:-1])

    def test_truncated_header(self):
        with self.assertRaises(ValueError):
            parse_pack(self.data[:7])

    def test_trailing_bytes(self):
        with self.assertRaises(ValueError):
            parse_pack(self.data + b"\x00")

    def test_count_mismatch(self):
        data = bytearray(self.data)
        data[4:8] = (6).to_bytes(4, "little")
        with self.assertRaises(ValueError):
            parse_pack(bytes(data))

    def test_duplicate_section(self):
        # Re-frame one section and append it; the count says 5+1 and the
        # duplicate tag must be rejected.
        offsets = section_offsets(self.data)
        extra = self.data[8 : offsets[TAG_DIC][0] + 5 + len(GOLDEN_SECTIONS["dic"]) + 32]
        data = bytearray(self.data[:8])
        data[4:8] = (6).to_bytes(4, "little")
        data += self.data[8:] + extra[5:]
        with self.assertRaises(ValueError):
            parse_pack(bytes(data))


if __name__ == "__main__":
    unittest.main()
