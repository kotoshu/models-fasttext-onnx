"""Conversion regression test for scripts/fasttext_to_onnx.py."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import fasttext_to_onnx

VOCAB_SIZE = 10
DIM = 5
WORDS = [
    "alpha", "beta", "gamma", "delta", "epsilon",
    "zeta", "eta", "theta", "iota", "kappa",
]


def vec_value(row, col):
    return (row * DIM + col + 1) / 100.0


class FasttextToOnnxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmpdir = Path(cls._tmp.name)
        cls.vec_path = tmpdir / "toy.vec"
        cls.onnx_path = tmpdir / "toy.onnx"
        cls.vocab_path = tmpdir / "toy.vocab.json"
        lines = [f"{VOCAB_SIZE} {DIM}"]
        for i, word in enumerate(WORDS):
            values = " ".join(f"{vec_value(i, j):.6f}" for j in range(DIM))
            lines.append(f"{word} {values}")
        cls.vec_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cls.word_to_idx, cls.embeddings, cls.metadata = fasttext_to_onnx.parse_fasttext_vec(
            str(cls.vec_path)
        )
        model = fasttext_to_onnx.create_onnx_model(cls.embeddings, cls.word_to_idx)
        onnx.save(model, str(cls.onnx_path))
        fasttext_to_onnx.save_vocabulary(cls.word_to_idx, str(cls.vocab_path))
        cls.session = ort.InferenceSession(
            str(cls.onnx_path), providers=["CPUExecutionProvider"]
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def embedding_for(self, index):
        outputs = self.session.run(None, {"word_index": np.array([index], dtype=np.int64)})
        self.assertEqual(len(outputs), 1)
        return outputs[0]

    def test_parse_reads_toy_vec(self):
        self.assertEqual(self.metadata["vocab_size"], VOCAB_SIZE)
        self.assertEqual(self.metadata["embedding_dim"], DIM)
        self.assertEqual(self.embeddings.shape, (VOCAB_SIZE, DIM))
        self.assertEqual(self.embeddings.dtype, np.float32)

    def test_saved_model_passes_onnx_checker(self):
        model = onnx.load(str(self.onnx_path))
        onnx.checker.check_model(model)
        self.assertEqual(model.opset_import[0].version, 11)
        self.assertEqual(self.session.get_inputs()[0].name, "word_index")
        self.assertEqual(self.session.get_outputs()[0].name, "embedding")

    def test_inference_output_shape(self):
        output = self.embedding_for(0)
        self.assertEqual(output.shape, (DIM,))

    def test_inference_values_match_vec_rows(self):
        for index in (0, VOCAB_SIZE // 2, VOCAB_SIZE - 1):
            output = self.embedding_for(index)
            expected = np.array(
                [vec_value(index, j) for j in range(DIM)], dtype=np.float32
            )
            np.testing.assert_allclose(output, expected, atol=1e-6)

    def test_vocab_json_sidecar(self):
        data = json.loads(self.vocab_path.read_text(encoding="utf-8"))
        self.assertEqual(data["vocab_size"], VOCAB_SIZE)
        self.assertEqual(data["word_to_idx"], {w: i for i, w in enumerate(WORDS)})


if __name__ == "__main__":
    unittest.main()
