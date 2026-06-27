#!/usr/bin/env python3
"""
Sample embeddings from ONNX model for test fixture generation.
Outputs JSON with statistics and sample values.
"""
import sys
import json
import onnxruntime as ort
import numpy as np

def sample_embedding(model_path, index, num_samples=10):
    """Sample embedding at given index, return stats and sample values."""
    try:
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # Run inference
        embedding = sess.run([output_name], {input_name: np.array([index], dtype=np.int64)})[0]

        # Calculate statistics
        stats = {
            'mean': round(float(np.mean(embedding)), 6),
            'std': round(float(np.std(embedding)), 6),
            'min': round(float(np.min(embedding)), 6),
            'max': round(float(np.max(embedding)), 6)
        }

        # Sample first N values (embedding is already 1D with shape (300,))
        samples = [round(float(v), 6) for v in embedding.flatten()[:num_samples]]

        output = {'stats': stats, 'samples': samples}
        print(json.dumps(output))
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <model_path> <index>", file=sys.stderr)
        sys.exit(1)

    model_path = sys.argv[1]
    index = int(sys.argv[2])

    sys.exit(sample_embedding(model_path, index))
