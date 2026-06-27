#!/usr/bin/env python3
"""
Get vocabulary size from ONNX model.

Usage:
    python3 scripts/get_vocab_size.py <model_path>
"""
import sys
import onnx

def get_vocab_size(model_path):
    """Extract vocabulary size from ONNX model embedding matrix."""
    model = onnx.load(model_path)

    # The Constant node contains the embedding matrix
    # Find the Constant node and get its shape
    for node in model.graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value':
                    # The tensor shape is [vocab_size, embedding_dim]
                    # Access dims from the tensor proto
                    tensor = onnx.helper.get_attribute_value(attr)
                    vocab_size = tensor.dims[0]
                    print(vocab_size)
                    return 0

    print("ERROR: Could not find vocab size", file=sys.stderr)
    return 1

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_path>", file=sys.stderr)
        sys.exit(1)

    sys.exit(get_vocab_size(sys.argv[1]))
