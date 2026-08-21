"""
onnx_quantize.py
=================
Optional utility: export the encoder model to ONNX and apply real INT8
dynamic quantization via onnxruntime, for CPU inference. This is the honest,
practical real-world equivalent of the architecture's "4-bit quantized ONNX
model" stage — see the note in encoder.py for why 4-bit is not standard
tooling for a model this size, and INT8 dynamic quantization is what's
actually used in production for CPU-only BERT-sized encoders.

Not required to run the app (app.py's default encoder uses torch's built-in
dynamic quantization, no ONNX export needed). Run this only if you want an
exported .onnx file, e.g. to serve the encoder from a non-Python runtime.

Usage:
    pip install onnx onnxruntime optimum[onnxruntime]
    python onnx_quantize.py
"""

import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "onnx_model"


def export_and_quantize():
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Exporting {MODEL_NAME} to ONNX...")
    model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Applying INT8 dynamic quantization...")
    quantizer = ORTQuantizer.from_pretrained(OUTPUT_DIR)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=OUTPUT_DIR, quantization_config=qconfig)

    print(f"Done. Quantized ONNX model written to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    export_and_quantize()
