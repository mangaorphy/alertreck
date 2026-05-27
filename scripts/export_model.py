"""
Export trained CNN to ONNX for Raspberry Pi deployment.

Usage:
    python3 scripts/export_model.py
    python3 scripts/export_model.py --model models/best_model.pt --out models/custom_cnn/alertreck_cnn.onnx

Requires: torch, onnx, onnxruntime (pip install onnx onnxruntime)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ── Model definition (must match notebooks/03a_train_cnn.ipynb) ──────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class AudioCNN(nn.Module):
    """Input: (B, 1, 128, 259)  →  Output: (B, 7)"""
    def __init__(self, n_classes: int = 7):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1,   32,  dropout=0.2),
            ConvBlock(32,  64,  dropout=0.2),
            ConvBlock(64,  128, dropout=0.2),
            ConvBlock(128, 256, dropout=0.2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.encoder(x)))


# ── Export ────────────────────────────────────────────────────────────────────

def export(model_pt: Path, onnx_out: Path) -> None:
    print(f"Loading checkpoint: {model_pt}")
    checkpoint = torch.load(model_pt, map_location="cpu")

    model = AudioCNN(n_classes=7)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Best epoch : {checkpoint.get('epoch', '?')}")
    print(f"Val acc    : {checkpoint.get('val_acc', '?'):.4f}")

    dummy = torch.randn(1, 1, 128, 259)
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_out),
        input_names=["mel_spec"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"mel_spec": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"ONNX model saved: {onnx_out}")

    # Quick validation with onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_out), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"mel_spec": dummy.numpy()})[0]
        assert out.shape == (1, 7), f"unexpected output shape: {out.shape}"
        print(f"ONNX validation OK  output shape: {out.shape}")
    except ImportError:
        print("onnxruntime not installed — skipping validation (install to verify)")

    print("Export complete.")


def main():
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(repo / "models" / "best_model.pt"))
    ap.add_argument("--out",   default=str(repo / "models" / "custom_cnn" / "alertreck_cnn.onnx"))
    args = ap.parse_args()
    export(Path(args.model), Path(args.out))


if __name__ == "__main__":
    main()
