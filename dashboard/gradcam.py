"""
Grad-CAM for the Alertreck CNN
==============================
Loads the trained PyTorch CNN (`models/custom_cnn/best_model.pt`) and computes
Grad-CAM heatmaps over the mel spectrogram — highlighting the time-frequency
regions that drove a given class prediction.

Runs on the dashboard machine (the Mac), where torch already lives — the Pi stays
ONNX-only. Inputs are the exact mel spectrograms the Pi saved per detection
(`*.mel.npy`), so the explanation matches what the edge model actually classified.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Model definition (must match scripts/export_model.py / notebooks/03a) ─────

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
    """Input: (B, 1, 128, 301)  →  Output: (B, 7)"""
    def __init__(self, n_classes: int = 7):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
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


CLASS_NAMES = [
    "background_animals", "background_wind_rain", "threat_chainsaw",
    "threat_dog", "threat_gunshot", "threat_human", "threat_vehicle",
]


class GradCAM:
    """Grad-CAM on the last convolutional feature map of the encoder."""

    def __init__(self, checkpoint_path: str | Path):
        self.model = AudioCNN(n_classes=len(CLASS_NAMES))
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.eval()

        # Hook the final conv feature map in the encoder (256 channels).
        # Use the conv (block[3]), not the ReLU (block[5]): the ReLU is inplace=True
        # and a backward hook on an inplace-modified view is forbidden by autograd.
        self.target_layer = self.model.encoder[3].block[3]
        self._activations = None
        self._gradients = None
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, mel: np.ndarray, class_idx: int | None = None):
        """
        Args:
            mel: (128, 301) float32 — the exact mel the Pi classified.
            class_idx: which class to explain (default = argmax / predicted).

        Returns:
            cam   : (128, 301) heatmap normalised to [0, 1]
            pred  : predicted class index
            probs : (7,) softmax probabilities
        """
        x = torch.from_numpy(np.ascontiguousarray(mel, dtype=np.float32))
        x = x.unsqueeze(0).unsqueeze(0)          # (1, 1, 128, 301)

        logits = self.model(x)                   # (1, 7)
        probs = F.softmax(logits, dim=1)[0].detach().numpy()
        pred = int(logits.argmax(dim=1))
        target = pred if class_idx is None else int(class_idx)

        self.model.zero_grad()
        logits[0, target].backward()

        # Grad-CAM: weight each feature map by the mean of its gradients
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)      # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=mel.shape, mode="bilinear", align_corners=False)
        cam = cam[0, 0].numpy()

        cam -= cam.min()
        denom = cam.max()
        if denom > 1e-8:
            cam /= denom
        return cam, pred, probs
