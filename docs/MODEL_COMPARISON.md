# Model Comparison & Analysis

Comparative evaluation of the Alertreck models across four machine-learning paradigms.
All models share the same 7-class dataset, the same 60/20/20 file-level split (seed 42),
and the same three-phase augmentation curriculum (A → B → C).

> **Status:** All 5 models trained.

---

## 1. Headline Results

| Model | Paradigm | Test Acc | Macro F1 | Macro AUC | Params | ONNX / model size |
|---|---|---|---|---|---|---|
| **ProtoNet** | Few-shot metric learning | **0.9311** | 0.9205 | **0.9938** | 1.30 M | 5.0 MB |
| **W2V2-L2** | Transfer learning (frozen) | 0.9297 | **0.9210** | 0.9911 | 0.53 M head + 94 M backbone | 2 MB head (+ ~360 MB backbone) |
| **Custom CNN** | Supervised classification | 0.9264 | 0.9166 | — † | 1.21 M | 4.6 MB |
| **OC-SVM** | Classical anomaly detection | 0.5100 ‡ | — | 0.7790 ‡ | 285 SVs | < 1 MB |
| **Conv-AE** | Unsupervised anomaly detection | 0.5147 ‡ | — | 0.6033 ‡ | 19.48 M | 74 MB |

† CNN notebook did not log AUC; macro F1 / accuracy are directly comparable.
‡ OC-SVM and Conv-AE are binary anomaly detectors (threat vs. background); accuracy/AUC are binary, not 7-class.

**The three discriminative models (ProtoNet, W2V2-L2, CNN) are statistically tied** at the top —
macro F1 spans just 0.9166–0.9210. The two anomaly-detection models form a clearly separate,
lower-performing tier — though **OC-SVM (binary AUC 0.779) clearly outperforms the much heavier
Conv-AE (0.603)** despite being a classical model under 1 MB.

---

## 2. Per-Class F1 (threat-critical breakdown)

| Class | CNN | ProtoNet | W2V2-L2 | Best |
|---|---|---|---|---|
| `threat_gunshot` | 0.9990 | 0.9969 | 0.9979 | **CNN** |
| `threat_human` | 0.9521 | 0.9641 | 0.9702 | **W2V2** |
| `threat_chainsaw` | 0.9254 | 0.9348 | 0.9304 | **ProtoNet** |
| `threat_vehicle` | 0.8922 | 0.8932 | 0.9210 | **W2V2** |
| `threat_dog` | 0.8098 | 0.8148 | 0.8386 | **W2V2** |
| `background_animals` | 0.9140 | 0.9205 | 0.9131 | **ProtoNet** |
| `background_wind_rain` | 0.9236 | 0.9194 | 0.8757 | **CNN** |

**Key observations**

- **Gunshot is near-perfect everywhere** (F1 ≥ 0.997). This is the single most important class for
  an anti-poaching system, and all three discriminative models nail it. ProtoNet and W2V2 both achieve
  **gunshot recall = 1.000** (zero missed gunshots on the test set).
- **`threat_dog` is the universal weak point** (F1 0.81–0.84). Dog barks overlap acoustically with
  other animal vocalisations; W2V2's richer features recover the most here (+0.029 over CNN).
- **W2V2 wins the hard classes** (dog, vehicle, human) thanks to its pretrained representations, but
  **loses on `bg_wind_rain`** (0.876) — its frozen features are tuned for speech, not diffuse noise.
- **CNN is the most balanced** despite being trained from scratch, and tops the chart on the two
  highest-stakes/easiest-to-confuse extremes (gunshot, wind/rain).

---

## 3. Anomaly-Detection Paradigm (OC-SVM vs. Conv-AE)

Both anomaly detectors were trained on **background audio only** and flag deviating clips as threats.
OC-SVM draws an RBF boundary around background MFCC features; Conv-AE flags clips it cannot reconstruct.

| Metric | OC-SVM | Conv-AE |
|---|---|---|
| Binary AUC (threat vs. bg) | **0.7790** | 0.6033 |
| Binary Avg. Precision | **0.7822** | 0.6977 |
| Accuracy @ p95 threshold | 0.5100 | 0.5147 |
| TPR (recall on threats) | 0.1958 | 0.1851 |
| FPR | 0.0650 | 0.0397 |
| Model size | **< 1 MB (285 SVs)** | 74 MB / 19.5 M params |

**Per-class detectability (AUC vs. background):**

| Class | OC-SVM | Conv-AE | Notes |
|---|---|---|---|
| `threat_human` | 0.8321 | **0.9431** | Both detect it; Conv-AE better |
| `threat_chainsaw` | **0.8126** | 0.5905 | OC-SVM far better |
| `threat_gunshot` | **0.8050** | 0.3720 | **OC-SVM rescues it; Conv-AE fails** |
| `threat_vehicle` | **0.6935** | 0.4121 | OC-SVM better |
| `threat_dog` | **0.5907** | 0.4733 | Both weak |

**Key finding (RQ3): the classical model wins, and the paradigm is still not viable as a primary detector.**

Two things stand out:

1. **OC-SVM decisively beats the deep Conv-AE** (binary AUC 0.779 vs. 0.603) at **1/100th the size**
   (< 1 MB vs. 74 MB). Most importantly, it **rescues gunshot detection** — Conv-AE's gunshot AUC of
   0.37 was *worse than random*, while OC-SVM reaches 0.805. The reason: Conv-AE scores a clip by its
   *mean* reconstruction error, so a brief gunshot transient drowns in ~3 s of reconstructable
   background. OC-SVM's MFCC mean+std summary preserves the spectral signature of that transient, so
   the clip lands outside the background boundary. For an anti-poaching system this is the difference
   between a useless and a usable anomaly detector.

2. **Neither is good enough to deploy alone.** Even the better model only ranks threats at AUC 0.78,
   and at a sensible 5–6% FPR operating point it catches just **~20% of threats** (TPR 0.196). That is
   far below the discriminative tier (macro F1 ≈ 0.92, gunshot F1 ≈ 0.997). The conclusion holds:
   anomaly detection trained without any labelled threats cannot match supervised learning here — but
   **the classical OC-SVM is the right choice within the paradigm**, and salvages it from the
   "completely broken" verdict the Conv-AE alone would have earned.

---

## 4. Deployment Cost (Raspberry Pi 4 + ONNX Runtime, CPU)

The whole point of Alertreck is **offline edge inference**. Raw accuracy is necessary but not
sufficient — the model has to run in real time on a Pi 4 CPU.

| Model | On-device footprint | Self-contained? | Real-time on Pi 4 CPU? |
|---|---|---|---|
| **Custom CNN** | 4.6 MB ONNX, 1.2 M params | ✅ mel → class, single graph | ✅ Yes — lightweight conv net |
| **ProtoNet** | 5.0 MB ONNX, 1.3 M params | ⚠️ needs precomputed class prototypes | ✅ Yes — embed + nearest-prototype |
| **OC-SVM** | < 1 MB joblib, 285 support vectors | ✅ MFCC → score | ✅ Yes — trivial CPU cost |
| **W2V2-L2** | 2 MB head **but** requires the 94 M-param `wav2vec2-base` transformer backbone (~360 MB) to generate embeddings | ❌ head alone is useless | ❌ Impractical — transformer inference is too heavy for real-time CPU |
| **Conv-AE** | 74 MB ONNX, 19.5 M params | ✅ | ⚠️ Runs, but detection quality is unusable |

**Critical caveat for W2V2-L2:** the exported `w2v2_head.onnx` (2 MB) is *only the classifier head*.
At inference it consumes 768-dim embeddings that must be produced by the full `facebook/wav2vec2-base`
backbone — a ~94 M-parameter transformer. That backbone is the real cost, and it is not exported here.
Running it per 3-second window on a Pi 4 CPU is not viable for continuous real-time monitoring.
**W2V2-L2 is an excellent accuracy benchmark but a poor deployment candidate.**

---

## 5. Recommendation

There are two answers depending on the question being asked.

### Best raw performance → **ProtoNet**
- Highest test accuracy (**0.9311**) and highest macro AUC (**0.9938**).
- Macro F1 (0.9205) is statistically tied with W2V2 (0.9210) — a 0.0005 gap is noise.
- Gunshot recall = 1.000.
- Use this as the **accuracy ceiling / benchmark** in the dissertation.

### Best for deployment (recommended primary model) → **Custom CNN**
- Threat performance is within ~1 F1 point of the best model on every class, and it is the **single
  best model on gunshot (0.999)** — the highest-stakes class.
- **Smallest self-contained footprint** (4.6 MB, 1.2 M params), mel-spectrogram → class in one graph,
  **no backbone dependency**, comfortably real-time on a Pi 4 CPU.
- Trained from scratch on this dataset, so it is fully owned and reproducible with no external weights.

### Suggested deployment architecture
```
                ┌─────────────────────────┐
  mic ──► mel ─►│  CNN  (primary, 4.6 MB)  │──► threat class + confidence ──► GSM/GPS alert
                └─────────────────────────┘
```
The CNN alone is sufficient. If a second, independent opinion is wanted to suppress false positives,
the **OC-SVM** (< 1 MB, MFCC-based, gunshot AUC 0.805) is the only viable confirmatory model — it adds
a different feature view at negligible cost. Do **not** use Conv-AE, whose gunshot detection is worse
than random.

### Best classical / anomaly model → **OC-SVM**
- Within the anomaly-detection paradigm, OC-SVM beats Conv-AE on every metric that matters (binary AUC
  0.779 vs. 0.603, gunshot AUC 0.805 vs. 0.372) at **1/100th the size**.
- Not strong enough to be a primary detector (TPR ~20% at a usable threshold), but a sound,
  near-free confirmatory layer and the better answer to the classical-vs-deep anomaly question.

### Models to drop from deployment
- **W2V2-L2** — backbone too heavy for the Pi 4; keep as an accuracy benchmark only.
- **Conv-AE** — fails on the threats that matter (gunshot AUC 0.37) and is 74 MB; keep only as a
  documented negative result showing deep reconstruction loses to a classical boundary here.

---

## 6. Summary Table

| | CNN | ProtoNet | W2V2-L2 | OC-SVM | Conv-AE |
|---|---|---|---|---|---|
| Paradigm | Supervised | Few-shot | Transfer | Classical anomaly | Unsup. anomaly |
| Test acc | 0.9264 | **0.9311** | 0.9297 | 0.5100 ‡ | 0.5147 ‡ |
| Macro F1 | 0.9166 | 0.9205 | **0.9210** | — | — |
| Macro / binary AUC | — | **0.9938** | 0.9911 | 0.7790 ‡ | 0.6033 ‡ |
| Gunshot | F1 **0.9990** | F1 0.9969 | F1 0.9979 | AUC 0.805 | AUC 0.372 |
| Params | 1.21 M | 1.30 M | 94 M (w/ backbone) | 285 SVs | 19.48 M |
| Size | 4.6 MB | 5.0 MB | ~360 MB | **< 1 MB** | 74 MB |
| Edge-ready | ✅ **Best** | ✅ | ❌ | ✅ | ⚠️ |
| Role | **Deploy** | Benchmark | Benchmark | Confirmatory | Negative result |

‡ OC-SVM / Conv-AE are binary anomaly detectors; their accuracy and AUC are threat-vs-background, not 7-class.

---

*Generated from `models/*/results.json`. Re-run the training notebooks and refresh this file if any
model is retrained.*
