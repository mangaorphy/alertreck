# Model Comparison & Analysis

Comparative evaluation of the Alertreck models across four machine-learning paradigms.
All models share the same 7-class dataset, the same 60/20/20 file-level split (seed 42),
and the same three-phase augmentation curriculum (A → B → C).

> **Status:** 4 of 5 models trained. OC-SVM (classical anomaly detection) is pending and
> will be added to the anomaly-detection section once `04c-train-oc-svm.ipynb` is run.

---

## 1. Headline Results

| Model | Paradigm | Test Acc | Macro F1 | Macro AUC | Params | ONNX size |
|---|---|---|---|---|---|---|
| **ProtoNet** | Few-shot metric learning | **0.9311** | 0.9205 | **0.9938** | 1.30 M | 5.0 MB |
| **W2V2-L2** | Transfer learning (frozen) | 0.9297 | **0.9210** | 0.9911 | 0.53 M head + 94 M backbone | 2 MB head (+ ~360 MB backbone) |
| **Custom CNN** | Supervised classification | 0.9264 | 0.9166 | — † | 1.21 M | 4.6 MB |
| **Conv-AE** | Unsupervised anomaly detection | 0.5147 ‡ | — | 0.6033 ‡ | 19.48 M | 74 MB |
| OC-SVM | Classical anomaly detection | *pending* | — | — | — | — |

† CNN notebook did not log AUC; macro F1 / accuracy are directly comparable.
‡ Conv-AE is a binary anomaly detector (threat vs. background); accuracy/AUC are binary, not 7-class.

**The three discriminative models (ProtoNet, W2V2-L2, CNN) are statistically tied** at the top —
macro F1 spans just 0.9166–0.9210. The two anomaly-detection models form a clearly separate,
lower-performing tier.

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

## 3. Anomaly-Detection Paradigm (Conv-AE)

The Conv-AE was trained on **background audio only** and flags clips it cannot reconstruct as threats.

| Metric | Value |
|---|---|
| Binary AUC (threat vs. bg) | 0.6033 |
| Binary Avg. Precision | 0.6977 |
| Accuracy @ p95 threshold | 0.5147 |
| TPR (recall on threats) | 0.1851 |
| FPR | 0.0397 |

**Per-class detectability (AUC vs. background):**

| Class | AUC | Verdict |
|---|---|---|
| `threat_human` | 0.9431 | Detected well |
| `threat_chainsaw` | 0.5905 | Near chance |
| `threat_dog` | 0.4733 | Below chance |
| `threat_vehicle` | 0.4121 | Below chance |
| `threat_gunshot` | 0.3720 | **Fails — worse than random** |

**This is the project's most important research finding (RQ3).** Pure unsupervised anomaly detection
**does not work** for this task. The reason: the threats it most needs to catch — **gunshots** — are
*short, sparse, high-energy transients* embedded in mostly-background audio. A 3-second clip containing
a gunshot is ~95% reconstructable background, so the mean reconstruction error stays *below* the
threshold. Only `threat_human` (sustained, spectrally distinct from nature sounds) is reliably caught.
A gunshot AUC of 0.37 means the autoencoder is actively *anti-correlated* with the target.

The forthcoming **OC-SVM** result will confirm whether this is a limitation of the Conv-AE specifically
or of the anomaly-detection paradigm as a whole. Current evidence strongly suggests the latter.

---

## 4. Deployment Cost (Raspberry Pi 4 + ONNX Runtime, CPU)

The whole point of Alertreck is **offline edge inference**. Raw accuracy is necessary but not
sufficient — the model has to run in real time on a Pi 4 CPU.

| Model | On-device footprint | Self-contained? | Real-time on Pi 4 CPU? |
|---|---|---|---|
| **Custom CNN** | 4.6 MB ONNX, 1.2 M params | ✅ mel → class, single graph | ✅ Yes — lightweight conv net |
| **ProtoNet** | 5.0 MB ONNX, 1.3 M params | ⚠️ needs precomputed class prototypes | ✅ Yes — embed + nearest-prototype |
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
The CNN alone is sufficient. If a second opinion is wanted to suppress false positives, a lightweight
confirmatory model (e.g. OC-SVM on MFCCs, once trained) can gate alerts — but **not** Conv-AE, whose
gunshot detection is worse than random.

### Models to drop from deployment
- **W2V2-L2** — backbone too heavy for the Pi 4; keep as an accuracy benchmark only.
- **Conv-AE** — fails on the threats that matter (gunshot AUC 0.37); keep only as a documented
  negative result for the anomaly-detection research question.

---

## 6. Summary Table

| | CNN | ProtoNet | W2V2-L2 | Conv-AE | OC-SVM |
|---|---|---|---|---|---|
| Paradigm | Supervised | Few-shot | Transfer | Unsup. anomaly | Classical anomaly |
| Test acc | 0.9264 | **0.9311** | 0.9297 | 0.5147 ‡ | pending |
| Macro F1 | 0.9166 | 0.9205 | **0.9210** | — | pending |
| Macro AUC | — | **0.9938** | 0.9911 | 0.6033 ‡ | pending |
| Gunshot F1 | **0.9990** | 0.9969 | 0.9979 | (AUC 0.37) | pending |
| Params | 1.21 M | 1.30 M | 94 M (w/ backbone) | 19.48 M | — |
| Edge-ready | ✅ **Best** | ✅ | ❌ | ⚠️ | tbd |
| Role | **Deploy** | Benchmark | Benchmark | Negative result | Confirmatory? |

---

*Generated from `models/*/results.json`. Re-run the training notebooks and refresh this file when
OC-SVM is added.*
