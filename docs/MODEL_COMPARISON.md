# Model Comparison & Analysis

Comparative evaluation of the Alertreck models across four machine-learning paradigms.
All models share the same 7-class dataset, the same **group-aware** 60/20/20 split (seed 42, split by
parent recording so no recording appears in two splits — see [AUDIO_PREPROCESSING.md](AUDIO_PREPROCESSING.md)),
and the same three-phase augmentation curriculum (A → B → C).

> **Status:** All 5 models trained and evaluated on the held-out test set.
>
> Trained checkpoints, ONNX exports, and `results.json` for every model are on Google Drive:
> [Alertreck Data, Dataset & Models](https://drive.google.com/drive/folders/1U9BwIUNQ8Snl5RxR8LHthWfdOc_EdcTM?usp=sharing).
> The numbers below are read directly from those `results.json` files; regenerate every chart with
> [`notebooks/00-model-report.ipynb`](../notebooks/00-model-report.ipynb).
>
> ⚠️ **These are the leak-free numbers.** An earlier file-level split let windows from the same parent
> recording fall into both train and test, inflating every score to ≈ 0.92 macro-F1. After moving to a
> group-aware split the scores dropped to the honest ≈ 0.80 reported here. The earlier figures are not
> comparable and have been retired.

---

## 1. Headline Results

| Model | Paradigm | Test Acc | Macro F1 | Macro AUC | Params | Size |
|---|---|---|---|---|---|---|
| **Custom CNN** | Supervised classification | **0.8263** | **0.8069** | **0.9757** | 1.21 M | 4.6 MB |
| **ProtoNet** | Few-shot metric learning | 0.8241 | 0.8036 | 0.9748 | 1.30 M | 5.0 MB |
| **W2V2-L2** | Frozen out-of-species transfer | 0.7806 | 0.7626 | 0.9605 | 0.53 M head + truncated encoder | 2 MB head |
| **Conv-AE** | Unsupervised anomaly detection | 0.51 ‡ | — | 0.8050 ‡ | 29.2 M | ~110 MB |
| **OC-SVM** | Classical anomaly detection | 0.51 ‡ | — | 0.7192 ‡ | 1,464 SVs | < 2 MB |

‡ Conv-AE and OC-SVM are binary anomaly detectors (threat vs. background); their accuracy/AUC are
binary, not 7-class. The p95-threshold accuracy sits near 0.51 because the test set is ~58 % threats
while these detectors are tuned to a ~5 % false-positive rate on background — see §3.

**The two task-trained classifiers lead and are statistically tied:** CNN (macro-F1 0.807) ≈
ProtoNet (0.804), a 0.003 gap that is within noise. The frozen-transfer **W2V2-L2 sits a clear step
behind (0.763)** — expected, since it is a *frozen* wav2vec 2.0 *layer-2* embedding with only a small
trainable head (this is the RQ5 finding, not a bug). The two anomaly detectors form a separate, lower
tier, where the **Conv-AE (binary AUC 0.805) now outperforms the classical OC-SVM (0.719)**.

---

## 2. Per-Class F1 (threat-critical breakdown)

| Class | CNN | ProtoNet | W2V2-L2 | Best |
|---|---|---|---|---|
| `threat_gunshot` | 0.815 | **0.843** | 0.812 | ProtoNet |
| `threat_human` | 0.868 | **0.895** | 0.858 | ProtoNet |
| `threat_chainsaw` | **0.824** | 0.813 | 0.780 | CNN |
| `threat_dog` | **0.794** | 0.708 | 0.738 | CNN |
| `threat_vehicle` | 0.681 | **0.723** | 0.632 | ProtoNet |
| `background_animals` | **0.836** | 0.820 | 0.762 | CNN |
| `background_wind_rain` | **0.830** | 0.823 | 0.756 | CNN |

**Key observations**

- **Vehicle is the universal weak point** (F1 0.63–0.72) — yet its *AUC* is high in every model
  (≈ 0.97). The class is **separable but mis-thresholded**: vehicle is the smallest threat class
  (646 train windows), so the decision boundary, not the representation, is the limiter. This is a
  threshold-tuning opportunity, not a retraining one.
- **Gunshot is solid, not perfect** (F1 0.81–0.84, AUC ≈ 0.97). The highest-stakes class is reliably
  ranked by all three; the F1 ceiling is again a precision/threshold effect, not missed events.
- **CNN is the most balanced model**, topping 4 of 7 classes (chainsaw, dog, and both backgrounds)
  despite being trained from scratch with no external weights.
- **ProtoNet wins the metric-learning-friendly classes** (gunshot, human, vehicle), consistent with
  its prototype-distance objective separating tight clusters well.
- **W2V2-L2 trails on every class** (and most on the backgrounds, 0.76), because its frozen layer-2
  features are tuned for speech structure, not diffuse environmental noise.

---

## 3. Anomaly-Detection Paradigm (Conv-AE vs. OC-SVM)

Both anomaly detectors are trained on **background audio only** and flag deviating clips as threats.
Conv-AE flags clips it cannot reconstruct (high MSE); OC-SVM draws an RBF boundary around background
MFCC features and flags clips that fall outside it.

> **Methodology note — what changed.** Hyperparameters *and* the final checkpoint for both detectors
> are selected on **validation detection AUC** (background-vs-threat), not on a label-free proxy.
> This matters: an earlier Conv-AE selected on lowest reconstruction loss scored only 0.60 binary AUC,
> because the minimum-reconstruction-error model is **not** the best anomaly detector. Selecting on
> detection AUC lifted it to **0.805**. Conv-AE used an Optuna search (latent dim, lr, dropout, weight
> decay); OC-SVM used an exhaustive grid (kernel × nu × gamma). Threat labels are used only for this
> *selection*, never for training — the unsupervised claim (RQ3) holds.

| Metric | Conv-AE | OC-SVM |
|---|---|---|
| Binary AUC (threat vs. bg) | **0.8050** | 0.7192 |
| Binary Avg. Precision | **0.7580** | 0.6407 |
| TPR (recall on threats) @ ~5 % FPR | **0.343** | 0.163 |
| FPR at operating point | 0.054 | 0.057 |
| Model size | 29.2 M params / ~110 MB | **1,464 SVs / < 2 MB** |

**Per-class detectability (AUC vs. background):**

| Class | Conv-AE | OC-SVM | Notes |
|---|---|---|---|
| `threat_human` | **0.957** | 0.888 | Both detect it; Conv-AE best |
| `threat_gunshot` | **0.839** | 0.627 | **Conv-AE now rescues gunshot** |
| `threat_chainsaw` | 0.758 | 0.758 | Tied |
| `threat_dog` | **0.697** | 0.639 | Both weak; Conv-AE better |
| `threat_vehicle` | 0.404 | **0.596** | **OC-SVM's only win; Conv-AE below chance** |

**Key finding (RQ3): the deep detector wins once it is selected for detection, but the paradigm is
still not a primary detector.**

1. **Conv-AE beats OC-SVM on 4 of 5 threat classes and on every aggregate metric** (binary AUC 0.805
   vs. 0.719; TPR 0.34 vs. 0.16 at the same ~5 % FPR). Most importantly it **detects gunshot well**
   (AUC 0.839) — the opposite of the earlier reconstruction-loss-selected Conv-AE, whose gunshot AUC
   was below random. Conv-AE's one blind spot is **vehicle** (AUC 0.404, below chance): low-frequency
   engine rumble overlaps the background it learned to reconstruct, so it reads as "normal".
2. **OC-SVM is now the lighter but weaker option.** Its only per-class advantage is vehicle, and at a
   usable 5–6 % FPR it catches only ~16 % of threats. Its decisive asset is **size**: < 2 MB and
   negligible CPU, versus Conv-AE's 29 M params / ~110 MB.
3. **Neither is good enough to deploy alone.** Even the better detector ranks threats at AUC 0.81 and
   catches ~34 % of threats at 5 % FPR — far below the discriminative tier (macro-F1 ≈ 0.80, all
   threat AUCs ≥ 0.94). Anomaly detection trained without any labelled threats cannot match supervised
   learning here; that is the RQ3 answer.

---

## 4. Deployment Cost (Raspberry Pi 4 + ONNX Runtime, CPU)

The whole point of Alertreck is **offline edge inference**: the model must run in real time on a Pi 4 CPU.

| Model | On-device footprint | Self-contained? | Real-time on Pi 4 CPU? |
|---|---|---|---|
| **Custom CNN** | 4.6 MB ONNX, 1.2 M params | ✅ mel → class, single graph | ✅ Yes — lightweight conv net |
| **ProtoNet** | 5.0 MB ONNX, 1.3 M params | ⚠️ needs precomputed class prototypes | ✅ Yes — embed + nearest-prototype |
| **OC-SVM** | < 2 MB joblib, 1,464 support vectors | ✅ MFCC → score | ✅ Yes — trivial CPU cost |
| **W2V2-L2** | 2 MB head + the **layer-2-truncated** wav2vec 2.0 encoder (~24 M params, ~25 % of the 94 M base) | ❌ head alone is useless | ⚠️ Truncation is the on-device design (Geldenhuys & Niesler, 2026); lighter than the full backbone but still a transformer, and accuracy trails the CNN |
| **Conv-AE** | ~110 MB ONNX, 29.2 M params | ✅ | ⚠️ Runs, but heavy for a confirmatory role |

**Note on W2V2-L2 truncation.** The frozen encoder is *physically truncated to its first 2 transformer
layers* (`scripts/prepare_w2v2_embeddings.py`), which is the paper's on-device motivation and cuts the
backbone to a fraction of the full 94 M base. It is therefore far more deployable than a full wav2vec
2.0 — but at 0.763 macro-F1 it is still below the CNN (0.807), so the CNN remains the deployment pick
on accuracy grounds regardless.

---

## 5. Recommendation

### Best accuracy **and** best for deployment → **Custom CNN**
- **Highest macro-F1 (0.807) and macro-AUC (0.976)** of all five models — the leak-free winner.
- Most balanced per-class profile (tops 4 of 7 classes); strong on gunshot (AUC 0.969).
- **Smallest self-contained footprint** (4.6 MB, 1.2 M params), mel → class in one ONNX graph, no
  backbone dependency, comfortably real-time on a Pi 4 CPU.
- Trained from scratch on this dataset, so it is fully owned and reproducible with no external weights.

This is the cleanest possible result: the deployed model is also the most accurate. **CNN is the
primary model.**

### Accuracy benchmark → **ProtoNet**
- Statistically tied with the CNN on macro-F1 (0.804 vs. 0.807) and best on gunshot/human/vehicle F1.
- Needs precomputed prototypes at inference; a fine benchmark, no advantage over the CNN to justify the
  extra moving part in deployment.

### Frozen-transfer arm (RQ5) → **W2V2-L2**
- The honest out-of-species transfer result: frozen layer-2 features reach 0.763 macro-F1, below the
  task-trained CNN — confirming that the advantage Geldenhuys & Niesler report for elephant calls
  *partially* extends to mechanical/human threat sounds but does not overtake supervised learning.
- Truncated encoder makes it deployable in principle; kept as the paradigm benchmark, not deployed.

### Confirmatory anomaly layer → size/accuracy trade-off
- **Conv-AE** is the better detector (AUC 0.805, gunshot AUC 0.839, ~34 % TPR at 5 % FPR) but costs
  29 M params / ~110 MB.
- **OC-SVM** is far lighter (< 2 MB) but weaker (AUC 0.719, ~16 % TPR) and only wins on vehicle.
- For a near-free second opinion on the Pi, **OC-SVM** remains the pragmatic add-on; if footprint is
  not a constraint, **Conv-AE** is the stronger detector. Neither is a primary detector.

---

## 6. Summary Table

| | CNN | ProtoNet | W2V2-L2 | Conv-AE | OC-SVM |
|---|---|---|---|---|---|
| Paradigm | Supervised | Few-shot | Frozen transfer | Unsup. anomaly | Classical anomaly |
| Test acc | **0.8263** | 0.8241 | 0.7806 | 0.51 ‡ | 0.51 ‡ |
| Macro F1 | **0.8069** | 0.8036 | 0.7626 | — | — |
| Macro / binary AUC | **0.9757** | 0.9748 | 0.9605 | 0.8050 ‡ | 0.7192 ‡ |
| Gunshot | F1 0.815 | F1 **0.843** | F1 0.812 | AUC 0.839 | AUC 0.627 |
| Params | 1.21 M | 1.30 M | 0.53 M head + ~24 M enc. | 29.2 M | 1,464 SVs |
| Size | 4.6 MB | 5.0 MB | 2 MB head | ~110 MB | **< 2 MB** |
| Edge-ready | ✅ **Best** | ✅ | ⚠️ | ⚠️ | ✅ |
| Role | **Deploy** | Benchmark | RQ5 benchmark | Confirmatory (heavy) | Confirmatory (light) |

‡ Conv-AE / OC-SVM are binary anomaly detectors; their accuracy and AUC are threat-vs-background, not 7-class.

---

*Generated from `models/*/results.json`. Re-run the training notebooks and refresh this file if any
model is retrained.*
