# Alertreck — Project Roadmap

**Student:** Orpheus Mhizha Manga · **Program:** ALU BSc SE (Machine Learning), 2026 Capstone
**Supervisor:** Hubert Apana · **Contact:** o.manga@alustudent.com
**Project:** Offline edge-AI acoustic poaching-threat detection for remote African conservation parks
**Full title:** *Offline Edge AI for Anti-Poaching Acoustic Surveillance: Benchmarking Pretrained Embeddings, Metric Learning, Supervised Classification, and Anomaly Detection on Identical Field Hardware*

> **Research questions (from proposal §1.4):**
> - RQ1: Which learning paradigm — supervised classification (CNN), metric learning (ProtoNet), frozen out-of-species transfer (W2V2-L2), or unsupervised anomaly detection (Conv-AE / OC-SVM) — achieves highest AUC-ROC, per-class F1, and lowest FPR on a 7-class African savannah acoustic dataset on identical edge hardware?
> - RQ2: How does per-class detection performance differ across the five models at SNR 5 / 10 / 20 dB, and which sound class poses the greatest detection challenge under field-representative interference?
> - RQ3: Can the SIM808 GSM/GPS module acquire a GPS fix and append device coordinates to a labelled SMS alert within the 30 s end-to-end latency budget on Pi 4 hardware, and what is the typical fix-acquisition time outdoors?
> - RQ4: Do Grad-CAM heatmaps on the winning model's mel-spectrogram inputs highlight acoustically expected time-frequency regions, supporting ranger trust?
> - RQ5: Does the out-of-species embedding advantage (Geldenhuys & Niesler, 2026) — a frozen, truncated wav2vec 2.0 layer-2 embedding approaching supervised performance for elephant calls — extend to non-biological threats (gunshots, chainsaws, vehicles) in a multi-class African savannah dataset, and does it retain that advantage on edge hardware after INT8 quantisation?

---

## 1. Locked design parameters

These are fixed by the approved proposal (May 2026). Deviations require supervisor approval.

### 1.1 Seven-class acoustic threat taxonomy

| Label | Kind | Description |
|---|---|---|
| `background_animals` | background | Non-threat wildlife: birds, elephants, lions, horses, insects, frogs |
| `background_wind_rain` | background | Weather / natural ambience: rain, wind, thunderstorm, sea waves, fire |
| `threat_gunshot` | threat | Firearm discharge (rifle, shotgun) |
| `threat_chainsaw` | threat | Chainsaw running / cutting |
| `threat_vehicle` | threat | Truck / motorbike / ATV engine in park |
| `threat_human` | threat | Human voice, shouting, footsteps, machete |
| `threat_dog` | threat | Barking / howling — poacher-dog indicator |

> **Note:** `normal` class was split into `background_animals` and `background_wind_rain` (2026-04-19) for cleaner source tracking and balanced augmentation control.

### 1.2 Audio specification

| Parameter | Value | Justification |
|---|---|---|
| Sample rate | 44.1 kHz mono | Captures full chainsaw / vehicle harmonic content up to Nyquist 22.05 kHz; matches USB microphone native rate |
| Clip length | 3 s, 50 % overlap | Ensures full gunshot transient captured regardless of onset position |
| Amplitude | RMS-normalised to −23 dBFS (EBU R128) | Removes gain variation across dataset sources |
| Mel features | 128-bin log-mel, 25 ms window, 10 ms hop, Hann | Input for CNN, ProtoNet, Conv-AE |
| wav2vec input | 3 s mono waveform resampled to 16 kHz | Native input for the frozen wav2vec 2.0 layer-2 embedding (W2V2-L2 arm); matches the model's pretraining sample rate |
| MFCC features | 40-coeff + Δ + ΔΔ → 120-dim vector per frame | Input for OC-SVM; RBF kernel tractable at this dimensionality |

### 1.3 Five-model comparative architecture (four paradigms)

| Model | Paradigm | Input | Loss | Key detail |
|---|---|---|---|---|
| **CNN** | Supervised | 128-bin log-mel + SpecAugment | Focal loss γ=2 | 4 conv blocks + FC; ~2.1 M params; encoder reused by ProtoNet |
| **ProtoNet (CNN enc)** | Metric learning | 128-bin log-mel + SpecAugment | Prototypical loss + SupCon (λ=0.1) | N=7, K=5 support, Q=15 query per episode; SupCon pulls same-class embeddings across episodes; t-SNE visualisation |
| **W2V2-L2 + linear head** | Frozen transfer (out-of-species) | 16 kHz raw waveform → frozen wav2vec 2.0 layer-2 embedding (768-dim mean-pooled) | Cross-entropy + focal weighting | wav2vec 2.0 base truncated at layer 2 (~10 % of params); encoder frozen, only a 768→7 linear head trained; no fine-tuning (Geldenhuys & Niesler, 2026) |
| **Conv-AE** | Unsupervised anomaly | 128-bin log-mel + compound aug | MSE reconstruction | 3-layer encoder → 16×16×128 latent; symmetric decoder |
| **OC-SVM** | Unsupervised (classical baseline) | 120-dim MFCC+Δ+ΔΔ + compound aug | One-class hinge | RBF kernel; ν=0.1; grid search for γ; provides the interpretability / compute floor |

> **Note:** tiny-AST (a supervised, fine-tuned audio transformer) was dropped on 2026-06-01 and replaced by **W2V2-L2**. Fine-tuning a transformer on a small corpus overfits under domain shift; a frozen, truncated wav2vec 2.0 layer-2 embedding introduces a genuinely distinct fourth paradigm (frozen out-of-species transfer) rather than a second supervised architecture, and is the empirical vehicle for RQ5 (Geldenhuys & Niesler, 2026).

### 1.4 Dataset split

- **60 / 20 / 20** train / val / test, stratified by class, fixed seed `42`
- Test set is locked before any augmentation is applied to training folds
- Validation and test sets receive **only base preprocessing** (steps 1–5), not augmentation steps 6a–6e

### 1.5 Hardware target

| Component | Spec | Role | Status |
|---|---|---|---|
| Raspberry Pi 4 Model B | 2 GB RAM, quad-core ARM Cortex-A72 @ 1.8 GHz | Edge processing, inference, alert transmission, local event logging | ● in hand · hostname `alertreck.local` (192.168.1.88) |
| USB microphone | Plug-and-play USB audio, 100 Hz–16 kHz frequency response | Continuous outdoor audio capture at 44.1 kHz mono | ● in hand · USB hum being resolved (gain + powered hub) |
| SIM808 GSM/GPRS/GPS module | Quad-band 850/900/1800/1900 MHz, UART serial, built-in GPS receiver | SMS alert delivery (MTN/Airtel Rwanda, no data plan) + device GPS coordinates | ● in hand |
| 32 GB Class 10 MicroSD card | Class 10, read ≥ 45 MB/s | OS, model weights, SQLite event database, evidence audio | ● in hand · Pi OS installed |
| Rwanda SIM card (MTN or Airtel) | Prepaid, GSM only (no data plan required) | Cellular network access for SMS delivery | ● in hand |

**Operational budgets:** ≤ USD 80 BOM · ≤ 5 W continuous · < 30 s end-to-end latency · TFLite INT8 · model < 10 MB

### 1.6 Experimental controls (ensures fair comparison across all 5 models)

- Identical preprocessed dataset at 44.1 kHz with 3 s windows, 50 % overlap
- Identical 60/20/20 stratified split, fixed seed 42
- Identical evaluation hardware (Raspberry Pi 4 Model B, 2 GB RAM)
- Identical nine evaluation techniques applied to all five models across all four paradigms
- Identical SNR injection conditions (5, 10, and 20 dB) for anomaly injection tests
- Experiment tracking: local **MLflow** (offline-first — consistent with no-cloud design constraint)

---

## 2. Dataset sources and provenance

### 2.1 Corpus status (as of 2026-05-27)

**Total on disk: 8,907 raw clips · Stage 1 complete ●**

| Class | Folder | Target | Current | Status | Sources |
|---|---|---|---|---|---|
| `background_animals` | `dataset/background_animals/` | ≥ 600 | 2,140 | **met ✓** | ESC-50 + DATASET02 animals & birds + ff1010bird |
| `background_wind_rain` | `dataset/background_wind_rain/` | ≥ 600 | 680 | **met ✓** | ESC-50 + DATASET02 rainfall & wind |
| `threat_gunshot` | `dataset/threat_gunshot/` | ≥ 300 | 2,400 | **met ✓** | AudioSet extraction |
| `threat_chainsaw` | `dataset/threat_chainsaw/` | ≥ 500 | 567 | **met ✓** | ESC-50 + AudioSet (complete) |
| `threat_vehicle` | `dataset/threat_vehicle/` | ≥ 200 | 1,040 | **met ✓** | ESC-50 + UrbanSound8K |
| `threat_human` | `dataset/threat_human/` | ≥ 200 | 1,040 | **met ✓** | Common Voice + ESC-50 |
| `threat_dog` | `dataset/threat_dog/` | ≥ 200 | 1,040 | **met ✓** | UrbanSound8K + ESC-50 |

All class targets met. Dataset is locked for training.

### 2.2 Dataset sources — detail

#### AudioSet (Google, CC BY 4.0)
CSV metadata vendored under [META_DATA_CSV/](META_DATA_CSV/). Audio queried via YouTube segments using `yt-dlp`.
> Gemmeke et al. (2017). *Audio Set: An ontology and human-labeled dataset for audio events.* IEEE ICASSP.

Scripts: [scripts/audio_extraction/](scripts/audio_extraction/) — `download_chainsaws_500.py`, `download_human_sounds.py`, `download_animal_sounds.py`, `download_env_noise.py`

#### ESC-50 (CC BY-NC 3.0)
880 clips across all 7 classes. 5 s clips at 44.1 kHz mono.
> Piczak (2015). *ESC: Dataset for Environmental Sound Classification.* ACM Multimedia 2015.

Script: [scripts/audio_extraction/extract_esc50.py](scripts/audio_extraction/extract_esc50.py)

#### Freefield1010 / ff1010bird (Mixed CC)
500 randomly sampled bird-positive clips (seed 42) → `background_animals`.
> Stowell & Plumbley (2013). *Detection of Bird Species in Audio.* British Library Audio Collection.

Script: [scripts/audio_extraction/extract_ff1010_birds.py](scripts/audio_extraction/extract_ff1010_birds.py)

#### DATASET02 (local, academic use)
1,800 clips covering elephant, lion, horse, birds, rainfall, wind.
Script: [scripts/audio_extraction/extract_dataset02.py](scripts/audio_extraction/extract_dataset02.py)

#### Mozilla Common Voice (CC0)
~500 English speech clips → `threat_human`.
> Ardila et al. (2020). *Common Voice.* LREC 2020.

#### UrbanSound8K (CC BY-NC)
2,000 clips: `dog_bark` → `threat_dog`; `engine_idling` → `threat_vehicle`.
> Salamon et al. (2014). *A Dataset and Taxonomy for Urban Sound Research.* ACM Multimedia 2014.

Merged via [scripts/audio_extraction/merge_extracted_audio.py](scripts/audio_extraction/merge_extracted_audio.py).

### 2.3 Extraction scripts

| Script | Source | Classes fed |
|---|---|---|
| [scripts/audio_extraction/extract_esc50.py](scripts/audio_extraction/extract_esc50.py) | ESC-50 | threat_chainsaw, threat_vehicle, threat_dog, threat_human, background_animals, background_wind_rain |
| [scripts/audio_extraction/extract_dataset02.py](scripts/audio_extraction/extract_dataset02.py) | DATASET02 | background_animals, background_wind_rain |
| [scripts/audio_extraction/extract_ff1010_birds.py](scripts/audio_extraction/extract_ff1010_birds.py) | ff1010bird | background_animals |
| [scripts/audio_extraction/download_chainsaws_500.py](scripts/audio_extraction/download_chainsaws_500.py) | AudioSet | threat_chainsaw |
| [scripts/audio_extraction/merge_extracted_audio.py](scripts/audio_extraction/merge_extracted_audio.py) | Common Voice + UrbanSound8K | threat_human, threat_dog, threat_vehicle |
| [scripts/data_manifest.py](scripts/data_manifest.py) | all | generates manifest.csv |

---

## 3. Compound augmentation and curriculum training pipeline

> **Source:** Proposal §2.3.7, §3.3, §3.3.1 — directly informed by Mega-ASR (Xie et al., 2026), SpecAugment++ (Wang et al., 2021), FilterAugment (Nam et al., 2022), and Morocutti et al. (2023).

This is the most critical section for model robustness. All prior conservation acoustic classifiers train on atomic single-condition data and fail on the compound degradation profiles of real African field deployment. This pipeline closes that gap.

### 3.1 Full preprocessing pipeline (all 7 steps)

Applied to **all clips** in order. Steps 6a–6e apply to **training folds only**.

| Step | Operation | Parameters | Justification |
|---|---|---|---|
| 1 | **Resample** | Kaiser-best resampling → 44.1 kHz mono | Full chainsaw/vehicle harmonic content up to 22.05 kHz; matches USB microphone native rate |
| 2 | **Normalise** | RMS amplitude → −23 dBFS (EBU R128) | Removes gain variation across sources; prevents loudness shortcuts |
| 3 | **Segment** | Sliding window: 3 s, step 1.5 s (50 % overlap) | Ensures full gunshot transient captured regardless of onset position |
| 4a | **Log-mel spectrogram** | STFT → 128-bin mel filterbank → log; 25 ms window, 10 ms hop, Hann | Input for CNN, ProtoNet, Conv-AE |
| 4b | **Raw waveform (16 kHz)** | 3 s mono waveform resampled 44.1 → 16 kHz | Native input for the frozen wav2vec 2.0 layer-2 encoder (W2V2-L2 arm); matches wav2vec 2.0 pretraining sample rate (Geldenhuys & Niesler, 2026) |
| 4c | **MFCC+Δ+ΔΔ** | DCT → delta → delta-delta; 40 coefficients → 120-dim vector per frame | Input for OC-SVM; RBF kernel tractable at this dimensionality |
| 5 | **USB mic DIR calibration** | Sweep-tone impulse response recorded via deployment USB microphone; convolved with all training clips | Closes device domain gap between downloaded training data and deployment microphone frequency response (Morocutti et al., 2023) |
| 6a | **SpecAugment** | Time masking: 2 masks, max 40 frames; frequency masking: 2 masks, max 20 bins | Forces model to learn full spectral shape of threats, not just onset timing (Wang et al., 2021) |
| 6b | **Compound augmentation** | Randomly sample 2–4 effects from pool below; physical plausibility filter applied | Simulates real compound field conditions per Mega-ASR (Xie et al., 2026) |
| 6c | **FilterAugment** | Random frequency response curve per clip; magnitude: ±6 dB across random sub-band | Simulates USB microphone frequency response variation across outdoor temperature/humidity (Nam et al., 2022) |
| 6d | **Mixup** | Beta distribution β(0.4); inter-class and intra-class pairs | Additional regularisation; improves class boundary generalisation (Zhang et al., 2018) |
| 6e | **Learnability filter** | Discard augmented clips where Conv-AE reconstruction error exceeds 95th percentile of background distribution | Removes unlearnable hard-negative samples that destabilise training (Mega-ASR WER > 70 % filter principle) |
| 7 | **Split** | Stratified 60/20/20; seed 42; test set locked before augmentation applied to training folds | Held-out test set reflects clean baseline; augmentation only in training folds |

### 3.2 Compound augmentation effect pool (step 6b)

Between **2 and 4 effects** are drawn simultaneously per clip. Physical plausibility filter: RIR convolution is **not** combined with dry clipping (acoustically contradictory).

| Effect | Parameters | What it simulates |
|---|---|---|
| Gaussian noise | 5–20 dB SNR | Background ambient noise at varying distances |
| RIR convolution | Recorded room impulse responses | Reverberation in dense vegetation, enclosed spaces |
| Low-pass filter | 2–8 kHz cutoff | Distance attenuation — high frequencies attenuate first |
| MP3 compression | VBR quality 2–7 | Recording device codec degradation |
| Gain perturbation | ±6 dB | Microphone gain variation, wind gust masking |
| Clipping distortion | ≤ 10th percentile amplitude threshold | Electronic overload / cheap hardware saturation |

### 3.3 Curriculum training schedule (three phases)

Following Mega-ASR's progressive difficulty strategy (Xie et al., 2026). Models trained on uniform mixtures of all difficulty levels simultaneously collapse on severe compound inputs — the curriculum prevents this.

| Phase | Training weeks | Augmentation applied | SNR range | Rationale |
|---|---|---|---|---|
| **Phase A** | W3–W4 | Clean + mild single-condition only (steps 6a, 6d) | ≥ 15 dB | Builds reliable acoustic feature extraction on clean spectral structure; stable encoder representations |
| **Phase B** | W5 | Medium compound: 2–3 effects at moderate severity (steps 6a–6d, severity k ≤ 0.5) | 10–15 dB | Introduces realistic field degradation; prevents overfitting to clean training distribution |
| **Phase C** | W6 | Full compound pipeline: 2–4 effects at full severity including far-field and device distortion (steps 6a–6e) | 5–10 dB | Exposes models to hardest compound conditions; learnability filter (step 6e) applied to prevent collapse on unrecoverable samples |

**Applies to:** CNN, W2V2-L2, ProtoNet training pipelines (identical schedule). For W2V2-L2 the wav2vec 2.0 layer-2 encoder is held **fixed in all phases** — only the lightweight linear head receives gradient updates across the curriculum (Geldenhuys & Niesler, 2026).

**For Conv-AE and OC-SVM** (trained on background-class audio only):
- Phase A: clean background clips only
- Phase B: medium compound-degraded background clips
- Phase C: full compound augmentation on background audio

This ensures the anomaly detectors build a robust model of what normal background sounds like under compound field conditions, not just under clean recording conditions — the primary cause of elevated false-positive rates in field-deployed anomaly detectors.

### 3.4 ProtoNet-specific: SupCon loss term

For the ProtoNet metric learning arm, a supervised contrastive loss (SupCon) is added alongside the standard prototypical loss:

- **Loss:** `L = L_proto + 0.1 × L_SupCon`
- **Effect:** Pulls all embeddings of the same class together across episodes (not just within one episode), improving robustness to intra-class acoustic variation
- **Especially relevant for:** `threat_human` and `threat_vehicle` classes, which exhibit high intra-class spectral diversity

Reference: Gazneli et al. (2022).

---

## 4. Nine-technique evaluation framework

> **Source:** Proposal §3.5.2 — applied identically to all five models.

| # | Technique | Target | What it measures |
|---|---|---|---|
| 1 | **AUC-ROC curve** | > 0.85 (all five models) | Primary headline metric; detection performance independent of threshold |
| 2 | **Per-class F1 score** | > 0.80 (all 7 classes, supervised + metric models) | Balance between alert sensitivity and FPR per threat category |
| 3 | **Threshold ablation** | — | Sweeps detection threshold across full range; identifies optimal operating point for field conditions |
| 4 | **False-positive rate test** | < 20 % | Evaluates on held-out normal audio (rain, wind, insects) not seen during training |
| 5 | **Anomaly injection test** | — | Recall evaluation: mixes labelled gunshot, human voice, vehicle into normal audio at SNR 5 / 10 / 20 dB |
| 6 | **Latency benchmark (Pi 4)** | < 30 s end-to-end | Measures inference time from audio onset to SMS delivery on deployment hardware |
| 7 | **Energy benchmark** | ≤ 5 W continuous | Measures continuous power draw on Pi 4 during active monitoring |
| 8 | **GPS fix quality** | Valid fix ≤ 60 s cold start; coordinates within 5 m CEP | Validates SIM808 GPS time-to-first-fix and coordinate accuracy under open-sky and light-canopy conditions |
| 9 | **Grad-CAM validation** | Qualitative | Confirms highlighted spectrogram regions match acoustically expected features of each class |

---

## 5. Implementation stages — design to deployment

Ten stages across eight proposal phases. Each stage maps to a proposal section and produces a concrete artefact.

---

### Stage 0 — Design lock-in ● done

**Goal:** Freeze all design parameters in version-controlled docs.
**Proposal ref:** §1.3–§1.5
**Outputs:** `ROADMAP.md` · `DESIGN.md` · `PROPOSAL.md`
**Status:** ● complete

---

### Stage 1 — Data engineering ● done

**Goal:** Assemble, organise, and document 8,907 raw audio clips across all 7 classes.
**Proposal ref:** §3.3 (Table 2)
**Tasks:**
1. ● `threat_chainsaw` at 567 clips — above 500 target (completed 2026-05-27)
2. ● Data manifest generated — 8,907 files across 7 classes
3. ● Classes mapped to 7-class taxonomy
4. ● ff1010bird merged into `background_animals/` with `ff1010bird__` prefix

**Artefact:** `dataset/` directory with 8,907 labelled clips across 7 class folders
**Status:** ● complete

---

### Stage 2 — Deterministic preprocessing pipeline ● done

**Goal:** One deterministic `AudioPreprocessor` implementing all 7 pipeline steps from §3.1 above.
**Proposal ref:** §3.3 (Table 3), §2.3.7
**Implementation:** Python 3.10 · librosa 0.10+ · numpy (no scipy, no audiomentations — all effects from scratch)

**Tasks:**
1. ● Rewrote [scripts/audio_preprocessing.py](scripts/audio_preprocessing.py) — full 7-step pipeline:
   - Step 1: Kaiser-best resample → 44.1 kHz mono
   - Step 2: RMS normalise → −23 dBFS (EBU R128)
   - Step 3: Sliding window, 3 s, 1.5 s step → 132,300 samples per clip
   - Step 4a: 128-bin log-mel (STFT win=1102/25 ms, hop=441/10 ms, Hann, n_fft=2048) → (128, 300) shape
   - Step 4b: 16 kHz raw-waveform branch for the W2V2-L2 frozen encoder — **new requirement from the updated proposal (2026-06-01); not yet emitted, pipeline extension + re-run pending**
   - Step 4c: MFCC+Δ+ΔΔ (40 coefficients → 120-dim per frame)
   - Step 5: DIR calibration stub (active with `--dir-ir usb_mic_ir.wav`)
   - Steps 6a–6e: SpecAugment, compound aug pool, FilterAugment, mixup utility, learnability filter stub
   - Step 7: 60/20/20 stratified split, seed 42; test set locked
2. ◐ USB mic DIR calibration (Step 5) — hardware in place at alertreck.local; hum issue being resolved before final re-run with `--dir-ir`
3. ● All augmentation steps 6a–6d implemented from scratch (numpy + librosa + ffmpeg); 6e stub until Conv-AE trained
4. ● `.npz` shards written to `data/processed/{mel,mfcc}/{train,val,test,train_aug_A,train_aug_B,train_aug_C}/`
5. ● `data/processed/manifest.json` and `splits.json` written

**Preprocessed output (2026-05-27):**

| Split | Files | Windows |
|---|---|---|
| `mel/train` (clean) | 5,343 | 7,359 |
| `mel/train_aug_A` (Phase A ×1) | 5,343 | 7,359 |
| `mel/train_aug_B` (Phase B ×2) | 5,343 | 14,718 |
| `mel/train_aug_C` (Phase C ×3) | 5,343 | 22,077 |
| `mel/val` | 1,782 | 2,394 |
| `mel/test` | 1,782 | 2,471 |
| `mfcc/{same splits}` | — | same counts |

**Artefact:** [scripts/audio_preprocessing.py](scripts/audio_preprocessing.py) · `data/processed/` shards · `manifest.json` · `splits.json`
**Status:** ◐ mel/mfcc shards complete; **16 kHz raw-waveform branch (step 4b) for W2V2-L2 still to be added and the pipeline re-run** (new requirement, updated proposal). DIR calibration also pending hardware fix — neither blocks CNN/ProtoNet training.

---

### Stage 3 — Supervised + metric model training (CNN · ProtoNet) ◐

**Goal:** Train and checkpoint the two CNN-encoder models (supervised + metric paradigms) under the three-phase curriculum.
**Proposal ref:** §3.5.1 (Table 4), §3.3.1
**Proposal weeks:** W3–W5
**Notebooks:** `notebooks/03a-train-cnn.ipynb` · `notebooks/03b-train-protonet.ipynb`
**Experiment tracking:** local MLflow (offline)

#### 3a. CNN (supervised, focal loss γ=2) ◐ notebook ready — awaiting Kaggle GPU run
- Input: 128-bin log-mel (128 × 300) + SpecAugment
- Architecture: 4 conv blocks + AdaptiveAvgPool + FC head; ~1.2 M parameters
- Encoder exposes `encode()` method returning 256-dim embedding — reused by ProtoNet (Stage 3b)
- Curriculum: Phase A (epochs 1–15, clean + aug_A) → Phase B (epochs 16–25, clean + aug_B) → Phase C (epochs 26+, clean + aug_C)
- Notebook updated 2026-05-27: paths, shape (259→300), curriculum DataLoader switching, SpecAugment spec-correct, ONNX export cell added

#### 3b. ProtoNet (metric learning, prototypical loss + SupCon λ=0.1)
- Input: 128-bin log-mel + SpecAugment
- Episode config: N=7 classes, K=5 support shots, Q=15 query shots per episode
- Encoder: reuse CNN encoder from Stage 3a (shared weights — train CNN first)
- Loss: `L = L_proto + 0.1 × L_SupCon`
- Curriculum: same three-phase schedule
- Visualisation: t-SNE of 7-class embedding space after each curriculum phase

**Outputs:** `models/cnn/` · `models/protonet/` — each with weights + MLflow metrics + training curves

---

### Stage 4 — Frozen-transfer + unsupervised detector training (W2V2-L2 · Conv-AE · OC-SVM) ○

**Goal:** Train the frozen out-of-species transfer arm plus both unsupervised detectors, under the curriculum (Conv-AE / OC-SVM on background-class audio only).
**Proposal ref:** §3.5.1 (Table 4), §3.3.1
**Proposal weeks:** W5–W6
**Notebooks:** `notebooks/04a-train-w2v2-l2.ipynb` · `notebooks/04b-train-conv-ae.ipynb` · `notebooks/04c-train-oc-svm.ipynb`

#### 4a. W2V2-L2 (frozen out-of-species transfer, cross-entropy + focal weighting) — **new arm, replaces tiny-AST**
- Input: 3 s mono waveform resampled to 16 kHz → frozen wav2vec 2.0 base, **truncated at layer 2** → 768-dim mean-pooled embedding
- Encoder **frozen in every phase** (~10 % of full wav2vec params retained); only a 768→7 linear head is trained — no fine-tuning of the embedding (Geldenhuys & Niesler, 2026)
- Frameworks: HuggingFace `transformers` + `torchaudio` for embedding extraction; linear head in PyTorch (scikit-learn `LogisticRegression` fallback)
- Curriculum: same three-phase schedule; only the linear head receives gradient updates
- Directly answers RQ5 — does the out-of-species advantage extend to non-biological threats and survive INT8 quantisation?
- Depends on Stage 2 emitting the 16 kHz raw-waveform shards (step 4b); independent of the CNN, so can train in parallel with Stage 3

#### 4b. Conv-AE (MSE reconstruction)
- Input: 128-bin log-mel + compound augmentation
- Architecture: 3-layer encoder → 16×16×128 latent; symmetric decoder
- Training data: `background_animals` + `background_wind_rain` only
- Curriculum:
  - Phase A (W3–W4): clean background clips only
  - Phase B (W5): medium compound-degraded background clips
  - Phase C (W6): full compound augmentation on background (learnability filter applied)
- Anomaly threshold: 95th percentile reconstruction error on held-out background val set

#### 4c. OC-SVM (one-class hinge, classical baseline)
- Input: 120-dim MFCC+Δ+ΔΔ + compound augmentation
- Kernel: RBF; ν=0.1 contamination; γ selected via grid search
- Training data: background classes only, same curriculum phasing
- Note: OC-SVM does not benefit from deep curriculum in the same way — apply augmented features from Phase C directly

**Outputs:** `models/w2v2_l2/` · `models/conv_ae/` · `models/oc_svm/` with weights, embeddings/linear head, reconstruction thresholds, decision boundaries

---

### Stage 5 — Comparative evaluation (nine-technique framework) ○

**Goal:** The headline study answering RQ1 and RQ2. Select the winning model for deployment.
**Proposal ref:** §3.5.2, §1.3.1 Objective 3
**Proposal week:** W7
**Notebook:** `notebooks/05_comparative_eval.ipynb`

**Tasks:**
1. Run all nine evaluation techniques (§4 above) on all five models using the locked test set
2. Compile four-paradigm results table: supervised (CNN) vs metric (ProtoNet) vs frozen transfer (W2V2-L2) vs unsupervised anomaly (Conv-AE / OC-SVM)
3. Run anomaly injection tests at SNR 5 / 10 / 20 dB for all models
4. Run threshold ablation sweeps; identify optimal operating threshold per model
5. Benchmark inference latency on Raspberry Pi 4 (not laptop) for all five models
6. Measure energy draw on Pi 4 during inference
7. Select winning model: highest AUC-ROC + F1 + lowest FPR + meets latency / size constraints

**Outputs:**
- `reports/comparative_eval.md` — four-paradigm results table
- `reports/figures/` — AUC-ROC curves, confusion matrices, t-SNE embedding plots
- `reports/winning_model.md` — selection rationale

---

### Stage 6 — Edge deployment (TFLite INT8 · Pi 4 integration) ○

**Goal:** Convert winning model to TFLite INT8 and integrate into the `alertrack/` runtime daemon.
**Proposal ref:** §3.5.1, §3.4
**Proposal week:** W8

**Tasks:**
1. Post-training quantisation: winning model → TFLite INT8 via `scripts/export_model.py`
2. Verify: model size < 10 MB; accuracy degradation < 2 % from float32 baseline
3. Integrate TFLite model into [alertrack/inference/model.py](alertrack/inference/model.py)
4. Wire [alertrack/audio/recorder.py](alertrack/audio/recorder.py) → [alertrack/audio/preprocess.py](alertrack/audio/preprocess.py) → `ModelInference`
5. Benchmark on Pi 4: end-to-end inference latency (audio capture → class label) — target < 25 s (leaving 5 s margin for SMS transmission)
6. Benchmark continuous power draw — target ≤ 5 W
7. Validate systemd service auto-restart on [alertrack/alertrack.service](alertrack/alertrack.service)

**Artefact:** Deployed TFLite model in `models/` · updated `alertrack/` runtime · latency + energy benchmark report

---

### Stage 7 — SIM808 GPS + GSM SMS integration ○

**Goal:** Add GPS coordinate tagging and SMS alert delivery via SIM808. Answers RQ3.
**Proposal ref:** §2.3.5, §3.4, §3.5.2 (technique 8)
**Proposal weeks:** W8–W9

#### 7a. SIM808 GPS coordinate acquisition
- SIM808 built-in GPS outputs NMEA sentences via UART serial at 9600 baud
- Enable GPS: `AT+CGNSPWR=1`, then `AT+CGNSOUT=1` to stream NMEA on the serial port
- Existing NMEA parser in [alertrack/sensors/gps.py](alertrack/sensors/gps.py) handles `$GPGGA` / `$GPRMC` directly — no code change required
- Alternative: poll `AT+CGNSINF` for structured single-line response (lat, lon, alt, speed, fix status)
- Require ≥ 3 satellites for a valid fix before any alert is dispatched; fall back to last known coordinates if fix lost
- Validate fix quality: log satellite count and fix status in SQLite `SystemStatus`
- Target: valid fix within 60 s cold start; coordinate accuracy within 5 m CEP (SIM808 spec)

#### 7b. SIM808 GSM SMS alert delivery
- Implement full AT command sequence in [alertrack/alerts/notifier.py](alertrack/alerts/notifier.py)
- Commands: `AT+CMGF=1` (text mode) → `AT+CMGS="<number>"` → message → `chr(26)` (send)
- SMS payload format: `ALERTRECK | <class> | Conf: x% | GPS: lat,lon | HH:MM`
- Network: MTN or Airtel Rwanda GSM (quad-band); no data plan required — SMS only
- Retry logic: up to 3 attempts with 5 s backoff; log each attempt and outcome to SQLite `AlertLog`
- Test end-to-end: loudspeaker playback → USB mic → inference → SMS delivered → measure 95th-percentile latency across 100 trials; target < 30 s

#### 7c. SQLite event database
- Implement in [alertrack/storage/logger.py](alertrack/storage/logger.py) and [alertrack/storage/evidence.py](alertrack/storage/evidence.py)
- Schema: `DetectedEvent` · `AudioSegment` · `AlertLog` · `SystemStatus`
- Log every detected event regardless of alert threshold (for audit and false-positive analysis)

**Artefact:** SIM808 GPS/SMS pipeline · SQLite event log · GPS fix validation report

---

### Stage 8 — Grad-CAM explainability + Flask LAN dashboard + final evaluation ○

**Goal:** Add explainability layer, build LAN dashboard, and complete the full nine-technique evaluation. Answers RQ4.
**Proposal ref:** §1.3.1 Objective 3, §3.5.2 (technique 9), §3.4
**Proposal weeks:** W9–W10

#### 8a. Grad-CAM (winning model only)
- Library: `pytorch-grad-cam` or `tf-explain` depending on winning model framework
- Apply to mel-spectrogram inputs of the winning model
- Generate heatmaps for each of the 7 threat classes using representative test clips
- Qualitative validation: do highlighted regions match expected acoustic features?
  - `threat_gunshot`: sharp transient in low-to-mid frequency range
  - `threat_chainsaw`: harmonic stack at 50–200 Hz + overtones
  - `threat_vehicle`: low-frequency engine harmonics
  - `threat_human`: voiced speech formants (300–3000 Hz)
  - `threat_dog`: bark envelope with formant structure

#### 8b. Flask LAN dashboard
- Implement in `alertrack/` or separate `dashboard/` module
- Served at port 5000 on Pi 4 local network interface (no external internet access)
- Pages: real-time anomaly score stream · class label + confidence · GPS coordinates (lat/lon from SIM808) · Grad-CAM spectrogram overlay · alert history from SQLite · system health (CPU temp, uptime, GSM signal dB)

#### 8c. Full nine-technique evaluation (system-level)
- Re-run techniques 6–8 (latency, energy, GPS fix quality) on the complete integrated system (not models in isolation)
- 100-trial latency test: loudspeaker playback → SMS delivered → measure 95th percentile
- Produce final `reports/system_eval.md`

**Artefact:** Grad-CAM heatmap gallery · Flask dashboard code · `reports/system_eval.md`

---

### Stage 9 — Final report + submission package ○

**Goal:** Reproduce all results from scratch in ≤ 1 command; produce final thesis.
**Proposal ref:** §1.8 (W10+)
**Proposal week:** W10

**Tasks:**
1. Write `Makefile` or `scripts/reproduce.sh`: one command runs full pipeline (download → preprocess → train 5 models under curriculum → evaluate → export → deploy)
2. Finalize data card, model card, and experiment registry in MLflow
3. Confirm all result tables in thesis match outputs of `make reproduce`
4. Thesis chapters follow proposal section numbering:
   - Chapter 1: Introduction (maps to §1)
   - Chapter 2: Literature Review (maps to §2)
   - Chapter 3: Methodology (maps to §3)
   - Chapter 4: Results & Discussion (comparative eval tables + Grad-CAM analysis)
   - Chapter 5: Conclusion + future work

**Artefact:** `make reproduce` command · final thesis PDF · submission package

---

## 6. Ten-week Gantt (aligned to proposal Table 11)

| Phase | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 | W9 | W10 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Phase 1:** Data & Preprocessing (Stages 1–2) | █ | █ | | | | | | | | |
| **Phase 2:** Supervised & Metric Models — CNN, ProtoNet (Stage 3) | | | █ | █ | █ | | | | | |
| **Phase 3:** Frozen Transfer & Unsupervised Detectors — W2V2-L2, Conv-AE, OC-SVM (Stage 4) | | | | | █ | █ | | | | |
| **Phase 4:** Comparative Evaluation (Stage 5) | | | | | | | █ | | | |
| **Phase 5:** Edge Deployment (Stage 6) | | | | | | | | █ | | |
| **Phase 6:** Direction + GSM Integration (Stage 7) | | | | | | | | █ | █ | |
| **Phase 7:** Grad-CAM, Dashboard & Evaluation (Stage 8) | | | | | | | | | █ | █ |
| **Phase 8:** Writing & Submission (Stage 9) | | | | | | | | | | █ |

> **Current date: 2026-06-01.** Stage 1 is complete; Stage 2 needs the new 16 kHz waveform branch (step 4b) for W2V2-L2. Stage 3a (CNN) notebook is ready and uploading to Kaggle for GPU training. The model line-up was updated 2026-06-01 (tiny-AST → W2V2-L2). All hardware is in hand and the Pi is accessible at `alertreck.local`.

---

## 7. Status snapshot (2026-06-01)

| Stage | Name | Status | Next action |
|---|---|---|---|
| 0 | Design lock-in | ● done | — |
| 1 | Data engineering | ● done | — |
| 2 | Preprocessing pipeline | ◐ | Add 16 kHz raw-waveform branch (step 4b) for W2V2-L2 + re-run; re-run with `--dir-ir` once USB mic hum resolved |
| 3a | CNN training | ◐ | Upload shards to Kaggle, run `03a-train-cnn.ipynb` on GPU |
| 3b | ProtoNet training | ○ | After CNN encoder weights available (Stage 3a) |
| 4a | W2V2-L2 frozen transfer | ○ | New arm (replaces tiny-AST); needs 16 kHz waveform shards — extract frozen layer-2 embeddings, train linear head |
| 4b | Conv-AE training | ○ | Unlocks learnability filter (step 6e) in preprocessing |
| 4c | OC-SVM training | ○ | After Conv-AE threshold established |
| 5 | Comparative evaluation | ○ | All nine techniques on all five models across four paradigms |
| 6 | Edge deployment | ◐ skeleton | Convert winning model to TFLite INT8; plug into `alertrack/` daemon |
| 7 | SIM808 GPS + GSM | ◐ skeleton | Hardware in hand; wire NMEA reader + AT command SMS; add ranger numbers to `config.py` |
| 8 | Grad-CAM + dashboard | ○ | After winning model selected in Stage 5 |
| 9 | Final report + submission | ○ | `make reproduce` + thesis writing |

Legend: ○ not started · ◐ partial / skeleton exists · ● done

---

## 8. Decisions logged

| Date | Decision | Reason |
|---|---|---|
| 2026-04-15 | `threat_dog` added as 6th/7th threat class | Poacher-dog indicator added per proposal amendment |
| 2026-04-19 | `normal` split → `background_animals` + `background_wind_rain` | Cleaner source tracking; balanced augmentation control |
| 2026-04-19 | Old YAMNet notebook deleted | Replaced by five-model comparative architecture |
| 2026-04-19 | MLflow chosen for experiment tracking | Offline-first — no cloud dependency |
| 2026-05-24 | ProtoNet added as third model paradigm | Proposal finalised; metric learning is the primary novel contribution |
| 2026-05-25 | USB microphone + SIM808 replaces ReSpeaker 4-Mic Array + SIM800L | SIM808 built-in GPS provides device coordinates for alert tagging; simpler hardware stack, lower cost, no multi-channel sync required |
| 2026-05-24 | Compound augmentation + curriculum training added | §2.3.7 from Mega-ASR (Xie et al., 2026); prevents collapse on field-deployment compound degradation conditions |
| 2026-05-27 | Stage 1 complete — 8,907 clips, all 7 class targets met | `threat_chainsaw` reached 567 (target 500); dataset locked for training |
| 2026-05-27 | Stage 2 complete — preprocessing pipeline runs end-to-end | All shards generated: 7,359 clean train windows + 44,154 curriculum aug windows; DIR calibration deferred pending USB mic hum fix |
| 2026-05-27 | All hardware in hand and deployed | Pi at alertreck.local (192.168.1.88); USB mic, SIM808, 32 GB SD, Rwanda SIM all confirmed |
| 2026-05-27 | CNN notebook updated for new data | Shape 259→300, curriculum DataLoader, SpecAugment corrected, ONNX export added; ready for Kaggle GPU run |
| 2026-06-01 | **tiny-AST dropped; replaced by W2V2-L2** (frozen wav2vec 2.0 layer-2 + linear head) | Updated proposal §2.3.4, §3.5.1: fine-tuning a transformer on a small corpus overfits under domain shift. A frozen, truncated layer-2 embedding is a genuinely distinct 4th paradigm (frozen out-of-species transfer) and the empirical vehicle for the new RQ5 (Geldenhuys & Niesler, 2026). `notebooks/03b-train-tiny-ast.ipynb` and `models/tiny_ast/` are superseded |
| 2026-06-01 | Study is now a **four-paradigm** comparison (was three) | Supervised (CNN) · metric (ProtoNet) · frozen transfer (W2V2-L2) · unsupervised anomaly (Conv-AE / OC-SVM); RQ5 added |
| 2026-06-01 | Preprocessing gains a **16 kHz raw-waveform branch (step 4b)** | wav2vec 2.0 was pretrained at 16 kHz; W2V2-L2 needs native-rate waveform input. MFCC branch renumbered 4b → 4c; Stage 2 reopened to emit waveform shards |

---

## 9. Critical path and blockers

**Current critical path:** Stage 3a (CNN, Kaggle GPU) → Stage 3b (ProtoNet, needs CNN encoder) → Stage 4b (Conv-AE) → Stage 4c (OC-SVM) → Stage 5 (comparative eval) → Stage 6 (deployment). Stage 4a (W2V2-L2) is independent of the CNN and can train in parallel once the 16 kHz waveform shards exist.

**Resolved blockers:**
- ~~`threat_chainsaw` gap~~ — 567 clips, target met
- ~~Stage 2 preprocessing rewrite~~ — mel/mfcc shards on disk

**Active dependencies:**
- Stage 2 must emit the **16 kHz raw-waveform shards (step 4b)** before Stage 4a (W2V2-L2) can extract frozen layer-2 embeddings — new requirement from the updated proposal
- Stage 3b (ProtoNet) cannot start until Stage 3a (CNN) produces `best_model.pt` — encoder weights are shared
- Stage 4b (Conv-AE) trains independently on background-class audio; unlocks learnability filter (step 6e) for a future preprocessing re-run
- DIR calibration (step 5) pending USB mic hum fix — resolve via powered USB hub or gain reduction in `alsamixer`; re-run preprocessing with `--dir-ir usb_mic_ir.wav` before final evaluation

**Hardware:** All five components in hand. Pi at `alertreck.local` (192.168.1.88), SSH accessible. SIM808 wiring to Pi UART pending Stage 7. Ranger phone numbers not yet added to `alertrack/config.py` (`RANGER_PHONE_NUMBERS`).
