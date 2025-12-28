# AudioSet Dataset Organization

This directory contains audio samples organized by threat level categories for classification tasks.

## Dataset Structure

```
AUDIOSET METADATA/
├── THREAT/                    # Direct threat sounds
│   ├── gunshot/              (598 files)
│   ├── chainsaw/             (628 files)
│   └── human_voice/          (1,114 files)
│
├── THREAT_CONTEXT/            # Contextual threat indicators
│   └── dog_bark/             (1,105 files)
│
├── BACKGROUND/                # Environmental/background sounds
│   ├── animal_sound/
│   │   ├── quadrupeds/       (585 files)
│   │   ├── birds/            (519 files)
│   │   └── insects/          (703 files)
│   ├── wind_rain/            (1,123 files)
│   └── ambient_noise/        (359 files)
│
└── scripts/                   # Download scripts
    ├── download_animal_sounds.py
    ├── download_chainsaws.py
    ├── download_env_noise.py
    ├── download_human_sounds.py
    ├── download_rain.py
    └── download_wind.py
```

## Class Labels

| Class ID | Class Name      | Category        | File Count |
|----------|----------------|-----------------|------------|
| 0        | gunshot        | THREAT          | 598        |
| 1        | chainsaw       | THREAT          | 628        |
| 2        | vehicle_engine | THREAT          | -          |
| 3        | human_voice    | THREAT          | 1,114      |
| 4        | dog_bark       | THREAT_CONTEXT  | 1,105      |
| 5        | animal_sound   | BACKGROUND      | 1,807      |
| 6        | wind_rain      | BACKGROUND      | 1,123      |
| 7        | ambient_noise  | BACKGROUND      | 359        |

**Total Audio Files: 6,734**

## Category Descriptions

### THREAT
Direct threat sounds that require immediate attention:
- **gunshot**: Gunfire and gunshot sounds
- **chainsaw**: Chainsaw operating sounds
- **human_voice**: Various human vocalizations (speech, screaming, etc.)

### THREAT_CONTEXT
Sounds that may indicate a threatening situation:
- **dog_bark**: Dog barking (can indicate intruders or danger)

### BACKGROUND
Environmental and ambient sounds:
- **animal_sound**: Wildlife sounds (quadrupeds, birds, insects)
- **wind_rain**: Weather-related sounds (wind and rain combined)
- **ambient_noise**: General environmental noise

## Audio Format

- **Format**: WAV
- **Duration**: ~10 seconds per clip
- **Source**: AudioSet (Google) and S3 bucket (ecosight-training-data)

## Data Sources

- AudioSet metadata: `class_labels_indices.csv`, `unbalanced_train_segments.csv`
- Downloaded using yt-dlp for YouTube sources
- S3 bucket sync for pre-extracted audio (gunshot, dog_bark)

## Scripts

All download scripts are located in the `scripts/` folder and use:
- Python 3.13
- yt-dlp for YouTube audio extraction
- ffmpeg for audio processing
- AWS CLI for S3 downloads

---

*Dataset compiled: December 2025*
