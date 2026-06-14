"""
Log existing Alertreck model results to Weights & Biases — no retraining needed.

Each model's `models/<name>/results.json` already holds the full per-epoch history and
final metrics. This script replays that into W&B so you get training curves, a summary
table comparing all models, per-class metrics, and the saved plots (confusion matrix,
ROC, etc.) — all in the W&B UI.

Usage:
    pip install wandb
    wandb login                       # paste your key from https://wandb.ai/authorize
    python scripts/log_results_to_wandb.py
    # offline test (no account needed):  WANDB_MODE=offline python scripts/log_results_to_wandb.py

One W&B run is created per model, grouped by paradigm, in project "alertreck".
"""

import json
from pathlib import Path

import wandb

REPO = Path(__file__).resolve().parent.parent
PROJECT = "alertreck"

MODELS = {
    "custom_cnn": "Supervised CNN",
    "protonet":   "Few-shot ProtoNet",
    "w2v2_l2":    "Transfer W2V2-L2",
    "conv_ae":    "Unsupervised Conv-AE",
    "oc_svm":     "Classical OC-SVM",
}

# Saved plots to attach if present
PLOTS = [
    "confusion_matrix.png", "roc_pr_curves.png", "roc_curves.png",
    "training_curves.png", "error_distribution.png", "score_distributions.png",
]


def log_model(name: str, paradigm: str):
    results_path = REPO / "models" / name / "results.json"
    if not results_path.exists():
        print(f"  skip {name}: no results.json")
        return

    r = json.loads(results_path.read_text())
    history = r.pop("history", {})

    # config = scalar / list-of-non-history hyperparameters
    config = {"model": name, "paradigm": paradigm}
    for k, v in r.items():
        if isinstance(v, (int, float, str, bool)):
            config[k] = v

    run = wandb.init(project=PROJECT, name=name, group=paradigm,
                     job_type="eval", config=config, reinit=True)

    # 1. Replay per-epoch training curves
    numeric = {k: v for k, v in history.items()
               if isinstance(v, list) and v and isinstance(v[0], (int, float))}
    if numeric:
        n_epochs = min(len(v) for v in numeric.values())
        for i in range(n_epochs):
            wandb.log({f"train/{k}": numeric[k][i] for k in numeric}, step=i + 1)
        print(f"  {name}: logged {n_epochs} epochs ({', '.join(numeric)})")

    # 2. Final scalar metrics → run summary
    for k, v in r.items():
        if isinstance(v, (int, float)):
            run.summary[k] = v

    # 3. Per-class dicts → summary scalars + a W&B table
    for k, v in r.items():
        if isinstance(v, dict) and k.startswith("per_class"):
            for cls, val in v.items():
                if isinstance(val, (int, float)):
                    run.summary[f"{k}/{cls}"] = val
            table = wandb.Table(columns=["class", k.replace("per_class_", "")],
                                data=[[c, val] for c, val in v.items()])
            wandb.log({f"{k}_table": table})

    # 4. Attach saved plots
    imgs = {}
    for png in PLOTS:
        p = REPO / "models" / name / png
        if p.exists():
            imgs[png.replace(".png", "")] = wandb.Image(str(p))
    if imgs:
        wandb.log(imgs)
        print(f"  {name}: attached {len(imgs)} plot(s)")

    run.finish()


def main():
    print(f"Logging {len(MODELS)} models to W&B project '{PROJECT}'")
    for name, paradigm in MODELS.items():
        log_model(name, paradigm)
    print("Done. Open the project at https://wandb.ai/<your-entity>/alertreck")


if __name__ == "__main__":
    main()
