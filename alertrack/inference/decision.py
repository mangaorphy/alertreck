"""
Threat Decision Logic
=====================
Evaluates per-class confidence against per-class thresholds.
Every fine-grained threat class is detected independently —
no class collapsing, no coarse grouping.

Alert hierarchy:
  background_animals   (0) — silent
  background_wind_rain (1) — silent
  threat_chainsaw      (2) — HIGH
  threat_dog           (3) — MEDIUM
  threat_gunshot       (4) — HIGH
  threat_human         (5) — HIGH
  threat_vehicle       (6) — HIGH
"""

import time
from typing import Optional, Dict, Tuple
from collections import defaultdict

import numpy as np

from ..config import (
    CLASS_NAMES, BACKGROUND_CLASSES, THREAT_CONFIG, DEBUG_MODE
)


class ThreatDecisionEngine:
    """
    Per-class threshold and independent cooldown for all 7 fine-grained classes.
    Background classes never trigger alerts.
    """

    def __init__(self):
        self.last_alert_times: Dict[str, float] = defaultdict(float)

        # Statistics
        self.total_predictions      = 0
        self.total_threats_detected = 0
        self.total_suppressed       = 0
        self.threat_counts          = defaultdict(int)

        print("ThreatDecisionEngine initialised (fine-grained 7-class):")
        for name, (thresh, level, cooldown) in THREAT_CONFIG.items():
            print(f"  {name:<26}  thresh={thresh}  level={level}  cooldown={cooldown}s")

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        class_idx: int,
        confidence: float,
        probabilities: np.ndarray,
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Decide whether to fire an alert for the top predicted class.

        Returns:
            (should_alert, threat_info | None)
        """
        self.total_predictions += 1
        class_name = CLASS_NAMES[class_idx]

        # ── Background — always silent ────────────────────────────────────────
        if class_name in BACKGROUND_CLASSES:
            return False, None

        # ── Not a configured threat class — ignore ────────────────────────────
        if class_name not in THREAT_CONFIG:
            return False, None

        threshold, level, cooldown = THREAT_CONFIG[class_name]

        # ── Below threshold — uncertain ───────────────────────────────────────
        if confidence < threshold:
            if DEBUG_MODE:
                print(f"Below threshold: {class_name}  conf={confidence:.3f}  "
                      f"(need {threshold})")
            return False, None

        # ── Threat above threshold — check cooldown ───────────────────────────
        self.total_threats_detected += 1
        self.threat_counts[class_name] += 1

        now = time.time()
        elapsed = now - self.last_alert_times[class_name]

        if elapsed < cooldown:
            self.total_suppressed += 1
            if DEBUG_MODE:
                print(f"Suppressed (cooldown): {class_name}  "
                      f"remaining={cooldown - elapsed:.0f}s")
            return False, None

        # ── Fire! ─────────────────────────────────────────────────────────────
        self.last_alert_times[class_name] = now

        # Top-3 predictions (highest confidence first) — sent to rangers so they
        # can judge a borderline call. e.g. gunshot 71% / human 22% / vehicle 5%
        # signals more uncertainty than gunshot 96% / wind 3% / animals 1%.
        order = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            {"class": CLASS_NAMES[i], "confidence": float(probabilities[i])}
            for i in order
        ]

        threat_info = {
            "threat_type":        class_name,
            "threat_level":       level,
            "confidence":         float(confidence),
            "top_predictions":    top_predictions,
            "class_probabilities": {
                CLASS_NAMES[i]: float(probabilities[i])
                for i in range(len(probabilities))
            },
            "timestamp": now,
        }

        if DEBUG_MODE:
            print(f"ALERT: {class_name}  ({level})  conf={confidence:.3f}")

        return True, threat_info

    def reset_cooldown(self, class_name: Optional[str] = None):
        if class_name:
            self.last_alert_times[class_name] = 0.0
        else:
            self.last_alert_times.clear()

    def get_cooldown_status(self) -> Dict[str, float]:
        now = time.time()
        return {
            name: max(0.0, THREAT_CONFIG[name][2] - (now - last))
            for name, last in self.last_alert_times.items()
            if name in THREAT_CONFIG
        }

    def get_stats(self) -> Dict:
        return {
            "total_predictions":      self.total_predictions,
            "total_threats_detected": self.total_threats_detected,
            "total_suppressed":       self.total_suppressed,
            "threat_counts":          dict(self.threat_counts),
            "cooldown_status":        self.get_cooldown_status(),
        }

    def is_threat_class(self, class_idx: int) -> bool:
        return CLASS_NAMES[class_idx] in THREAT_CONFIG

    def get_threat_level(self, class_idx: int) -> str:
        name = CLASS_NAMES[class_idx]
        cfg  = THREAT_CONFIG.get(name)
        return cfg[1] if cfg else "NONE"


def test_decision_engine():
    print("\nTesting ThreatDecisionEngine...")
    print("=" * 60)

    engine = ThreatDecisionEngine()
    dummy_probs = np.zeros(7, dtype=np.float32)

    tests = [
        (4, 0.92, "gunshot HIGH — should alert"),
        (4, 0.92, "gunshot again immediately — suppressed"),
        (2, 0.88, "chainsaw — should alert"),
        (3, 0.82, "dog MEDIUM — should alert"),
        (0, 0.99, "bg_animals — silent"),
        (1, 0.99, "bg_wind — silent"),
        (5, 0.50, "human below threshold — no alert"),
        (5, 0.90, "human above threshold — should alert"),
    ]

    for idx, conf, label in tests:
        dummy_probs[:] = 0
        dummy_probs[idx] = conf
        alert, info = engine.evaluate(idx, conf, dummy_probs)
        status = "ALERT" if alert else "no alert"
        print(f"  [{CLASS_NAMES[idx]:<26}  conf={conf}]  → {status}  ({label})")

    print("\nStats:", engine.get_stats())
    print("=" * 60)


if __name__ == "__main__":
    test_decision_engine()
