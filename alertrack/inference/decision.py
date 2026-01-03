"""
Threat Decision Logic Module
=============================
Applies thresholds and cooldown logic to determine if a threat should trigger an alert.
Prevents alert spam and manages threat state.
"""

import time
from typing import Optional, Dict, Tuple
from collections import defaultdict

from ..config import (
    THREAT_THRESHOLD, THREAT_CONTEXT_THRESHOLD,
    BACKGROUND_THRESHOLD, COOLDOWN_SECONDS,
    THREAT_IDX, THREAT_CONTEXT_IDX, BACKGROUND_IDX,
    CLASS_NAMES, DEBUG_MODE
)


class ThreatDecisionEngine:
    """
    Determines if model predictions should trigger alerts.
    Implements thresholds and cooldown logic.
    """
    
    def __init__(self):
        """Initialize threat decision engine."""
        self.threat_threshold = THREAT_THRESHOLD
        self.threat_context_threshold = THREAT_CONTEXT_THRESHOLD
        self.background_threshold = BACKGROUND_THRESHOLD
        self.cooldown_seconds = COOLDOWN_SECONDS
        
        # Track last alert time for each threat type
        self.last_alert_times: Dict[str, float] = defaultdict(lambda: 0.0)
        
        # Statistics
        self.total_predictions = 0
        self.total_threats_detected = 0
        self.total_threats_suppressed = 0  # Due to cooldown
        self.threat_counts = defaultdict(int)
        
        print(f"ThreatDecisionEngine initialized:")
        print(f"  THREAT threshold: {self.threat_threshold}")
        print(f"  THREAT_CONTEXT threshold: {self.threat_context_threshold}")
        print(f"  Cooldown: {self.cooldown_seconds}s")
    
    def evaluate(
        self,
        class_idx: int,
        confidence: float,
        probabilities: list
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Evaluate if prediction should trigger an alert.
        
        Args:
            class_idx: Predicted class index
            confidence: Confidence score
            probabilities: All class probabilities
            
        Returns:
            Tuple of (should_alert, threat_info)
            threat_info is None if no alert should be triggered
        """
        self.total_predictions += 1
        
        class_name = CLASS_NAMES[class_idx]
        current_time = time.time()
        
        # Determine if this is a threat based on class and confidence
        is_threat = False
        threat_level = "NONE"
        
        if class_idx == THREAT_IDX and confidence >= self.threat_threshold:
            is_threat = True
            threat_level = "HIGH"
        elif class_idx == THREAT_CONTEXT_IDX and confidence >= self.threat_context_threshold:
            is_threat = True
            threat_level = "MEDIUM"
        elif class_idx == BACKGROUND_IDX:
            # Background - no threat
            is_threat = False
        else:
            # Below threshold - uncertain
            if DEBUG_MODE:
                print(f"Below threshold: {class_name} confidence={confidence:.3f}")
            is_threat = False
        
        if not is_threat:
            return False, None
        
        # Threat detected! Check cooldown
        self.total_threats_detected += 1
        self.threat_counts[class_name] += 1
        
        last_alert_time = self.last_alert_times[class_name]
        time_since_last = current_time - last_alert_time
        
        if time_since_last < self.cooldown_seconds:
            # Still in cooldown period
            self.total_threats_suppressed += 1
            
            if DEBUG_MODE:
                remaining = self.cooldown_seconds - time_since_last
                print(f"🔕 Threat suppressed (cooldown): {class_name}, "
                      f"remaining={remaining:.1f}s")
            
            return False, None
        
        # Not in cooldown - trigger alert!
        self.last_alert_times[class_name] = current_time
        
        threat_info = {
            'threat_type': class_name,
            'threat_level': threat_level,
            'confidence': float(confidence),
            'class_probabilities': {
                CLASS_NAMES[i]: float(probabilities[i])
                for i in range(len(probabilities))
            },
            'timestamp': current_time
        }
        
        if DEBUG_MODE:
            print(f"🚨 THREAT ALERT: {class_name} ({threat_level}) "
                  f"confidence={confidence:.3f}")
        
        return True, threat_info
    
    def reset_cooldown(self, class_name: Optional[str] = None):
        """
        Reset cooldown for specific class or all classes.
        
        Args:
            class_name: Class to reset (None = reset all)
        """
        if class_name:
            self.last_alert_times[class_name] = 0.0
            print(f"Cooldown reset for: {class_name}")
        else:
            self.last_alert_times.clear()
            print("All cooldowns reset")
    
    def get_cooldown_status(self) -> Dict[str, float]:
        """
        Get remaining cooldown time for each threat type.
        
        Returns:
            Dictionary of {class_name: remaining_seconds}
        """
        current_time = time.time()
        status = {}
        
        for class_name, last_time in self.last_alert_times.items():
            elapsed = current_time - last_time
            remaining = max(0, self.cooldown_seconds - elapsed)
            status[class_name] = remaining
        
        return status
    
    def get_stats(self) -> Dict:
        """Get decision engine statistics."""
        return {
            'total_predictions': self.total_predictions,
            'total_threats_detected': self.total_threats_detected,
            'total_threats_suppressed': self.total_threats_suppressed,
            'threat_counts': dict(self.threat_counts),
            'cooldown_status': self.get_cooldown_status()
        }
    
    def is_threat_class(self, class_idx: int) -> bool:
        """Check if class index represents a threat."""
        return class_idx in [THREAT_IDX, THREAT_CONTEXT_IDX]
    
    def get_threat_level_description(self, class_idx: int) -> str:
        """Get human-readable threat level."""
        if class_idx == THREAT_IDX:
            return "HIGH - Immediate threat (gunshot, chainsaw, vehicle)"
        elif class_idx == THREAT_CONTEXT_IDX:
            return "MEDIUM - Potential threat indicator (dog bark)"
        else:
            return "NONE - Background/ambient sound"


def test_decision_engine():
    """Test the threat decision engine."""
    print("\n🧠 Testing ThreatDecisionEngine...")
    print("=" * 60)
    
    engine = ThreatDecisionEngine()
    
    # Test 1: High confidence THREAT
    print("\n[Test 1] High confidence THREAT")
    probabilities = [0.05, 0.05, 0.90]  # BACKGROUND, THREAT_CONTEXT, THREAT
    should_alert, info = engine.evaluate(THREAT_IDX, 0.90, probabilities)
    print(f"  Should alert: {should_alert}")
    if info:
        print(f"  Info: {info}")
    
    # Test 2: Same threat immediately (should be suppressed)
    print("\n[Test 2] Same THREAT immediately (cooldown)")
    should_alert, info = engine.evaluate(THREAT_IDX, 0.92, probabilities)
    print(f"  Should alert: {should_alert} (expected: False)")
    
    # Test 3: Different threat type
    print("\n[Test 3] Different threat type (THREAT_CONTEXT)")
    probabilities = [0.1, 0.80, 0.1]
    should_alert, info = engine.evaluate(THREAT_CONTEXT_IDX, 0.80, probabilities)
    print(f"  Should alert: {should_alert}")
    if info:
        print(f"  Info: {info}")
    
    # Test 4: Below threshold
    print("\n[Test 4] THREAT but below threshold")
    probabilities = [0.3, 0.2, 0.50]
    should_alert, info = engine.evaluate(THREAT_IDX, 0.50, probabilities)
    print(f"  Should alert: {should_alert} (expected: False)")
    
    # Test 5: Background class
    print("\n[Test 5] BACKGROUND class")
    probabilities = [0.90, 0.05, 0.05]
    should_alert, info = engine.evaluate(BACKGROUND_IDX, 0.90, probabilities)
    print(f"  Should alert: {should_alert} (expected: False)")
    
    # Show stats
    print("\n📊 Decision Engine Statistics:")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show cooldown status
    print("\n⏱️  Cooldown Status:")
    cooldown = engine.get_cooldown_status()
    for class_name, remaining in cooldown.items():
        print(f"  {class_name}: {remaining:.1f}s remaining")
    
    # Reset cooldown and test again
    print("\n[Test 6] After cooldown reset")
    engine.reset_cooldown()
    probabilities = [0.05, 0.05, 0.90]
    should_alert, info = engine.evaluate(THREAT_IDX, 0.90, probabilities)
    print(f"  Should alert: {should_alert} (expected: True)")
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")


if __name__ == "__main__":
    test_decision_engine()
