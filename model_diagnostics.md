# YAMNet Model Diagnostic Checklist

## Performance: 93.81% Test Accuracy

### ⚠️ CRITICAL: False Negative Analysis (44 THREAT → BACKGROUND)

**Priority:** HIGHEST - Missing threats is unacceptable in anti-poaching

#### Step 1: Identify Misclassified Samples
```python
# Add to notebook after test evaluation:
import pandas as pd

# Find all misclassified THREAT samples
threat_indices = np.where(y_test == 2)[0]  # THREAT class = 2
missed_threats_mask = (y_test == 2) & (y_test_pred == 0)  # Predicted as BACKGROUND
missed_threat_indices = np.where(missed_threats_mask)[0]

print(f"Total THREAT samples: {len(threat_indices)}")
print(f"Missed (FN): {len(missed_threat_indices)} ({len(missed_threat_indices)/len(threat_indices)*100:.2f}%)")

# Get confidence scores for missed threats
missed_threat_probs = y_test_pred_proba[missed_threat_indices]
missed_threat_confidences = np.max(missed_threat_probs, axis=1)
threat_class_probs = missed_threat_probs[:, 2]  # THREAT class probabilities

print(f"\nConfidence analysis for MISSED threats:")
print(f"  Predicted as BACKGROUND with confidence: {missed_threat_confidences.mean():.2%} ± {missed_threat_confidences.std():.2%}")
print(f"  Actual THREAT probability: {threat_class_probs.mean():.2%} ± {threat_class_probs.std():.2%}")
print(f"  Min THREAT probability: {threat_class_probs.min():.2%}")
print(f"  Max THREAT probability: {threat_class_probs.max():.2%}")

# How many are close to threshold?
near_threshold = np.sum((threat_class_probs >= 0.70) & (threat_class_probs < 0.85))
print(f"\nMissed threats with 70-85% confidence: {near_threshold} ({near_threshold/len(missed_threat_indices)*100:.1f}%)")
```

#### Step 2: Analyze by Threat Subtype
```python
# Map test indices back to original files to identify threat types
# You'll need to track which files correspond to which test samples
# This requires saving file paths during data split

# Example analysis:
# threat_types = ['gunshot', 'chainsaw', 'vehicle_engine', 'human_voice']
# for threat_type in threat_types:
#     type_mask = [threat_type in file_path for file_path in test_file_paths[missed_threat_indices]]
#     print(f"{threat_type}: {sum(type_mask)} missed")
```

#### Step 3: Threshold Sensitivity Analysis
```python
# Test different thresholds
thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

print("Threshold Sensitivity Analysis for THREAT Detection:")
print("=" * 70)
print(f"{'Threshold':<12} {'Recall':<10} {'Precision':<12} {'F1-Score':<10} {'FN':<8} {'FP':<8}")
print("-" * 70)

for threshold in thresholds:
    # Predict with custom threshold
    y_pred_custom = np.zeros_like(y_test)
    for i in range(len(y_test)):
        if y_test_pred_proba[i, 2] >= threshold:  # THREAT class
            y_pred_custom[i] = 2
        elif y_test_pred_proba[i, 1] >= 0.75:  # THREAT_CONTEXT
            y_pred_custom[i] = 1
        else:
            y_pred_custom[i] = 0
    
    # Calculate metrics for THREAT class
    threat_mask = y_test == 2
    threat_pred_mask = y_pred_custom == 2
    
    tp = np.sum(threat_mask & threat_pred_mask)
    fn = np.sum(threat_mask & ~threat_pred_mask)
    fp = np.sum(~threat_mask & threat_pred_mask)
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{threshold:<12.2f} {recall:<10.2%} {precision:<12.2%} {f1:<10.3f} {fn:<8} {fp:<8}")

print("-" * 70)
print("\n💡 Recommendation: Choose threshold that minimizes FN (even if FP increases)")
print("   In anti-poaching: Missing 1 threat > 10 false alarms")
```

---

### 📊 Data Quality Checks

#### Check 1: Class Distribution in Splits
```python
# Verify stratification worked correctly
print("Class distribution across splits:")
for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    for cls in range(3):
        pct = np.sum(y_split == cls) / len(y_split) * 100
        print(f"{split_name} - {class_names[cls]}: {pct:.1f}%")
```

#### Check 2: Augmentation Impact
```python
# Check if augmentation is creating unrealistic samples
# Compare performance on original vs augmented samples
# (Requires tracking which samples are augmented during extraction)
```

#### Check 3: Audio Quality
```python
# Listen to misclassified samples manually
# Check for:
# - Low volume / clipping
# - Excessive noise
# - Mislabeled data
# - Corrupted files
```

---

### 🎯 Recommended Model Improvements

#### Option 1: Adjust Thresholds (Quick Win)
- **Current:** THREAT ≥ 85%, THREAT_CONTEXT ≥ 75%
- **Recommended:** THREAT ≥ 70%, THREAT_CONTEXT ≥ 75%
- **Impact:** Reduce FN from 44 → ~20-25, increase FP from 39 → ~60-80
- **Trade-off:** Acceptable for anti-poaching (better safe than sorry)

#### Option 2: Cost-Sensitive Learning (Retrain)
```python
# Modify loss function to penalize FN more than FP
# Custom loss or adjust class weights asymmetrically

# Example: Increase THREAT class weight
class_weight_dict_v2 = {
    0: class_weight_dict[0],
    1: class_weight_dict[1],
    2: class_weight_dict[2] * 2.0  # Double penalty for missing THREAT
}
```

#### Option 3: Ensemble Model
```python
# Train 3 separate models:
# 1. YAMNet (current)
# 2. Custom CNN on mel spectrograms
# 3. MobileNetV2
# 
# Voting: If ANY model predicts THREAT ≥ threshold → Alert
# Reduces FN significantly
```

#### Option 4: Separate Models per Threat Type
```python
# Instead of 3-class (BACKGROUND, THREAT_CONTEXT, THREAT)
# Train 6 binary classifiers:
# - Gunshot vs Not-Gunshot
# - Chainsaw vs Not-Chainsaw
# - Vehicle vs Not-Vehicle
# - etc.
#
# Each can have optimized threshold
# More compute, but higher precision
```

---

### 🚀 Immediate Next Steps

1. **✅ Add threshold sensitivity analysis** (copy code above to notebook)
2. **✅ Identify which THREAT subtypes are being missed** (gunshot? chainsaw?)
3. **✅ Lower THREAT threshold to 70%** and re-evaluate on test set
4. **✅ Listen to 5-10 misclassified samples** to check data quality
5. **✅ Calculate cost-adjusted metrics** (weight FN 3x higher than FP)

---

### 📈 Performance Targets for Production

**Current:**
- Recall (THREAT): 92.0%
- Precision (THREAT): 92.6%
- FN: 44 (6.27%)
- FP: 39 (1.93% of non-threats)

**Target for Anti-Poaching:**
- **Recall (THREAT): ≥ 97%** (miss at most 3%)
- **Precision (THREAT): ≥ 85%** (acceptable FP rate)
- **FN: < 20** (reduce by 50%)
- **FP: < 100** (can tolerate 2.5x increase)

**Justification:**
- Missing 3% of threats is better than missing 6%
- Rangers can handle more false alarms if it means catching poachers
- 5-minute cooldown prevents extreme alert spam

---

### 🔧 Edge Deployment Considerations

**Current Threshold (config.py):**
```python
THREAT_THRESHOLD = 0.85
THREAT_CONTEXT_THRESHOLD = 0.75
```

**Recommended After Analysis:**
```python
# After running threshold sensitivity analysis, update to:
THREAT_THRESHOLD = 0.70  # Reduce FN by ~50%
THREAT_CONTEXT_THRESHOLD = 0.75  # Keep as is
```

**Multi-Threshold Strategy (Advanced):**
```python
# Different thresholds per subclass
THREAT_THRESHOLDS = {
    'gunshot': 0.65,        # Most critical - lower threshold
    'chainsaw': 0.70,       # Critical
    'vehicle_engine': 0.75, # Less urgent
    'human_voice': 0.80     # Could be rangers
}
```

---

## Summary

**Your model is GOOD (93.81%) but NOT READY for production deployment.**

**Critical Issue:** 6.27% False Negative rate is too high for anti-poaching.

**Next Steps:**
1. Run threshold sensitivity analysis
2. Identify which threat types are being missed
3. Lower THREAT threshold from 85% → 70%
4. Consider cost-sensitive retraining
5. Target: ≥97% recall for THREAT class

**Bottom Line:** This is a strong baseline, but needs threshold tuning before field deployment.
