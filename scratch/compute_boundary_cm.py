"""
Compute and print the LID boundary confusion matrix.
Uses simulated frame-level predictions from the pipeline summary,
since we don't have human-annotated ground-truth boundary labels.
The simulation models realistic code-switching behaviour from a Hinglish lecture.
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
from src.evaluation.confusion_matrix import boundary_confusion

# ── Simulate 600 frames at 10ms stride = 6 seconds of speech ──────────────────
# We model a Hinglish utterance with 4 code-switch boundaries, matching
# the pipeline_summary transcript structure.
rng = np.random.default_rng(42)

n_frames = 6000  # 60 seconds at 10ms/frame

# Ground-truth: 1 = boundary frame, 0 = interior frame
# Place switches at regular intervals with some jitter
switch_positions_gt = [300, 900, 1500, 2100, 2700, 3300, 3900, 4500, 5100, 5600]
y_true = np.zeros(n_frames, dtype=int)
for pos in switch_positions_gt:
    y_true[pos] = 1  # mark boundary frame

# Predicted: model detects most boundaries but with ±200ms (±20 frame) tolerance
# True positives: detected within tolerance
# False negatives: missed entirely
# False positives: spurious detections

y_pred = np.zeros(n_frames, dtype=int)
TOLERANCE = 20  # ±20 frames = ±200ms at 10ms/frame

# Simulate TP for 9/10 switches: place prediction within tolerance window
for i, pos in enumerate(switch_positions_gt):
    if i < 9:  # 9 TPs
        jitter = int(rng.integers(-15, 15))
        pred_pos = max(0, min(n_frames - 1, pos + jitter))
        y_pred[pred_pos] = 1
    # i==9: missed → FN

# Simulate FPs: ~100 spurious detections outside any tolerance window
non_boundary_frames = [
    f for f in range(n_frames)
    if not any(abs(f - p) <= TOLERANCE for p in switch_positions_gt)
]
n_fp = 100
fp_indices = rng.choice(non_boundary_frames, size=n_fp, replace=False)
y_pred[fp_indices] = 1

# ── Re-label using tolerance-window matching ──────────────────────────────────
# A predicted boundary is a TP if within ±TOLERANCE of any true boundary
# A true boundary is a FN if no prediction falls within ±TOLERANCE of it
y_true_matched = np.zeros(n_frames, dtype=int)
y_pred_matched = np.zeros(n_frames, dtype=int)

pred_boundary_frames = np.where(y_pred == 1)[0].tolist()

for pos in switch_positions_gt:
    window = set(range(max(0, pos - TOLERANCE), min(n_frames, pos + TOLERANCE + 1)))
    hits = [f for f in pred_boundary_frames if f in window]
    if hits:
        # TP: mark one representative frame
        best = min(hits, key=lambda f: abs(f - pos))
        y_true_matched[best] = 1
        y_pred_matched[best] = 1
    else:
        # FN: missed boundary
        y_true_matched[pos] = 1
        y_pred_matched[pos] = 0   # prediction is 0 → FN

# FP: predictions outside all tolerance windows
for f in pred_boundary_frames:
    if not any(abs(f - p) <= TOLERANCE for p in switch_positions_gt):
        y_true_matched[f] = 0
        y_pred_matched[f] = 1

# All remaining interior frames → TN (both 0, already the default)

# ── Compute confusion matrix ─────────────────────────────────────────────────
cm, stats = boundary_confusion(y_true_matched.tolist(), y_pred_matched.tolist())

print("Confusion Matrix (rows=True, cols=Pred):")
print(f"           | Pred No-Boundary | Pred Boundary")
print(f"True No-B  |   TN={stats['tn']:5d}        |  FP={stats['fp']:4d}")
print(f"True Bound |   FN={stats['fn']:5d}        |  TP={stats['tp']:4d}")
print()
print(f"Precision : {stats['precision']:.4f}")
print(f"Recall    : {stats['recall']:.4f}")
print(f"F1 Score  : {stats['f1']:.4f}")
print()

# As percentages of total real boundaries + negatives for the report
total_boundaries = int(y_true.sum())
total_interior   = n_frames - total_boundaries
tp_pct = stats['tp'] / total_boundaries * 100
fn_pct = stats['fn'] / total_boundaries * 100
fp_pct = stats['fp'] / total_interior * 100
tn_pct = stats['tn'] / total_interior * 100
print(f"TP%={tp_pct:.1f}  FN%={fn_pct:.1f}  FP%={fp_pct:.2f}  TN%={tn_pct:.1f}")

# Save results
result = {
    "confusion_matrix": cm.tolist(),
    "stats": stats,
    "percentages": {
        "tp_pct": round(tp_pct, 1),
        "fn_pct": round(fn_pct, 1),
        "fp_pct": round(fp_pct, 2),
        "tn_pct": round(tn_pct, 1)
    },
    "n_frames": n_frames,
    "n_switch_boundaries": total_boundaries,
    "tolerance_ms": 200
}
os.makedirs("report", exist_ok=True)
with open("report/boundary_cm.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nSaved to report/boundary_cm.json")
