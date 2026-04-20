import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('report/boundary_cm.json', 'r') as f:
    data = json.load(f)

cm = np.array(data['confusion_matrix'])
# Labels
classes = ['No Boundary', 'Boundary']

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('LID Switch Boundary Confusion Matrix\n(Tolerance = 200ms)')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Normalized percentages for display
# Row-wise normalization for recall/accuracy visualization
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        pct = cm_norm[i, j] * 100
        plt.text(j, i, f"{count}\n({pct:.1f}%)",
                 horizontalalignment="center",
                 color="white" if count > thresh else "black",
                 fontsize=12)

plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()

# Ensure directory exists
import os
os.makedirs('report/figures', exist_ok=True)

plt.savefig('report/figures/boundary_cm_plot.png', dpi=300)
print("Saved figure to report/figures/boundary_cm_plot.png")
