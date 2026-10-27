import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
from sklearn.metrics import confusion_matrix

from dataset import get_loaders

device = "cpu"
loaders, classes = get_loaders()
model = timm.create_model("resnet18", pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load("tl_model.pth", map_location=device))
model.eval()

probs, labels = [], []
with torch.no_grad():
    for x, y in loaders["test"]:
        probs.append(torch.softmax(model(x), 1).numpy())
        labels.append(y.numpy())
probs, labels = np.concatenate(probs), np.concatenate(labels)
preds, conf = probs.argmax(1), probs.max(1)

# Confusion matrix
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.xticks(rotation=90); plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)

# Per-class accuracy
per_class = cm.diagonal() / cm.sum(1)
df = pd.DataFrame({"class": classes, "accuracy": per_class.round(3), "n_test": cm.sum(1)})
df.sort_values("accuracy").to_csv("per_class_accuracy.csv", index=False)
print(df.sort_values("accuracy").to_string(index=False))

# Confidence threshold analysis ("Not sure" logic)
print("\nThreshold | answered % | accuracy on answered")
rows = []
for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]:
    m = conf >= t
    acc = (preds[m] == labels[m]).mean() if m.any() else 0
    rows.append({"threshold": t, "answered": m.mean(), "accuracy": acc})
    print(f"{t:>9} | {m.mean()*100:9.1f}% | {acc*100:.1f}%")
json.dump(rows, open("threshold_analysis.json", "w"))