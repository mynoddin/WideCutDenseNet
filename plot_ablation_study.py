
import json
import os
import matplotlib.pyplot as plt

# Define log files and model labels
log_files = {
    "WideCutDenseNet (Full)": "/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/logs/train_log2.json",
    "No SE-Attention": "/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/logs/train_log_nose2.json",
    "No Cutout": "/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/logs/train_log_nocutout2.json",
    "Residual Only": "/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/logs/train_log_residual2.json"
}

# Load final validation accuracies
accuracies = []
labels = []

for label, filepath in log_files.items():
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            log = json.load(f)
            val_acc = log['val_acc'][-1]
            accuracies.append(val_acc)
            labels.append(label)
    else:
        print(f"Warning: {filepath} not found. Skipping.")

# Plot bar chart
colors = ['darkorange' if "Full" in lbl else 'gray' for lbl in labels]
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, accuracies, color=colors)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2.0, acc + 0.4,
             f"{acc:.2f}%", ha='center', va='bottom', fontsize=9)

plt.title("Ablation Study: Validation Accuracy Comparison")
plt.ylabel("Top-1 Accuracy (%)")
plt.ylim(min(accuracies) - 2, max(accuracies) + 3)
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# Save output
output_path = "/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/logs/ablation_study_accuracy_comparison.png"
plt.savefig(output_path)
plt.show()
