
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

# Load final validation losses
losses = []
labels = []

for label, filepath in log_files.items():
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            log = json.load(f)
            val_acc = log['val_acc'][-1]
            losses.append(val_acc)
            labels.append(label)
    else:
        print(f"Warning: {filepath} not found. Skipping.")

# Plot bar chart
colors = ['darkorange' if "Full" in lbl else 'gray' for lbl in labels]
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, losses, color=colors)

for bar, loss in zip(bars, losses):
    plt.text(bar.get_x() + bar.get_width()/2.0, loss + 0.01,
             f"{loss:.4f}", ha='center', va='bottom', fontsize=9)

plt.title("Ablation Study: Final Validation Loss Comparison")
plt.ylabel("Validation Loss")
plt.ylim(min(losses) - 0.05, max(losses) + 0.1)
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# Save plot
output_path = "/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/logs/ablation_study_loss_comparison.png"
plt.savefig(output_path)
plt.show()
