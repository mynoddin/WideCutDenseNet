
import json
import os
import matplotlib.pyplot as plt

# Define log files and labels
log_files = {
    "WideCutDenseNet (Full)": "/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/logs/train_log2.json",
    "No SE-Attention": "/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/logs/train_log_nose2.json",
    "No Cutout": "/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/logs/train_log_nocutout2.json",
    "Residual Only": "/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/logs/train_log_residual2.json"
}

# Load validation accuracy across epochs
plt.figure(figsize=(10, 6))
for label, filepath in log_files.items():
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            log = json.load(f)
            val_acc = log["val_acc"]
            plt.plot(val_acc, label=label)
    else:
        print(f"Warning: {filepath} not found. Skipping.")

plt.title("Validation Accuracy Trends Across Variants")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save plot
output_path = "/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/logs/val_accuracy_trends.png"
plt.savefig(output_path)
plt.show()
