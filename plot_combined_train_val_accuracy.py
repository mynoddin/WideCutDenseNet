
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

# Setup plot
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axs = axs.flatten()

for idx, (label, filepath) in enumerate(log_files.items()):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            log = json.load(f)
            train_acc = log["train_acc"]
            val_acc = log["val_acc"]
            axs[idx].plot(train_acc, label='Train Accuracy', linestyle='-', color='blue')
            axs[idx].plot(val_acc, label='Val Accuracy', linestyle='--', color='orange')
            axs[idx].set_title(label)
            axs[idx].legend()
            axs[idx].grid(True, linestyle='--', alpha=0.5)
    else:
        axs[idx].set_title(f"{label} (Log Missing)")
        axs[idx].axis('off')

fig.suptitle("Training vs Validation Accuracy per Model", fontsize=14)
fig.text(0.5, 0.04, 'Epoch', ha='center')
fig.text(0.04, 0.5, 'Accuracy (%)', va='center', rotation='vertical')
plt.tight_layout(rect=[0.03, 0.04, 1, 0.96])

# Save plot
output_path = "/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/logs/combined_train_val_accuracy_trends.png"
plt.savefig(output_path)
plt.show()
