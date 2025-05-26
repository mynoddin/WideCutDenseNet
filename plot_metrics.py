
import json
import matplotlib.pyplot as plt
import os

log_path = "/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/logs/train_log.json"
if not os.path.exists(log_path):
    raise FileNotFoundError(f"Log file not found at: {log_path}")

with open(log_path, "r") as f:
    log = json.load(f)

epochs = log["epoch"]
train_loss = log["train_loss"]
train_acc = log["train_acc"]
val_acc = log["val_acc"]

# Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/plots/accuracy_curve.png")
plt.show()

# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/plots/loss_curve.png")
plt.show()
