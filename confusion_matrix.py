
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

from models.WideCutDenseNet import WideCutDenseNet
from utils.dataloaders import get_cifar10_dataloaders

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = WideCutDenseNet(num_classes=10).to(device)
model.load_state_dict(torch.load("/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/checkpoints/best_model2.pth"))
model.eval()

_, test_loader = get_cifar10_dataloaders()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for CIFAR-10")
os.makedirs("plots", exist_ok=True)
plt.savefig("/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/plots/confusion_matrix.png")
plt.show()
