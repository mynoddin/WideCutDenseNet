
import torch
from models.WideCutDenseNet import WideCutDenseNet
from utils.dataloaders import get_cifar10_dataloaders
import os, time
import numpy as np
from sklearn.metrics import top_k_accuracy_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = WideCutDenseNet(num_classes=10).to(device)
model.load_state_dict(torch.load("/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/checkpoints/best_model2.pth"))
model.eval()

_, test_loader = get_cifar10_dataloaders()

y_true = []
y_pred = []
y_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

# Top-1 and Top-5 accuracy
top1 = top_k_accuracy_score(y_true, y_probs, k=1) * 100
top5 = top_k_accuracy_score(y_true, y_probs, k=5) * 100

# Param count
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Model size
model_size = os.path.getsize("/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/checkpoints/best_model2.pth") / (1024 * 1024)

# Inference speed
dummy_input = torch.randn(1, 3, 32, 32).to(device)
for _ in range(10):
    model(dummy_input)

start = time.time()
for _ in range(100):
    model(dummy_input)
end = time.time()

avg_time = (end - start) / 100
fps = 1.0 / avg_time

# Final summary
print("📊 FINAL EVALUATION METRICS")
print(f"Top-1 Accuracy:        {top1:.2f}%")
print(f"Top-5 Accuracy:        {top5:.2f}%")
print(f"Total Parameters:      {total_params / 1e6:.2f} Million")
print(f"Model Size:            {model_size:.2f} MB")
print(f"Inference Time/Image:  {avg_time*1000:.2f} ms")
print(f"Inference Throughput:  {fps:.2f} FPS")
