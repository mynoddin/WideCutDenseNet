# New version of train.py with resume support from checkpoint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.WideCutDenseNet import WideCutDenseNet
from utils.dataloaders import get_cifar10_dataloaders
import os
import json
from tqdm import tqdm

def train_model(epochs=200, batch_size=128, lr=0.1, device='cuda', resume=True):
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=batch_size)

    model = WideCutDenseNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Logs
    os.makedirs("/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/checkpoints", exist_ok=True)
    os.makedirs("/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/logs", exist_ok=True)
    train_log = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_acc = 0.0
    start_epoch = 0

    if resume and os.path.exists("/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/checkpoints/best_model2.pth"):
        print("✅ Resuming training from checkpoint...")
        model.load_state_dict(torch.load("/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/checkpoints/best_model2.pth"))
        if os.path.exists("/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/logs/train_log2.json"):
            with open("/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/logs/train_log2.json", "r") as f:
                train_log = json.load(f)
                start_epoch = train_log['epoch'][-1]
                best_acc = max(train_log['val_acc'])

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

        scheduler.step()

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()

        val_acc = 100. * correct_val / total_val
        print(f"Validation Accuracy: {val_acc:.2f}%")

        # Logging
        train_log['epoch'].append(epoch + 1)
        train_log['train_loss'].append(train_loss / total)
        train_log['train_acc'].append(100. * correct / total)
        train_log['val_acc'].append(val_acc)
        with open('/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/logs/train_log2.json', 'w') as f:
            json.dump(train_log, f)

        # Save checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/checkpoints/best_model2.pth")
            print(f"💾 New best model saved with accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_model()
