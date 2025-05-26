
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.ResidualOnlyNet import ResidualOnlyNet
from utils.dataloaders import get_cifar10_dataloaders
from tqdm import tqdm
import os
import json

def train_model(epochs=200, batch_size=128, lr=0.1, device='cuda'):
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=batch_size)

    model = ResidualOnlyNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    log = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": []
    }

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    for epoch in range(epochs):
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

        avg_train_loss = train_loss / total
        train_acc = 100. * correct / total

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()

        avg_val_loss = val_loss / total_val
        val_acc = 100. * correct_val / total_val

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        log["train_acc"].append(train_acc)
        log["val_acc"].append(val_acc)
        log["train_loss"].append(avg_train_loss)
        log["val_loss"].append(avg_val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "/content/drive/MyDrive/ResearchforPub/WideCutDenseNet/checkpoints/best_model_residual2.pth")
            print(f"New best model saved with accuracy: {best_acc:.2f}%")

        scheduler.step()

    with open("/content/drive/MyDrive/ResearchforPub/WideCutDenseNet_CIFAR10/logs/train_log_residual2.json", "w") as f:
        json.dump(log, f)

if __name__ == "__main__":
    train_model()
