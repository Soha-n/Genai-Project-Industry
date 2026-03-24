"""
Training script for the bearing fault CNN classifier.

Usage:
    python -m src.models.train                          # uses default config
    python -m src.models.train --config configs/config.yaml
"""

import argparse
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm

from src.models.cnn_classifier import BearingFaultCNN


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_transforms(image_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for images, labels in tqdm(loader, desc="  Val", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def run(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    mcfg = cfg["model"]
    scfg = cfg["spectrogram"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    image_size = scfg["image_size"]
    train_tf, val_tf = get_transforms(image_size)

    # Dataset — ImageFolder expects data/spectrograms/{class}/*.png
    data_dir = cfg["paths"]["spectrograms"]
    full_dataset = datasets.ImageFolder(data_dir, transform=train_tf)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Total samples: {len(full_dataset)}")

    # Split
    n = len(full_dataset)
    n_train = int(n * mcfg["train_split"])
    n_val = int(n * mcfg["val_split"])
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # Override transform for val/test splits
    val_ds.dataset = datasets.ImageFolder(data_dir, transform=val_tf)
    # Note: random_split indices still apply correctly

    train_loader = DataLoader(train_ds, batch_size=mcfg["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=mcfg["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=mcfg["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = BearingFaultCNN(num_classes=num_classes, pretrained=mcfg["pretrained"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=mcfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    best_val_acc = 0.0
    model_save_path = Path(cfg["paths"]["cnn_model"])
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, mcfg["epochs"] + 1):
        print(f"\nEpoch {epoch}/{mcfg['epochs']}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "num_classes": num_classes,
                "val_acc": val_acc,
                "epoch": epoch,
            }, str(model_save_path))
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")

    # Test evaluation
    print("\n--- Test Evaluation ---")
    checkpoint = torch.load(str(model_save_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f}")

    # Per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.config)
