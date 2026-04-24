"""
opt_f1.py
---------
Phase 3 experimentation: retrains the four finalist models optimising for
Macro F1-Score instead of accuracy. Introduces Weighted CrossEntropyLoss
to penalise errors on under-represented classes proportionally to their
frequency in the training set.

Protocol: single phase, 15 epochs, Adam (lr=1e-5), full network unfrozen
from epoch 0. Best checkpoint selected by Macro F1 on the validation set.
"""

import os

import torch
import torch.nn as nn
import torchvision
import wandb
from sklearn.metrics import f1_score

from cnn import CNN, load_data, get_class_weights

# ─── Configuration ────────────────────────────────────────────────────────────
ENTITY   = "202514287-universidad-pontificia-comillas"
PROJECT  = "real-estate-f1"
DATA_DIR = "./dataset"

MODELS = {
    "convnext_base":     torchvision.models.convnext_base,
    "resnext101_32x8d":  torchvision.models.resnext101_32x8d,
    "efficientnet_v2_m": torchvision.models.efficientnet_v2_m,
    "densenet121":       torchvision.models.densenet121,
}

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")


def train_for_f1(model, train_loader, valid_loader, criterion, name):
    """Train for 15 epochs optimising Macro F1. Saves best checkpoint."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    best_f1   = 0.0

    for epoch in range(15):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device).float(), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in valid_loader:
                outputs = model(images.to(device).float())
                y_true.extend(labels.numpy())
                y_pred.extend(outputs.argmax(1).cpu().numpy())

        current_f1 = f1_score(y_true, y_pred, average="macro")
        print(f"[{name}] Epoch {epoch + 1}/15 - Macro F1: {current_f1:.4f}")

        wandb.log({f"{name}/f1_score": current_f1})

        if current_f1 > best_f1:
            best_f1 = current_f1
            model.save(f"{name}_best_f1")


def main():
    train_loader, valid_loader, num_classes = load_data(
        os.path.join(DATA_DIR, "training"),
        os.path.join(DATA_DIR, "validation"),
        16, 224,
    )

    class_weights = get_class_weights(os.path.join(DATA_DIR, "training")).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    for name, builder in MODELS.items():
        wandb.init(entity=ENTITY, project=PROJECT, name=f"F1-Focus-{name}")
        print(f"\nTraining {name} with Macro F1 objective...")

        base  = builder(weights="DEFAULT")
        model = CNN(base, num_classes, device=device)

        for param in model.parameters():
            param.requires_grad = True

        train_for_f1(model, train_loader, valid_loader, criterion, name)
        wandb.finish()


if __name__ == "__main__":
    main()