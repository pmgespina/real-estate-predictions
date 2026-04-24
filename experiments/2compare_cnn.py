"""
compare_cnn.py
--------------
Phase 2 experimentation: evaluates high-capacity CNN architectures using a
full two-phase training protocol. Models selected based on Phase 1 screening
results, focusing on modern architectures with strong representational capacity.

Phase 1 — Warmup (5 epochs, lr=1e-3): backbone frozen, only the
classification head is trained.
Phase 2 — Fine-tuning (10 epochs, lr=1e-5): full network unfrozen with a
lower learning rate to preserve ImageNet representations.
"""

import os

import torch
import torch.nn as nn
import torchvision
import wandb

from cnn import CNN, load_data

# ─── Configuration ────────────────────────────────────────────────────────────
ENTITY     = "202514287-universidad-pontificia-comillas"
PROJECT    = "real-estate-transfer-learning"
DATA_DIR   = "./dataset"
BATCH_SIZE = 16
IMAGE_SIZE = 224

MODELS = {
    "efficientnet_v2_m": torchvision.models.efficientnet_v2_m,
    "resnext101_32x8d":  torchvision.models.resnext101_32x8d,
    "convnext_base":     torchvision.models.convnext_base,
}

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")


def main():
    train_loader, valid_loader, num_classes = load_data(
        os.path.join(DATA_DIR, "training"),
        os.path.join(DATA_DIR, "validation"),
        BATCH_SIZE, IMAGE_SIZE,
    )

    for name, builder in MODELS.items():
        print(f"\nTraining: {name}")

        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            name=f"comp-{name}",
            config={
                "model":          name,
                "batch_size":     BATCH_SIZE,
                "lr_warmup":      1e-3,
                "lr_finetune":    1e-5,
                "epochs_warmup":  5,
                "epochs_finetune":10,
            },
        )

        try:
            base      = builder(weights="DEFAULT")
            model     = CNN(base, num_classes, device=device)
            criterion = nn.CrossEntropyLoss()

            # ── Phase 1: Warmup ───────────────────────────────────────────
            print(f"  Phase 1: Warmup (5 epochs, lr=1e-3)")
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            h1 = model.train_model(
                train_loader, valid_loader, optimizer, criterion, epochs=5
            )

            for epoch in range(5):
                run.log({
                    "epoch":          epoch + 1,
                    "train_loss":     h1["train_loss"][epoch],
                    "train_accuracy": h1["train_accuracy"][epoch],
                    "valid_loss":     h1["valid_loss"][epoch],
                    "valid_accuracy": h1["valid_accuracy"][epoch],
                    "phase":          "warmup",
                })

            # ── Phase 2: Fine-tuning ──────────────────────────────────────
            print(f"  Phase 2: Fine-tuning (10 epochs, lr=1e-5)")
            for param in model.parameters():
                param.requires_grad = True

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            h2 = model.train_model(
                train_loader, valid_loader, optimizer, criterion, epochs=10
            )

            for epoch in range(10):
                run.log({
                    "epoch":          5 + epoch + 1,
                    "train_loss":     h2["train_loss"][epoch],
                    "train_accuracy": h2["train_accuracy"][epoch],
                    "valid_loss":     h2["valid_loss"][epoch],
                    "valid_accuracy": h2["valid_accuracy"][epoch],
                    "phase":          "finetune",
                })

            best_valid_acc = max(h2["valid_accuracy"])
            run.summary["best_valid_acc"] = best_valid_acc
            model.save(f"{name}_phase2")

            print(f"  Finished {name}. Best val accuracy: {best_valid_acc:.4f}")

        except Exception as e:
            print(f"  Error training {name}: {e}")

        finally:
            run.finish()


if __name__ == "__main__":
    main()