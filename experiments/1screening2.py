"""
screening.py
------------
Phase 3 experimentation: initial model screening, evaluates nine backbone architectures with
architecture-specific hyperparameters to identify the most promising
candidates for further experimentation. All runs are logged to W&B.

Protocol: single training phase, 10 epochs, CrossEntropyLoss.
Hyperparameters (lr, batch size, optimizer) are tuned per architecture
based on established best practices for each model family.
"""

import torch
import torch.nn as nn
import torchvision
import wandb

from cnn import CNN, load_data

# ─── Configuration ────────────────────────────────────────────────────────────
ENTITY     = "202514287-universidad-pontificia-comillas"
PROJECT    = "real-estate-screening"
TRAIN_DIR  = "./dataset/training"
VALID_DIR  = "./dataset/validation"
IMAGE_SIZE = 224
EPOCHS     = 10

MODELS_CONFIG = {
    # Round 1 — canonical representatives of each family
    "resnet50": {
        "builder": torchvision.models.resnet50,
        "lr": 1e-4, "batch_size": 32, "optimizer": "Adam", "weight_decay": 0.0,
    },
    "efficientnet_b0": {
        "builder": torchvision.models.efficientnet_b0,
        "lr": 1e-4, "batch_size": 64, "optimizer": "Adam", "weight_decay": 1e-5,
    },
    "mobilenet_v3_large": {
        "builder": torchvision.models.mobilenet_v3_large,
        "lr": 2e-4, "batch_size": 64, "optimizer": "Adam", "weight_decay": 0.0,
    },
    "convnext_tiny": {
        "builder": torchvision.models.convnext_tiny,
        "lr": 1e-4, "batch_size": 32, "optimizer": "AdamW", "weight_decay": 1e-4,
    },
    # Round 2 — extended search including dense connections and attention-based models
    "densenet121": {
        "builder": torchvision.models.densenet121,
        "lr": 1e-4, "batch_size": 32, "optimizer": "Adam", "weight_decay": 0.0,
    },
    "efficientnet_b3": {
        "builder": torchvision.models.efficientnet_b3,
        "lr": 1e-4, "batch_size": 32, "optimizer": "Adam", "weight_decay": 1e-5,
    },
    "convnext_small": {
        "builder": torchvision.models.convnext_small,
        "lr": 1e-4, "batch_size": 8, "optimizer": "AdamW", "weight_decay": 1e-4,
    },
    "vit_b_16": {
        "builder": torchvision.models.vit_b_16,
        "lr": 3e-5, "batch_size": 8, "optimizer": "AdamW", "weight_decay": 0.05,
    },
    "swin_t": {
        "builder": torchvision.models.swin_t,
        "lr": 5e-5, "batch_size": 8, "optimizer": "AdamW", "weight_decay": 0.05,
    },
}

# ─── Training loop ────────────────────────────────────────────────────────────
for model_name, config in MODELS_CONFIG.items():
    print(f"\nScreening: {model_name}")

    lr           = config["lr"]
    batch_size   = config["batch_size"]
    opt_name     = config["optimizer"]
    weight_decay = config["weight_decay"]

    train_loader, valid_loader, num_classes = load_data(
        TRAIN_DIR, VALID_DIR, batch_size=batch_size, img_size=IMAGE_SIZE
    )

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=model_name,
        config={
            "model":         model_name,
            "epochs":        EPOCHS,
            "batch_size":    batch_size,
            "image_size":    IMAGE_SIZE,
            "optimizer":     opt_name,
            "learning_rate": lr,
            "weight_decay":  weight_decay,
            "criterion":     "CrossEntropyLoss",
            "dataset":       "real-estate-predictions",
        },
    )

    try:
        base_model = config["builder"](weights="DEFAULT")
        model      = CNN(base_model, num_classes)

        if opt_name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        criterion = nn.CrossEntropyLoss()

        history = model.train_model(
            train_loader, valid_loader, optimizer, criterion, epochs=EPOCHS
        )

        for epoch in range(EPOCHS):
            run.log({
                "epoch":          epoch + 1,
                "train_loss":     history["train_loss"][epoch],
                "train_accuracy": history["train_accuracy"][epoch],
                "train_f1":       history["train_f1"][epoch],
                "valid_loss":     history["valid_loss"][epoch],
                "valid_accuracy": history["valid_accuracy"][epoch],
                "valid_f1":       history["valid_f1"][epoch],
            })

        run.summary["best_valid_accuracy"] = max(history["valid_accuracy"])
        run.summary["best_valid_f1"]       = max(history["valid_f1"])

        model.save(f"{model_name}-{EPOCHS}epoch")
        print(f"[{model_name}] best_valid_accuracy={max(history['valid_accuracy']):.4f}")

    finally:
        run.finish()