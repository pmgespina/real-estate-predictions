"""
screening.py
------------
Phase 1 experimentation: initial model screening, evaluates nine backbone architectures under identical
training conditions to identify the most promising candidates for further
experimentation. All runs are logged to Weights & Biases.

Protocol: single training phase, 10 epochs, Adam (lr=1e-4), CrossEntropyLoss,
batch size 32. No warmup or fine-tuning differentiation at this stage.
"""

import torch
import torch.nn as nn
import torchvision
import wandb

from cnn import CNN, load_data

# ─── Configuration ────────────────────────────────────────────────────────────
ENTITY     = "202514287-universidad-pontificia-comillas"
PROJECT    = "real-estate-transfer-learning"
TRAIN_DIR  = "./dataset/training"
VALID_DIR  = "./dataset/validation"
BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS     = 10

MODELS = {
    # Round 1 — canonical representatives of each family
    "resnet50":           torchvision.models.resnet50,
    "efficientnet_b0":    torchvision.models.efficientnet_b0,
    "mobilenet_v3_large": torchvision.models.mobilenet_v3_large,
    "convnext_tiny":      torchvision.models.convnext_tiny,
    # Round 2 — extended search including dense connections and attention-based models
    "densenet121":        torchvision.models.densenet121,
    "efficientnet_b3":    torchvision.models.efficientnet_b3,
    "convnext_small":     torchvision.models.convnext_small,
    "vit_b_16":           torchvision.models.vit_b_16,
    "swin_t":             torchvision.models.swin_t,
}

# ─── Data ─────────────────────────────────────────────────────────────────────
train_loader, valid_loader, num_classes = load_data(
    TRAIN_DIR, VALID_DIR, batch_size=BATCH_SIZE, img_size=IMAGE_SIZE
)

# ─── Training loop ────────────────────────────────────────────────────────────
for model_name, model_builder in MODELS.items():
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=model_name,
        config={
            "model":         model_name,
            "epochs":        EPOCHS,
            "batch_size":    BATCH_SIZE,
            "image_size":    IMAGE_SIZE,
            "optimizer":     "Adam",
            "learning_rate": 1e-4,
            "criterion":     "CrossEntropyLoss",
            "dataset":       "real-estate-predictions",
        },
    )

    try:
        base_model = model_builder(weights="DEFAULT")
        model      = CNN(base_model, num_classes)
        optimizer  = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion  = nn.CrossEntropyLoss()

        history = model.train_model(
            train_loader, valid_loader, optimizer, criterion, epochs=EPOCHS
        )

        for epoch in range(EPOCHS):
            run.log({
                "epoch":          epoch + 1,
                "train_loss":     history["train_loss"][epoch],
                "train_accuracy": history["train_accuracy"][epoch],
                "valid_loss":     history["valid_loss"][epoch],
                "valid_accuracy": history["valid_accuracy"][epoch],
            })

        best_valid_accuracy = max(history["valid_accuracy"])
        run.summary["best_valid_accuracy"] = best_valid_accuracy
        model.save(f"{model_name}-{EPOCHS}epoch")

        print(f"[{model_name}] best_valid_accuracy={best_valid_accuracy:.4f}")

    finally:
        run.finish()