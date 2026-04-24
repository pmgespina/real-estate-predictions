"""
evaluate.py
-----------
Loads a trained model checkpoint and generates a full evaluation report
on the validation set: classification report (precision, recall, F1 per class)
and confusion matrix saved as a PNG image.
 
Change MODEL_NAME and WEIGHTS_PATH to evaluate a different checkpoint.
"""
 
import os
 
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
from sklearn.metrics import classification_report, confusion_matrix
 
from cnn import CNN, load_data
 
# ─── Configuration
MODEL_NAME   = "resnext101_32x8d"
WEIGHTS_PATH = "resnext101_32x8d_produ.pt.pt"
 
DATA_DIR   = "./dataset"
BATCH_SIZE = 16
IMAGE_SIZE = 224
 
SUPPORTED_MODELS = {
    "resnet50":           torchvision.models.resnet50,
    "densenet121":        torchvision.models.densenet121,
    "efficientnet_b0":    torchvision.models.efficientnet_b0,
    "efficientnet_b3":    torchvision.models.efficientnet_b3,
    "efficientnet_v2_m":  torchvision.models.efficientnet_v2_m,
    "mobilenet_v3_large": torchvision.models.mobilenet_v3_large,
    "convnext_tiny":      torchvision.models.convnext_tiny,
    "convnext_small":     torchvision.models.convnext_small,
    "convnext_base":      torchvision.models.convnext_base,
    "resnext101_32x8d":   torchvision.models.resnext101_32x8d,
    "vit_b_16":           torchvision.models.vit_b_16,
    "swin_t":             torchvision.models.swin_t,
}
 
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
 
 
def main():
    print(f"Device:  {device}")
    print(f"Model:   {MODEL_NAME}")
    print(f"Weights: {WEIGHTS_PATH}\n")
 
    # ── Load data
    _, valid_loader, num_classes = load_data(
        os.path.join(DATA_DIR, "training"),
        os.path.join(DATA_DIR, "validation"),
        BATCH_SIZE, IMAGE_SIZE,
    )
    class_names = valid_loader.dataset.classes
 
    # ── Load model
    if MODEL_NAME not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{MODEL_NAME}' not supported. "
            f"Choose from: {list(SUPPORTED_MODELS.keys())}"
        )
 
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights file not found: {WEIGHTS_PATH}")
 
    base  = SUPPORTED_MODELS[MODEL_NAME](weights=None)
    model = CNN(base, num_classes, device=device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
 
    # ── Inference ─────────────────────────────────────────────────────────────
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images.to(device).float())
            y_true.extend(labels.numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())
 
    # ── Classification report ─────────────────────────────────────────────────
    print("=" * 60)
    print(f"CLASSIFICATION REPORT — {MODEL_NAME.upper()}")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))
 
    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues",
    )
    plt.title(f"Confusion Matrix — {MODEL_NAME}")
    plt.ylabel("Ground Truth")
    plt.xlabel("Prediction")
    plt.tight_layout()
 
    output_path = f"confusion_matrix_{MODEL_NAME}.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")
 
 
if __name__ == "__main__":
    main()