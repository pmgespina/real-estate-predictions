import os
import torch
import torch.nn as nn
import torchvision
from cnn import CNN, load_data, get_class_weights

BATCH_SIZE = 16
EPOCHS_WARMUP = 4
EPOCHS_FINETUNE = 8
LR_WARMUP = 0.001
LR_FINETUNE = 5e-06
UNFREEZE_BLOCKS = 2
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 0

DATA_DIR = "./dataset"
IMAGE_SIZE = 224
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def train_production_model():
    print("Iniciando entrenamiento definitivo para la API...")
    train_loader, valid_loader, num_classes = load_data(
        os.path.join(DATA_DIR, 'training'),
        os.path.join(DATA_DIR, 'validation'),
        BATCH_SIZE, IMAGE_SIZE
    )
    
    weights = get_class_weights(os.path.join(DATA_DIR, 'training')).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    base = torchvision.models.resnext101_32x8d(weights="DEFAULT")
    model = CNN(base, num_classes, device=device)

    # FASE 1: Warmup (Feature Extraction)
    opt_warmup = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LR_WARMUP,
        weight_decay=WEIGHT_DECAY
    )
    model.train_model(train_loader, valid_loader, opt_warmup, criterion, epochs=EPOCHS_WARMUP)

    # FASE 2: Progressive Fine-Tuning
    model.unfreeze_last_blocks(blocks_to_unfreeze=UNFREEZE_BLOCKS)
    
    opt_finetune = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LR_FINETUNE,
        weight_decay=WEIGHT_DECAY
    )
    
    model.train_model(train_loader, valid_loader, opt_finetune, criterion, epochs=EPOCHS_FINETUNE)
    
    model.save("resnext101_32x8d_prod.pt")

if __name__ == "__main__":
    train_production_model()