"""
tuning_resnet.py
--------------------
Optimización Bayesiana de hiperparámetros para ResNeXt101_32x8d.
Fase 1: Feature Extraction (Warmup) -> Fase 2: Progressive Fine-Tuning
Incluye corrección de desbalanceo por Class Weights.
"""

import argparse
import os
import torch
import torch.nn as nn
import torchvision
import wandb
from sklearn.metrics import f1_score
from cnn import CNN, load_data, get_class_weights

ENTITY     = "202514287-universidad-pontificia-comillas"
PROJECT    = "real-estate-f1"
DATA_DIR   = "./dataset"
IMAGE_SIZE = 224

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# Definimos el espacio de búsqueda
SWEEP_CONFIG = {
    "name":   "resnext101-progressive-sweep",
    "method": "bayes",
    "metric": {"name": "best_macro_f1", "goal": "maximize"},
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 3,
        "eta": 2
    },
    "parameters": {
        "lr_warmup":       {"values": [1e-3, 5e-4]},
        "lr_finetune":     {"values": [1e-5, 5e-6]},
        "epochs_warmup":   {"values": [3, 4]},
        "epochs_finetune": {"values": [6, 8]},
        "batch_size":      {"values": [8, 16]},
        "weight_decay":    {"values": [0.0, 1e-4]},
        "label_smoothing": {"values": [0.0, 0.1]},
        "unfreeze_blocks": {"values": [1, 2]}
    }
}

def _run_epoch(model, loader, optimizer, criterion, device, phase="train"):
    """Ejecuta una época y devuelve loss y F1 Macro."""
    if phase == "train": model.train()
    else: model.eval()

    total_loss, all_true, all_pred = 0.0, [], []
    
    with torch.set_grad_enabled(phase == "train"):
        for images, labels in loader:
            images, labels = images.to(device).float(), labels.to(device)
            if phase == "train": optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            if phase == "train":
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(outputs.argmax(1).cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_true, all_pred, average='macro')
    return avg_loss, f1

def train_sweep():
    with wandb.init() as run:
        cfg = wandb.config
        train_loader, valid_loader, num_classes = load_data(
            os.path.join(DATA_DIR, 'training'),
            os.path.join(DATA_DIR, 'validation'),
            cfg.batch_size, IMAGE_SIZE
        )
        
        # Para evitar el sesgo por desbalanceo de clases
        weights = get_class_weights(os.path.join(DATA_DIR, 'training')).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)

        # Construcción del modelo base
        base = torchvision.models.resnext101_32x8d(weights="DEFAULT")
        model = CNN(base, num_classes, device=device)

        best_f1 = 0.0

        # FASE 1: WARMUP (Feature Extraction)
        # Solo se entrena la capa de clasificación final
        opt_warmup = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=cfg.lr_warmup
        )
        
        for e in range(cfg.epochs_warmup):
            t_loss, t_f1 = _run_epoch(model, train_loader, opt_warmup, criterion, device, "train")
            v_loss, v_f1 = _run_epoch(model, valid_loader, opt_warmup, criterion, device, "valid")
            wandb.log({"macro_f1": v_f1, "phase": 1, "epoch_global": e+1})

        # FASE 2: PROGRESSIVE FINE-TUNING
        # Descongelamos los bloques definidos por el Sweep
        model.unfreeze_last_blocks(blocks_to_unfreeze=cfg.unfreeze_blocks)
        
        # Nuevo optimizador con TODOS los parámetros descongelados, LR muy bajo
        opt_finetune = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=cfg.lr_finetune, 
            weight_decay=cfg.weight_decay
        )
        
        for e in range(cfg.epochs_finetune):
            t_loss, t_f1 = _run_epoch(model, train_loader, opt_finetune, criterion, device, "train")
            v_loss, v_f1 = _run_epoch(model, valid_loader, opt_finetune, criterion, device, "valid")
            
            if v_f1 > best_f1: 
                best_f1 = v_f1
                
            wandb.log({"macro_f1": v_f1, "phase": 2, "epoch_global": cfg.epochs_warmup + e + 1})

        wandb.summary["best_macro_f1"] = best_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["init", "agent"], required=True)
    parser.add_argument("--sweep_id", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "init":
        s_id = wandb.sweep(SWEEP_CONFIG, entity=ENTITY, project=PROJECT)
        print(f"ID DEL SWEEP GENERADO: {s_id}")
        print(f"python tuning_resnet.py --mode agent --sweep_id {s_id}")
    elif args.mode == "agent":
        if not args.sweep_id:
            raise ValueError("Debes proporcionar un --sweep_id para el modo agent.")
        wandb.agent(f"{ENTITY}/{PROJECT}/{args.sweep_id}", function=train_sweep, count=5)

if __name__ == "__main__":
    main()