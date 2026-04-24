import os
from tempfile import TemporaryDirectory

import torch
import torchvision
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


def get_default_device():
    """Return CUDA if available, MPS for Apple Silicon, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_class_weights(train_dir):
    """Compute inverse-frequency class weights for weighted loss functions.

    Returns:
        Tensor of shape (num_classes,) with weight for each class.
    """
    dataset = torchvision.datasets.ImageFolder(train_dir)
    targets = torch.tensor(dataset.targets)
    class_counts = torch.bincount(targets)
    total_samples = len(targets)
    weights = total_samples / (len(class_counts) * class_counts.float())
    return weights


class CNN(nn.Module):
    """Transfer learning wrapper compatible with any torchvision backbone,
    including CNNs (ResNet, EfficientNet, DenseNet, ConvNeXt) and
    attention-based models (ViT, Swin Transformer).
    """

    def __init__(self, base_model, num_classes, unfreezed_layers=0, device=None):
        """
        Args:
            base_model: Pre-trained torchvision model used as backbone.
            num_classes: Number of output classes.
            unfreezed_layers: Number of backbone layers to unfreeze from the end.
            device: Torch device. Defaults to the best available.
        """
        super().__init__()
        self.num_classes = num_classes
        self.device = device if device is not None else get_default_device()

        # Freeze all backbone parameters
        for param in base_model.parameters():
            param.requires_grad = False

        # Replace the original classification head with Identity so the backbone
        # outputs raw feature vectors. Covers CNNs, ViT and Swin Transformer.
        if hasattr(base_model, "fc"):
            base_model.fc = nn.Identity()
        elif hasattr(base_model, "classifier"):
            base_model.classifier = nn.Identity()
        elif hasattr(base_model, "heads"):
            base_model.heads = nn.Identity()
        elif hasattr(base_model, "head"):
            base_model.head = nn.Identity()

        self.feature_extractor = base_model

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_classes),
        )

        # Initialise LazyLinear before moving to device (required for Apple MPS)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            self.classifier(self.feature_extractor(dummy))

        if unfreezed_layers > 0:
            children = list(self.feature_extractor.children())
            for layer in children[-unfreezed_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.to(self.device)

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))

    def unfreeze_last_blocks(self, blocks_to_unfreeze=1):
        """Progressively unfreeze the last blocks of the backbone for fine-tuning.

        For ResNet/ResNeXt architectures, unfreezes layer4 and optionally layer3.

        Args:
            blocks_to_unfreeze: Number of blocks to unfreeze from the end (1 or 2).
        """
        for param in self.classifier.parameters():
            param.requires_grad = True

        if hasattr(self.feature_extractor, "layer4") and blocks_to_unfreeze >= 1:
            for param in self.feature_extractor.layer4.parameters():
                param.requires_grad = True

        if hasattr(self.feature_extractor, "layer3") and blocks_to_unfreeze >= 2:
            for param in self.feature_extractor.layer3.parameters():
                param.requires_grad = True

    def train_model(self, train_loader, valid_loader, optimizer, criterion, epochs):
        """Train the model, keeping the best checkpoint by validation accuracy.

        Returns:
            history: dict with train/valid loss, accuracy and macro F1 per epoch.
        """
        with TemporaryDirectory() as tmp:
            best_path     = os.path.join(tmp, "best_model.pt")
            best_accuracy = 0.0
            torch.save(self.state_dict(), best_path)

            history = {
                "train_loss": [], "train_accuracy": [], "train_f1": [],
                "valid_loss": [], "valid_accuracy": [], "valid_f1": [],
            }

            for epoch in range(epochs):
                # Training
                self.train()
                train_loss, train_correct = 0.0, 0
                train_preds, train_labels = [], []

                for images, labels in train_loader:
                    images = images.to(self.device).float()
                    labels = labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self(images)
                    loss    = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss    += loss.item()
                    preds          = outputs.argmax(1)
                    train_correct += (preds == labels).sum().item()
                    train_preds.extend(preds.cpu().numpy())
                    train_labels.extend(labels.cpu().numpy())

                train_loss /= len(train_loader)
                train_acc   = train_correct / len(train_loader.dataset)
                train_f1    = f1_score(train_labels, train_preds, average="macro")

                history["train_loss"].append(train_loss)
                history["train_accuracy"].append(train_acc)
                history["train_f1"].append(train_f1)

                print(
                    f"Epoch {epoch + 1}/{epochs}  "
                    f"train_loss: {train_loss:.4f}  "
                    f"train_acc: {train_acc:.4f}  "
                    f"train_f1: {train_f1:.4f}"
                )

                # Validation
                self.eval()
                valid_loss, valid_correct = 0.0, 0
                valid_preds, valid_labels = [], []

                with torch.no_grad():
                    for images, labels in valid_loader:
                        images = images.to(self.device).float()
                        labels = labels.to(self.device)
                        outputs = self(images)
                        loss    = criterion(outputs, labels)

                        valid_loss    += loss.item()
                        preds          = outputs.argmax(1)
                        valid_correct += (preds == labels).sum().item()
                        valid_preds.extend(preds.cpu().numpy())
                        valid_labels.extend(labels.cpu().numpy())

                valid_loss /= len(valid_loader)
                valid_acc   = valid_correct / len(valid_loader.dataset)
                valid_f1    = f1_score(valid_labels, valid_preds, average="macro")

                history["valid_loss"].append(valid_loss)
                history["valid_accuracy"].append(valid_acc)
                history["valid_f1"].append(valid_f1)

                print(
                    f"Epoch {epoch + 1}/{epochs}  "
                    f"valid_loss: {valid_loss:.4f}  "
                    f"valid_acc: {valid_acc:.4f}  "
                    f"valid_f1: {valid_f1:.4f}"
                )

                if valid_acc > best_accuracy:
                    best_accuracy = valid_acc
                    torch.save(self.state_dict(), best_path)

            self.load_state_dict(torch.load(best_path, map_location=self.device))
            return history

    def save(self, filename: str):
        """Save model weights to models/<filename>.pt"""
        os.makedirs("models", exist_ok=True)
        torch.save(self.state_dict(), os.path.join("models", filename))


def load_data(train_dir, valid_dir, batch_size, img_size):
    """Load training and validation datasets with standard ImageNet transforms.

    Returns:
        train_loader, valid_loader, num_classes
    """
    train_transforms = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_data = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = torchvision.datasets.ImageFolder(valid_dir, transform=valid_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, len(train_data.classes)