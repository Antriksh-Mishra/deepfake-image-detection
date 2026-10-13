import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
import logging

from ml.src.models.factory import get_model
from ml.src.datasets.dataset import CLASS_MAPPING

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch for experiment reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DeepfakeTrainer:
    """Trainer class managing PyTorch model training, validation, early stopping, and checkpointing."""

    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 2,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.0001,
        save_dir: str = "ml/models",
        device: Optional[str] = None,
        seed: int = 42
    ):
        set_seed(seed)
        self.model_name = model_name
        self.num_classes = num_classes
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        logger.info(f"Using compute device: {self.device}")

        self.model = get_model(model_name=model_name, num_classes=num_classes, pretrained=True).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = (correct / max(total, 1)) * 100.0
        return epoch_loss, epoch_acc

    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in enumerate(val_loader):
                if isinstance(inputs, tuple):
                    inputs, targets = inputs
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = (correct / max(total, 1)) * 100.0
        return epoch_loss, epoch_acc

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        patience: int = 5
    ) -> Dict[str, Any]:
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": []
        }

        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            self.scheduler.step(val_loss)
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            logger.info(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% ({elapsed:.1f}s)")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(best_val_loss, val_acc)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
                    break

        return history

    def save_checkpoint(self, val_loss: float, val_acc: float) -> None:
        """Saves best_model.pth, model_config.json, and class_mapping.json."""
        model_path = os.path.join(self.save_dir, "best_model.pth")
        torch.save(self.model.state_dict(), model_path)

        config_meta = {
            "model_name": self.model_name,
            "version": "1.0.0",
            "num_classes": self.num_classes,
            "best_val_loss": val_loss,
            "best_val_acc": val_acc,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(os.path.join(self.save_dir, "model_config.json"), "w") as f:
            json.dump(config_meta, f, indent=2)

        with open(os.path.join(self.save_dir, "class_mapping.json"), "w") as f:
            json.dump(CLASS_MAPPING, f, indent=2)

        logger.info(f"Saved model checkpoint to '{model_path}'.")
