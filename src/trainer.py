"""
Training script for the UNet model.
"""

import torch
from torch.utils.data import DataLoader
import os


class ModelTrainer:


    def __init__(
        self,
        test_dataset,
        model,
        loss_fn,
        optimizer,
        device,
        batch_size=1,
        num_epochs=100,
        num_workers=4
    ):
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.train_dice_scores = []
        self.test_dice_scores = []
        self.set_model_name(loss_fn.__class__.__name__, batch_size, optimizer)
        print(f"Model name set to: {self.model_name}")

    def set_model_name(self, loss_fn, batch_size, optimizer, epoch=None):
        try:
            learning_rate = optimizer.param_groups[0]["lr"]
        except:
            learning_rate = "lr"
            
        base_name = f"unet_{loss_fn}_bs{batch_size}_lr{learning_rate}"
        if epoch is not None:
            self.model_name = f"{base_name}_epoch{epoch}"
        else:
            self.model_name = base_name


    def model_exists(self, file_path="./models"):
        if not hasattr(self, "model_name"):
            print("Model name not set")
            filename = "unet_model"
        else:
            filename = self.model_name

        model_path = os.path.join(file_path, f"{filename}.pth")
        metrics_path = os.path.join(file_path, f"{filename}.json")
        print(model_path)
        print(metrics_path)
        return os.path.exists(model_path) and os.path.exists(metrics_path)

    def load_model(self, file_path="./models"):
        if not hasattr(self, "model_name"):
            print("Model name not set")
            filename = "unet_model"
        else:
            filename = self.model_name

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        path = os.path.join(file_path, f"{filename}.pth")
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
        return self.model