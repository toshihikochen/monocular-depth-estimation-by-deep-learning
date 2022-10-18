import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from trainers.base_trainer import BaseTrainer

class EMATrainer(BaseTrainer):
    def __init__(self, model:nn, ema_model, optimizer:optim=None, criterion=None, metrics=None, tensorboard_dir=""):
        super(EMATrainer, self).__init__(model, optimizer, criterion, metrics, tensorboard_dir)
        self.ema_model = ema_model

    def to(self, device):
        super(EMATrainer, self).to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)

    def ema_one_batch(self, image, y_true):
        image = image.to(self.device, dtype=torch.float32, non_blocking=True)
        y_true = y_true.to(self.device, dtype=torch.float32, non_blocking=True)

        y_pred = self.ema_model(image)
        loss = self.criterion(y_true, y_pred)

        self.loss_logger(loss.detach())
        self.metrics(y_true, y_pred)

        return loss.detach()

    def test_one_batch(self, image, model):
        image = image.to(self.device, dtype=torch.float32, non_blocking=True)
        y_pred = model(image)
        return y_pred

    def train_one_epoch(self, epoch, train_loader, val_loader, verbose=0):
        # train phase
        self.set_train_mode(True)
        for i, (image, y_true, _) in enumerate(train_loader, 1):
            self.train_one_batch(image, y_true)
            if i % verbose == 0:
                print(f"Epoch {epoch} Training [{i}/{len(train_loader)}] [{self.timer(i, len(train_loader))}]")
        train_result = self.get_metrics_dict(prefix="train/", step=epoch)
        self.reset_metrics()

        # validation phase
        self.set_train_mode(False)
        for i, (image, y_true, _) in enumerate(val_loader, 1):
            self.val_one_batch(image, y_true)
            if i % verbose == 0:
                print(f"Epoch {epoch} Validating [{i}/{len(val_loader)}] [{self.timer(i, len(val_loader))}]")
        val_result = self.get_metrics_dict(prefix="val/", step=epoch)
        self.reset_metrics()

        # ema phase
        self.set_train_mode(True)
        self.ema_model.update_parameters(self.model)
        for i, (image, y_true, _) in enumerate(val_loader, 1):
            self.ema_one_batch(image, y_true)
            if i % verbose == 0:
                print(f"Epoch {epoch} EMA Validating [{i}/{len(val_loader)}] [{self.timer(i, len(val_loader))}]")
        ema_result = self.get_metrics_dict(prefix="ema/", step=epoch)
        self.reset_metrics()

        # history
        lr = self.optimizer.param_groups[0]["lr"]
        result = {"Epoch": epoch, "lr": lr, **train_result, **val_result, **ema_result}
        self.history = self.history.append(result, ignore_index=True)

        return result

    def test(self, test_loader, model_selection="ema", verbose=0):
        if model_selection.lower() == "model":
            model = self.model
        elif model_selection.lower() == "ema":
            model = self.ema_model
        else:
            raise ValueError("model_selection should be either 'model' or 'ema'")

        self.set_train_mode(False)
        for i, (image, y_true, filenames) in enumerate(test_loader, 1):
            if i % verbose == 0:
                print(f"Testing [{i}/{len(test_loader)}] [{self.timer(i, len(test_loader))}]")
            yield self.test_one_batch(image, model), y_true, filenames

    def save_checkpoint(self, filename):
        print(f"Saving checkpoint to {filename}")
        checkpoints = {
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": self.history
        }
        torch.save(checkpoints, filename)

    def load_checkpoint(self, filename):
        checkpoints = super(EMATrainer, self).load_checkpoint(filename)
        if self.ema_model is not None:
            self.ema_model.load_state_dict(checkpoints["ema_model"])
