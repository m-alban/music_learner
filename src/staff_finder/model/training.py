from src.staff_finder.model import StaffFasterRCNN

import torch
from torch import optim
from torchmetrics.detection.map import MeanAveragePrecision
import pytorch_lightning as pl
from typing import Dict

class MuscimaLightning(pl.LightningModule):
    """Trains Muscima staff finder with AdamW optimizer.
    """
    def __init__(self, lr: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = StaffFasterRCNN()
        map_metric = MeanAveragePrecision
        self.train_metrics = map_metric()
        self.test_metrics = map_metric()
        self.val_metrics = map_metric()

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(self.parameters(), lr = self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = [image.to(self.device) for image in images]
        targets = [{k:v for k, v in t.items()} for t in targets]
        targets = [
            {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} 
            for t in targets]
        #with torch.cuda.amp.autocast(enabled = scalar is not None):
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        return {'loss': loss, 'log': loss_dict}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = [image.to(self.device) for image in images]
        targets = [
            {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} 
            for t in targets]
        predictions = self.model(images)
        predictions = [
            {k:v for k,v in pred.items() if k in ['boxes', 'scores', 'labels']}
            for pred in predictions
        ]
        map_dict = self.val_metrics(predictions, targets)
        map_dict = {f'val_{k}': v for k, v in map_dict.items() if 'map' in k and 'class' not in k}
        self.log('val_map', map_dict['val_map'])
        self.log_dict(map_dict)
