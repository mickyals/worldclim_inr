# define base model
# define training logic
# define validation logic
# define test logic
# define saving logic
# using pytorch lightning

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from helpers import get_logger, set_seed, set_device, instantiate_from_config

LOGGER = get_logger(name = "BaseINR", log_file="worldclim-dataset.log")

class BaseINR(pl.LightningModule):
    def __init__(self, config, device=None):
        super().__init__()

        self.config = config
        self.net = instantiate_from_config(self.config.model)


    def forward(self, x):
        return self.net(x.to(self.device))

    def configure_optimizers(self):
        optimizer_config = self.config.Optimizer
        optim_class = getattr(torch.optim, optimizer_config.name, torch.optim.Adam)

        optimizer = optim_class(self.parameters(), **optimizer_config.params)

        scheduler_config = self.config.Scheduler
        if scheduler_config and scheduler_config.name:
            scheduler = getattr(torch.optim.lr_scheduler, scheduler_config.name, torch.optim.lr_scheduler.CosineAnnealingLR)
            scheduler = scheduler(optimizer, **scheduler_config.params)

            interval = scheduler_config.get("interval", "epoch")
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": interval}}

        return optimizer



    def training_step(self, batch, batch_idx):
        inputs = batch['input'].to(self.device)
        targets = batch['target'].to(self.device)

        predictions = self.foward(inputs)

        loss = self.loss(predictions, targets) # need to define loss functions/scripts
        self.log("train_loss", loss, on_step=True, on_epoch=True) #consider incorportation within loss functions/scripts
        return loss


