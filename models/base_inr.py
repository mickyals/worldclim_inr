# define base model
# define training logic
# define validation logic
# define test logic
# define saving logic
# using pytorch lightning

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import os
from tqdm import tqdm
from helpers import get_logger, set_seed, set_device, instantiate_from_config

LOGGER = get_logger(name = "INRModel", log_file="worldclim-dataset.log")



class INRModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        LOGGER.debug("INITIALIZING INR MODEL")
        self.config = config
        self.model_config = self.config['Model']

        self.model = instantiate_from_config(self.model_config['name'])

        loss_config = self.model_config.get('Loss', {})
        loss_type = loss_config.get('type', 'MSELoss')
        loss_params = loss_config.get('params', {})
        loss_class = getattr(nn, loss_type, nn.MSELoss)
        self.loss = loss_class(**loss_params)

        self.save_hyperparameters()



    def forward(self, x):
        LOGGER.debug("FORWARDING INR MODEL")
        return self.model(x)


    def training_step(self, batch, batch_idx):
        _ , loss = self._get_preds_loss(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def configure_optimizers(self):
        optimizer_config = self.config['Optimizer']
        scheduler = self.config.get('Scheduler', None)

        optimizer_config['params'] = self.model.parameters()
        optimizer = instantiate_from_config(optimizer_config)

        if scheduler is not None:
            scheduler['optimizer'] = optimizer
            scheduler = instantiate_from_config(scheduler)

            return optimizer, scheduler

        return optimizer

    def _get_preds_loss(self, batch):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)

        return preds, loss



    # def prepare_training(self):
    #     os.makedirs("../results", exist_ok=True)
    #     os.makedirs("../checkpoints", exist_ok=True)
    #
    #     wandb_config = self.config.get('wandb', None)
    #     if wandb_config is not None and wandb_config.get("enabled", False):
    #         wandb.init(
    #             project=wandb_config['project'],
    #             entity=wandb_config['entity'],
    #             name=wandb_config['name'],
    #             config=self.config,
    #             dir=wandb_config.get('dir', "../results/"),
    #             resume=wandb_config.get('resume', None),
    #             id=wandb_config.get('id', None),
    #             save_code=wandb_config.get('save_code', True),
    #             job_type=wandb_config.get('job_type', None),
    #         )







