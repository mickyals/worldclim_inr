# define base model
# define training logic
# define validation logic
# define test logic
# define saving logic
# using pytorch lightning

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os
from helpers import get_logger, set_seed, set_device, instantiate_from_config

LOGGER = get_logger(name = "TrainINR", log_file="worldclim-dataset.log")



class TrainINR:
    def __init__(self, config):
        self.config = config

        self.model = instantiate_from_config(self.config['Model'])


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


    def prepare_training(self):
        os.makedirs("../results", exist_ok=True)
        os.makedirs("../checkpoints", exist_ok=True)

        wandb_config = self.config.get('wandb', None)
        if wandb_config is not None and wandb_config.get("enabled", False):
            wandb.init(
                project=wandb_config['project'],
                entity=wandb_config['entity'],
                name=wandb_config['name'],
                config=self.config,
                dir=wandb_config.get('dir', "../results/"),
                resume=wandb_config.get('resume', None),
                id=wandb_config.get('id', None),
                save_code=wandb_config.get('save_code', True),
                job_type=wandb_config.get('job_type', None),
            )

    def configure_loss(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def train_pipeline(self):
        pass

