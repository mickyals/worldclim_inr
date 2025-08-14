import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import lightning as L
import os
from tqdm import tqdm
from helpers import get_logger, set_seed, set_device, instantiate_from_config
import datashader as ds
import datashader.transfer_functions as tf
import colorcet as cc
import pandas as pd



LOGGER = get_logger(name = "TropoModel", log_file="era5-tropo-dataset.log")


class TropoModel(L.LightningModule):
    def __init__(self, config):
        super(TropoModel, self).__init__()
        self.config = config

        self.model = instantiate_from_config(config['Model'])

        loss_config = self.config.get('Loss', {})
        loss_type = loss_config.get('type', 'MSELoss')
        loss_params = loss_config.get('params', {})
        loss_class = getattr(nn, loss_type, nn.MSELoss)
        self.loss = loss_class(**loss_params)

        self.save_hyperparameters()
        self.device = set_device()

    def forward(self, x):
        return self.model(x.to(self.device))

    def training_step(self, batch, batch_idx):
        X, Y = batch
        X = X.to(self.device)
        Y = Y.to(self.device)
        preds = self(X)
        loss = self.loss(preds, Y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        fixed_lon = 0
        fixed_lat = 0

        # ===== Helper plotting function =====
        def plot_datashader(x, y, vals, cmap=cc.CET_D1A, fname="plot.png"):
            df = pd.DataFrame({
                "x": x.ravel(),
                "y": y.ravel(),
                "val": vals.ravel()
            })
            cvs = ds.Canvas(
                plot_width=800, plot_height=400,
                x_range=(df["x"].min(), df["x"].max()),
                y_range=(df["y"].min(), df["y"].max())
            )
            agg = cvs.points(df, "x", "y", ds.mean("val"))
            img = tf.shade(agg, cmap=cmap, how="eq_hist")
            img.to_pil().save(fname)

        # ===== lat–plev slice (fixed lon) =====
        lat_vals = np.linspace(-90, 90, 721)
        plev_vals = np.linspace(1000, 1, 100)
        LAT, PLEV = np.meshgrid(lat_vals, plev_vals)
        LON_fixed = np.full_like(LAT, fixed_lon)

        lat_radian = np.deg2rad(LAT)
        lon_radian = np.deg2rad(LON_fixed)
        plev_norm = 2 * (PLEV - 1) / (1000 - 1) - 1

        X_lat = np.stack([lon_radian, lat_radian, plev_norm], axis=-1)
        X_lat_t = torch.tensor(X_lat, dtype=torch.float32).view(-1, 3).to(self.device)

        with torch.no_grad():
            pred_lat_plev = self(X_lat_t).cpu().numpy().reshape(LAT.shape + (2,))
            lat_temp = pred_lat_plev[..., 0]
            lat_z = pred_lat_plev[..., 1]

        plot_datashader(LAT, PLEV, lat_temp, fname=f"epoch_{self.current_epoch}_lat_temp.png")
        plot_datashader(LAT, PLEV, lat_z, fname=f"epoch_{self.current_epoch}_lat_z.png")

        # ===== lon–plev slice (fixed lat) =====
        lon_vals = np.linspace(-180, 180, 1440)
        LON, PLEV2 = np.meshgrid(lon_vals, plev_vals)
        LAT_fixed = np.full_like(LON, fixed_lat)

        lat_radian = np.deg2rad(LAT_fixed)
        lon_radian = np.deg2rad(LON)
        plev_norm = 2 * (PLEV2 - 1) / (1000 - 1) - 1

        X_lon = np.stack([lon_radian, lat_radian, plev_norm], axis=-1)
        X_lon_t = torch.tensor(X_lon, dtype=torch.float32).view(-1, 3).to(self.device)

        with torch.no_grad():
            pred_lon_plev = self(X_lon_t).cpu().numpy().reshape(LON.shape + (2,))
            lon_temp = pred_lon_plev[..., 0]
            lon_z = pred_lon_plev[..., 1]

        plot_datashader(LON, PLEV2, lon_temp, fname=f"epoch_{self.current_epoch}_lon_temp.png")
        plot_datashader(LON, PLEV2, lon_z, fname=f"epoch_{self.current_epoch}_lon_z.png")

    def configure_optimizers(self):
        optim_cfg = self.config.optimizer_config
        optim_class = getattr(torch.optim, optim_cfg.type, torch.optim.Adam)
        optimizer = optim_class(self.parameters(), lr=optim_cfg.learning_rate, **optim_cfg.params)

        sched_cfg = self.config.scheduler
        if sched_cfg and sched_cfg.type:
            scheduler_class = getattr(torch.optim.lr_scheduler, sched_cfg.type,
                                      torch.optim.lr_scheduler.CosineAnnealingLR)
            scheduler = scheduler_class(optimizer, **sched_cfg.params)
            interval = sched_cfg.params.get("interval", "epoch")
            return [optimizer], [{"scheduler": scheduler, "interval": interval}]
        return optimizer