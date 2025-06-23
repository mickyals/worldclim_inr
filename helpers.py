import wandb
import random
import numpy as np
import torch
import inspect
import os
import datetime
import logging


# ==============================================================#
###                        SET SEED                          ###
# ==============================================================#
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================================================#
###                        SET DEVICE                        ###
# ==============================================================#
def set_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================#
###                      COUNT GPUS                          ###
# ==============================================================#

def get_gpu_count():
    return torch.cuda.device_count()


# ==============================================================#
###                      ERROR LOGGING                       ###
# ==============================================================#
def get_logger(name="main", level=logging.DEBUG, log_file=None):
    # create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # set up the log format with class and method names
    log_format = "[%(asctime)s] [%(levelname)s] %(name)s [%(funcName)s] [%(module)s.%(lineno)d]: %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # If a log file path is provided, add a FileHandler to write logs there
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger