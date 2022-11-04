import math
import pandas as pd
import torch
import gpytorch
import os
import sys
import numpy as np
import matplotlib.dates as mdates  # v 3.3.2

sys.path.append(os.getcwd())

from tqdm import tqdm
import yaml
import models.spectralGP_model
import matplotlib.pyplot as plt
import argparse

from matplotlib.dates import YearLocator
from PIL import Image
import wandb  # library for tracking and visualization
from evaluation.forecasting_metrics import *



def compute_metrics(metrics, actual, predicted):

    metrics_list = [[] for _ in range(len(metrics))]  # list of lists to store error metric results

    for j in range(len(metrics)):
        metrics_list[j].append(metrics[j](actual,predicted))

    df_metrics = pd.DataFrame({"metrics":metrics,"metrics_values":metrics_list})
    wandb.log({"table":df_metrics})
    return metrics_list