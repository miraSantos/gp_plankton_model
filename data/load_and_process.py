import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.examples.bart import load_bart_od
from pyro.contrib.forecast import ForecastingModel, Forecaster, backtest, eval_crps
from pyro.infer.reparam import LocScaleReparam, StableReparam
from pyro.ops.tensor_utils import periodic_cumsum, periodic_repeat, periodic_features
from pyro.ops.stats import quantile
import matplotlib.pyplot as plt


%matplotlib inline
assert pyro.__version__.startswith('1.8.1')
pyro.set_rng_seed(20200221)



# def load_data(path):
#
#
#
# if __name__ == '__main__':
#     load_data(path)