import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats as ss
from matplotlib import pyplot as plt


class InitDist:
    """
    hold data for distribution analysis
    """

    def __init__(self, dataframe):
        super().__init__()
        self.data = dataframe

    def render(self, size, dpi, subplots=False, sub_count=(1, 1)):
        """
        create graphs
        :return:
        """
        if not subplots:
            return plt.Figure(figsize=size, dpi=dpi)
        else:
            fig, axes = plt.subplots(nrows=sub_count[0],
                                     ncols=sub_count[1],
                                     figsize=size,
                                     dpi=dpi)
            return fig, axes

    def __getattribute__(self, item):
        """
        get properties
        :param item:
        :return:
        """
        if item == 'data':
            return super().__getattribute__('data')

        return super().__getattribute__(item)


def normal_dist(loc=0, scale=1, size=100):
    """
    normal random variable
    :return:
    """
    return ss.norm.rvs(size=size, loc=loc, scale=scale)


def uniform_dist(loc=0, scale=1, size=100):
    """
    uniform random variable
    :param loc:
    :param scale:
    :param size:
    :return:
    """
    return ss.uniform.rvs(size=size, loc=loc, scale=scale)


def cauchy_dist(loc=0, scale=1, size=100):
    """
    cauchy random variable
    :return:
    """
    return ss.cauchy.rvs(size=size, loc=loc, scale=scale)


def f_dist(shape=(1, 1), loc=0, scale=1, size=100):
    """
    f random variable
    :return:
    """
    return ss.f.rvs(shape[0], shape[1], size=size, loc=loc, scale=scale)


def gamma_dist(shape=1, loc=0, scale=1, size=100):
    """
    gamma random variable
    :param shape:
    :param loc:
    :param scale:
    :param size:
    :return:
    """
    return ss.gamma.rvs(shape, size=size, loc=loc, scale=scale)


def exponential_dist(loc=0, scale=1, size=100):
    """
    exponential random variable
    :param loc:
    :param scale:
    :param size:
    :return:
    """
    return ss.expon.rvs(loc=loc, scale=scale, size=size)


def chi_square_dist(shape=1, loc=0, scale=1, size=100):
    """
    chi square random variable
    :param shape:
    :param loc:
    :param scale:
    :param size:
    :return:
    """
    return ss.chi2.rvs(shape, size=size, loc=loc, scale=scale)


def beta_dist(shape=(1, 1), loc=0, scale=1, size=100):
    """
    beta random variable
    :param shape:
    :param loc:
    :param scale:
    :param size:
    :return:
    """
    return ss.beta.rvs(shape[0], shape[1], size=size, loc=loc, scale=scale)


def log_normal_dist(shape=1, loc=0, scale=1, size=100):
    """
    log normal random variable
    :param shape:
    :param loc:
    :param scale:
    :param size:
    :return:
    """
    return ss.lognorm.rvs(shape, size=size, loc=loc, scale=scale)


def weibull_min_dist(shape=1, loc=0, scale=1, size=100):
    """
    weibull minimum random variable
    :param shape:
    :param loc:
    :param scale:
    :param size:
    :return:
    """
    return ss.weibull_min.rvs(shape, size=size, loc=loc, scale=scale)


def weibull_max_dist(shape=1, loc=0, scale=1, size=100):
    """
    weibull maximum random variable
    :param shape:
    :param loc:
    :param scale:
    :param size:
    :return:
    """
    return ss.weibull_max.rvs(shape, size=size, loc=loc, scale=scale)
