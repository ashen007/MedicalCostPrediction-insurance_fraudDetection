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
