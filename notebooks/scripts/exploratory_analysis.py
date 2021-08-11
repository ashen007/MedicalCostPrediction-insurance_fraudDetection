import numpy as np
import scipy.stats as ss
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt


class CentralTendency:
    """
    estimate of location
    """

    def __init__(self, dataframe):
        """
        initialize dataframe
        :param dataframe:
        """
        self.data = dataframe

    def typical_value(self,feature):
        """
        mean
        :param feature:
        :return:
        """
        return np.mean(self.data[feature])
