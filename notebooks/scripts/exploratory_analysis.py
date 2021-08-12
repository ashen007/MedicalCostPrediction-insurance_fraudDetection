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

    def typical_value(self, feature):
        """
        mean
        :param feature:
        :return:
        """
        return np.mean(self.data[feature])

    def trimmed_mean(self, feature, method='both', proportion=0.1):
        """
        calculate trimmed average
        :param feature:
        :param method:
        :param proportion:
        :return:
        """
        if method in ['start', 'end', 'both']:
            if isinstance(proportion, float):
                if method == 'start':
                    return np.mean(ss.trim1(self.data[feature], proportiontocut=proportion, tail='left'))
                elif method == 'end':
                    return np.mean(ss.trim1(self.data[feature], proportiontocut=proportion, tail='right'))
                else:
                    return ss.trim_mean(self.data[feature], proportiontocut=proportion)

            elif isinstance(proportion, int):
                lower_bound = np.percentile(self.data[feature], proportion / 100)
                upper_bound = np.percentile(self.data[feature], (100 - proportion) / 100)

                if method == 'start':
                    return np.mean(self.data[self.data[feature] > lower_bound][feature])
                elif method == 'end':
                    return np.mean(self.data[self.data[feature] < upper_bound][feature])
                else:
                    return np.mean(
                        self.data[(self.data[feature] > lower_bound) & (self.data[feature] < upper_bound)][feature])

            else:
                raise ValueError('proportion must be int or a float.')
        else:
            raise ValueError('invalid method.')

    def median(self, feature):
        """
        calculate median
        :param feature:
        :return:
        """
        return np.median(self.data[feature])
