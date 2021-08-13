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

    def winsorized_mean(self, feature, portion=0.1):
        """
        arithmetic mean in the which extreme values are replaced by values closer to the mean
        :param portion:
        :param feature:
        :return:
        """
        temp = self.data.copy()[feature].sort_values()
        temp.iloc[:int(self.data.shape[0] * portion)] = np.median(self.data[feature])
        temp.iloc[-int(self.data.shape[0] * portion):] = np.median(self.data[feature])

        return np.mean(temp)


class Variability:
    """
    dispersion of the data
    """

    def __init__(self, dataframe):
        self.data = dataframe

    def variance(self, feature):
        """
        calculate the variance
        :param feature:
        :return:
        """
        return np.var(self.data[feature])

    def standard_variance(self, feature):
        """
        calculate standard deviation
        :param feature:
        :return:
        """
        return np.std(self.data[feature])

    def trimmed_std(self, feature, method='present', proportion=0.1):
        """
        trimmed standard deviation
        :param proportion:
        :param method:
        :param feature:
        :return:
        """
        if method == 'present':
            if isinstance(proportion, float):
                return np.std(ss.trimboth(self.data[feature], proportion))
            else:
                raise ValueError('when use present proportion must be float between 0 and 1.')
        elif method == 'quartile':
            if isinstance(proportion, int):
                return ss.tstd(self.data[feature],
                               limits=(np.quantile(self.data[feature], proportion),
                                       np.quantile(self.data[feature], 100 - proportion)))
            else:
                raise ValueError('when use quartile proportion must be float between 0 and 100.')
        else:
            raise ValueError('wrong method.')

    def mean_absolute_deviation(self, feature):
        """
        calculate mean absolute deviation
        :param feature:
        :return:
        """
        return np.sum(np.abs(self.data[feature] - np.mean(self.data[feature]))) / self.data.shape[0]

    def mad(self, feature):
        """
        calculate median absolute deviation
        :param feature:
        :return:
        """
        return ss.median_abs_deviation(self.data[feature])
