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
        :param dataframe dataframe: dataframe want to examine
        """
        self.data = dataframe

    def typical_value(self, feature):
        """
        mean
        :param str feature: column name
        :return:
        """
        return np.mean(self.data[feature])

    def trimmed_mean(self, feature, method='both', proportion=0.1):
        """
        calculate trimmed average
        :param str feature:
        :param str method:
        :param int,float proportion:
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

    def inter_quartile_range(self, feature, range=(25, 75)):
        """
        calculate range between given quartiles percentile
        :param range:
        :param feature:
        :return:
        """
        return ss.iqr(self.data[feature], rng=range)


class distribution:
    """
    examine the distribution metrics and rander visualizing graphs
    """

    def __init__(self, dataframe):
        self.data = dataframe

    @staticmethod
    def render(size, dpi, subplots=False, sub_count=(1, 1)):
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

    def frequency_table(self, feature, bins, type='quantitative', stat='count'):
        """
        calculate frequency statistics
        :param int bins:
        :param str feature:
        :param str type:
        :param str stat:
        :return:
        """
        if type == 'quantitative':
            if stat == 'count':
                return pd.cut(self.data[feature], bins).value_counts()
            elif stat == 'frequency':
                return pd.cut(self.data[feature], bins).value_counts() / self.data.shape[0]
            else:
                raise ValueError()

        elif type == 'qualitative':
            if stat == 'count':
                return self.data[feature].value_counts()
            elif stat == 'frequency':
                return self.data[feature].value_counts() / self.data.shape[0]
            else:
                raise ValueError()

        else:
            raise ValueError()

    def raw_moment(self, feature, k):
        """
        calculate raw moment in the distribution
        :param feature:
        :param k:
        :return:
        """
        return np.sum(self.data[feature] ** k) / self.data.shape[0]

    def central_moment(self, feature, k):
        """
        calculate central moment in the distribution
        :param feature:
        :param k:
        :return:
        """
        return np.sum((self.data[feature] - self.raw_moment(feature, 1)) ** k) / self.data.shape[0]

    def standardized_moment(self, feature, k):
        """
        calculate normalized moments
        :param feature:
        :param k:
        :return:
        """
        return self.central_moment(feature, k) / np.sqrt(self.central_moment(feature, 2)) ** k

    def skewness(self, feature):
        """
        calculate un-bias metric to identify how much where to distribution skew
        :param feature:
        :return:
        """
        return ss.skew(self.data[feature], bias=False)

    def hist(self, feature,
             fig_size=(12, 6), dpi=300,
             sub_plots=False, sub_structure=(1, 1),
             bins='auto', stat='count',
             cumulative=False, kde=False,
             save=False, path='filename', format='png'):
        """
        histogram
        :param format:
        :param cumulative:
        :param dpi:
        :param fig_size:
        :param path:
        :param save:
        :param bins:
        :param kde:
        :param stat:
        :param feature:
        :param sub_plots:
        :param sub_structure:
        :return:
        """
        if not sub_plots:
            if isinstance(feature, str):
                self.render(size=fig_size, dpi=dpi)
                sns.histplot(data=self.data, x=feature,
                             bins=bins, stat=stat,
                             cumulative=cumulative, kde=kde)
                if save:
                    plt.savefig(path, format=format)

                plt.show()

        if sub_plots:
            if isinstance(feature, list):
                fig, axes = self.render(size=fig_size, dpi=dpi, subplots=sub_plots, sub_count=sub_structure)
                axes = axes.ravel()

                for i in range(axes):
                    sns.histplot(data=self.data, x=feature[i],
                                 bins=bins, stat=stat,
                                 cumulative=cumulative, kde=kde,
                                 ax=axes[i])

                if save:
                    plt.savefig(path, format=format)

                plt.show()

            else:
                raise ValueError('for sub plots feature must be a list.')
