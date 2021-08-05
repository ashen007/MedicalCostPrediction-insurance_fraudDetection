import numpy as np
import pandas as pd
import seaborn as sns
import missingno as mn

from matplotlib import pyplot as plt


class SanityCheck:
    """
    quickly check data quality before more dep analysis
    """

    def __init__(self, data):
        self.data = data

    def measurement_level(self):
        """
        identify numerical and categorical variable
        :return: dict
        """
        num_fet = self.data.select_dtypes(include=np.number).columns
        cat_fet = self.data.select_dtypes(include=np.object).columns
        discrete = self.data[num_fet].select_dtypes(include=np.int).columns
        continues = self.data[num_fet].select_dtypes(include=np.float).columns

        return {'numerical': num_fet,
                'discrete': discrete,
                'continues': continues,
                'categorical': cat_fet}

    def missing_data(self):
        """
        quantify missing data
        :return: pandas series
        """
        n = self.data.shape
        col_missing = (self.data.isnull().sum() / n[0]).to_dict()
        total_missing = np.sum(self.data.isnull().sum())

        print('total missing count:{} ({}%)'.format(total_missing, total_missing * 100 / np.product(n)))
        print('total rows: {}'.format(n[0]))
        print('total columns: {}'.format(n[1]))
        print('-' * 75)

        return pd.Series(col_missing, name=['feature', 'missing'])

    def cardinality(self):
        """
        determine cardinalities in categorical features
        :return:
        """
        n = self.data.shape[0]
        cat = self.data.select_dtypes(include=np.object)
        card = {}

        for col in cat.columns:
            card[col] = [self.data[col].value_counts(), list(self.data[col].unique())]

        return card

    def magnitude(self):
        """
        compare feature magnitude
        :return:
        """
        mag = {}

        for col in self.data.select_dtypes(include=np.number).columns:
            minimum = np.min(self.data[col])
            maximum = np.max(self.data[col])
            mean = np.mean(self.data[col])
            median = np.median(self.data[col])
            var = np.var(self.data[col])
            std = np.std(self.data[col])
            fifth = np.percentile(self.data[col], 0.05)
            tw_fifth = np.percentile(self.data[col], 0.25)
            sev_fifth = np.percentile(self.data[col], 0.75)
            nin_fifth = np.percentile(self.data[col], 0.95)

            mag[col] = [minimum, maximum, mean, median, var, std, fifth, tw_fifth, sev_fifth, nin_fifth]

        return pd.DataFrame(data=mag.values(),
                            index=mag.keys(),
                            columns=['min', 'max', 'mean', 'median', 'variance', 'std', '5%', '25%', '75%', '95%'])

    def examine_dist(self, feature, hue=None,
                     bins='auto', stat='frequency', kde=False, discrete=False,
                     title=None, path=None):
        """
        examine the shape of the distribution
        :param feature:
        :param hue:
        :param bins:
        :param stat:
        :param kde:
        :param discrete:
        :param title:
        :param path:
        :return:
        """
        plt.figure(figsize=[12, 6], dpi=300)
        sns.histplot(x=feature,
                     hue=hue,
                     data=self.data,
                     bins=bins,
                     stat=stat,
                     kde=kde,
                     discrete=discrete
                     )
        plt.title(title)

        if path is not None:
            plt.savefig(path)

        plt.show()
