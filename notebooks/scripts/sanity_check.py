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

    def date_columns(self, columns, format):
        """
        change data type into datetime
        :param columns:
        :param format:
        :return:
        """
        for col in columns:
            self.data[col] = pd.to_datetime(self.data[col], format=format)

    def change_dtypes(self, features, inplace=True):
        """
        :param features: dict
        :param inplace: make changes on original dataframe
        :return: dataframe
        """
        if inplace:
            self.data = self.data.astype(features)
        else:
            return self.data.astype(features, copy=True)

    def missing_data(self):
        """
        quantify missing data
        :return: pandas series
        """
        n = self.data.shape
        col_missing = (self.data.isnull().sum() * 100 / n[0]).to_dict()
        total_missing = np.sum(self.data.isnull().sum())

        print('total missing count:{} ({}%)'.format(total_missing, total_missing * 100 / np.product(n)))
        print('total rows: {}'.format(n[0]))
        print('total columns: {}'.format(n[1]))
        print('-' * 75)

        return pd.Series(col_missing)

    def render_case_missing_matrix(self, axis=0):
        """
        examine the pattern of missing
        :param axis:
        :return:
        """
        if axis == 0:
            missing_case = (self.data.isnull().sum(axis=0) * 100 / self.data.shape[0]).sort_values()
            plt.figure(figsize=[12, 6], dpi=300)
            sns.barplot(x=missing_case.index, y=missing_case.values, palette='Greys')
            plt.xticks(rotation=90)
            plt.show()

        elif axis == 1:
            missing_feat = self.data.isnull().sum(axis=1)
            plt.figure(figsize=[12, 6], dpi=300)
            sns.histplot(x=missing_feat, discrete=True, color='#000')
            plt.show()

        else:
            raise ValueError

    def render_missing_matrix(self, features):
        """
        missing value matrix
        :return:
        """
        plt.figure(figsize=[12, 6], dpi=300)
        mn.matrix(self.data[features])
        plt.show()

    def cardinality(self, exclude=None):
        """
        determine cardinalities in categorical features
        :return:
        """
        n = self.data.shape[0]
        cat = self.data.select_dtypes(include=np.object).columns
        include = [ele for ele in cat if ele not in exclude]
        card = {}

        for col in include:
            card[col] = [self.data[col].value_counts(), len(list(self.data[col].unique()))]

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

    def __str__(self):
        return 'check measurement level, cardinality and missing values.'
