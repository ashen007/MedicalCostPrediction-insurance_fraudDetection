import numpy as np
import pandas as pd


class get:
    def __init__(self, path):
        self.__dataframe = pd.read_csv(path)

    def detect_missing_values(self):
        missing_values_by_feature = pd.DataFrame(data=self.__dataframe.isna().sum(axis=1),
                                                 columns=['missing count'])
        missing_values_by_rows = pd.DataFrame(data=self.__dataframe.isna().sum(axis=0),
                                              columns=['missing count'])

        return missing_values_by_feature, missing_values_by_rows

    def modified_zScore(self):
        """:return absolute z score values calculated using modified z-score"""
        numeric_features = self.__dataframe.select_dtypes(include=np.number).columns
        ZScores = pd.DataFrame()

        for feature in numeric_features:
            feat_median = np.median(self.__dataframe[feature])
            MAD = np.median(self.__dataframe[feature].apply(lambda x: np.abs(x - feat_median)))
            z_score = self.__dataframe[feature].apply(lambda x: 0.6745 * (x - feat_median) / MAD)
            ZScores[feature] = z_score

        return np.abs(ZScores)
