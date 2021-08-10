from feature_engine.imputation import CategoricalImputer, MeanMedianImputer, ArbitraryNumberImputer
from feature_engine.encoding import CountFrequencyEncoder, OrdinalEncoder, RareLabelEncoder


class FeatureEngineering:
    """
    feature engineering pipe
    """

    def __init__(self, data):
        """
        initialize pipe-line
        :param data:
        """
        self.data = data

    def impute(self, features=None, method=None):
        """
        impute missing values by using specified method
        :param data:
        :param features:
        :param method:
        :return:
        """
        if method == 'missing':
            imputes = CategoricalImputer(imputation_method='missing',
                                         fill_value='neither',
                                         variables=features)
            self.data = imputes.fit_transform(self.data)

            return imputes.imputer_dict_

        elif method == 'frequency':
            imputes = CategoricalImputer(imputation_method='frequent',
                                         variables=features)
            self.data = imputes.fit_transform(self.data)

            return imputes.imputer_dict_

        elif method == 'median':
            imputes = MeanMedianImputer(imputation_method='median',
                                        variables=features)
            self.data = imputes.fit_transform(self.data)

            return imputes.imputer_dict_

        elif method == 'arbitrary':
            imputes = ArbitraryNumberImputer(arbitrary_number=999999,
                                             variables=features)
            self.data = imputes.fit_transform(self.data)

            return imputes.imputer_dict_

    def encoding(self, features=None, method=None):
        """
        encode categorical features
        :param method:
        :return:
        """
        if method == 'count':
            imputes = CountFrequencyEncoder(encoding_method='count',
                                            variables=features)
            self.data = imputes.fit_transform(self.data)

            return imputes.encoder_dict_

        elif method == 'frequency':
            imputes = CountFrequencyEncoder(encoding_method='frequency',
                                            variables=features)
            self.data = imputes.fit_transform(self.data)

            return imputes.encoder_dict_

        elif method == 'ordinal':
            imputes = OrdinalEncoder(encoding_method='ordered',
                                     variables=features)
            self.data = imputes.fit_transform(self.data)

            return imputes.encoder_dict_

        elif method == 'rare':
            imputes = RareLabelEncoder(tol=0.0005,
                                       variables=features,
                                       ignore_format=True)
            self.data = imputes.fit_transform(self.data)

            return imputes.encoder_dict_
