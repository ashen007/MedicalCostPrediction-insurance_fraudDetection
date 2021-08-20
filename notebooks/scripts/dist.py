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


class CompareDistribution:
    """
    compare data with standard distributions
    """

    def __init__(self, factory=None, fig_size=(12, 6), dpi=300):
        self._factory = factory
        self.data = self._factory.__getattribute__('data')
        self.fig = fig_size
        self.dpi = dpi

    def qq_compare(self, feature, comp_dist=None):
        """
        compare feature distribution with standards with QQ plot
        :param feature:
        :param comp_dist:
        :return:
        """
        n = self.data[feature].shape[0]
        comp_dist = comp_dist
        bins = np.linspace(0, 100, n)

        self._factory.render(size=self.fig, dpi=self.dpi)
        sns.regplot(x=np.percentile(comp_dist, bins),
                    y=np.percentile(self.data[feature]))
        plt.xlabel('theoretical quartile')
        plt.ylabel('sample quartile')
        plt.show()

    def chi_square_test(self, feature, bins=10):
        """
        calculate
        :param feature:
        :param bins:
        :return:
        """
        test_param = {}
        test_dists = ['norm', 'powernorm', 'uniform', 'cauchy', 'f', 't', 'gamma', 'expon',
                      'chi2', 'beta', 'lognorm', 'powerlognorm', 'weibull_min', 'weibull_max']
        test_stat = []
        percentiles_bins = np.linspace(0, 100, bins)
        thresholds = np.percentile(self.data[feature], percentiles_bins)
        observe_frq, bins = (np.histogram(self.data[feature], thresholds))
        cumulative_obs_frq = np.cumsum(observe_frq)

        for dist in test_dists:
            # Set up distribution and get fitted distribution parameters
            std_dist = getattr(ss, dist)
            param = std_dist.fit(self.data[feature])
            test_param[dist] = param

            # Get expected counts in percentile bins
            # cdf of fitted sistrinution across bins
            cdf_fitted = std_dist.cdf(thresholds, *param)
            exp_frq = []

            for bin in range(len(percentiles_bins) - 1):
                expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
                exp_frq.append(expected_cdf_area)

            exp_frq = np.asarray(exp_frq) * self.data.shape[0]
            cumulative_exp_frq = np.cumsum(exp_frq)
            test_stat.append(((cumulative_obs_frq - cumulative_exp_frq) ** 2) / cumulative_exp_frq)

            return {'test_stat': test_stat,
                    'dist_param': test_param}

    def anderson_darling_test(self, feature, test_dist='norm'):
        """
        test data distribution against normal, exponential, logistic,
        or Gumbel (Extreme Value Type I) distributions.

        H0 = The null hypothesis assumes no difference between the observed
             and theoretical distribution
        Ha = The alternative hypothesis assumes there is a difference between the observed
             and theoretical distribution

        :param feature:
        :param test_dist:
        :return:
        """
        if test_dist == 'norm':
            return ss.anderson(self.data[feature], dist='norm')

        elif test_dist == 'expon':
            return ss.anderson(self.data[feature], dist='expon')

        elif test_dist == 'logistic':
            return ss.anderson(self.data[feature], dist='logistic')

        elif test_dist == 'gumbel':
            return ss.anderson(self.data[feature], dist='gumbel')

        elif test_dist == 'gumbel_r':
            return ss.anderson(self.data[feature], dist='gumbel_r')

    def ks_test(self, feature, test_dist='norm'):
        """
        test data distribution against norm, powernorm, uniform, cauchy, f, t, gamma, expon,
        chi2, beta, lognorm, powerlognorm, weibull_min, weibull_max distributions.

        H0 = The null hypothesis assumes no difference between the observed
             and theoretical distribution
        Ha = The alternative hypothesis assumes there is a difference between the observed
             and theoretical distribution

        :param feature:
        :param test_dist:
        :return:
        """
        theoretical_dists = ['norm', 'powernorm', 'uniform', 'cauchy', 'f', 't', 'gamma', 'expon',
                             'chi2', 'beta', 'lognorm', 'powerlognorm', 'weibull_min', 'weibull_max']

        if test_dist in theoretical_dists:
            return ss.kstest(self.data[feature], test_dist, alternative='two-sided’')
        else:
            raise ValueError()


def normal_dist(loc=0, scale=1, size=100):
    """
    normal random variable
    :param loc:
    :param scale:
    :param size:
    :return:
    """
    return ss.norm.rvs(size=size, loc=loc, scale=scale)


def power_norm_dist(shape=1, loc=0, scale=1, size=100):
    """
    normal random variable
    :param shape:
    :param loc:
    :param scale:
    :param size:
    :return:
    """
    return ss.powernorm.rvs(shape, size=size, loc=loc, scale=scale)


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


def t_dist(shape=1, loc=0, scale=1, size=100):
    """
    t random variable
    :param shape:
    :param loc:
    :param scale:
    :param size:
    :return:
    """
    return ss.t.rvs(shape, loc=loc, scale=scale, size=size)


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


def power_log_norm_dist(shape=(1, 1), loc=0, scale=1, size=100):
    """
    log normal random variable
    :param shape:
    :param loc:
    :param scale:
    :param size:
    :return:
    """
    return ss.powerlognorm.rvs(shape[0], shape[1], size=size, loc=loc, scale=scale)


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
