import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import quad
from sklearn.neighbors import KernelDensity

class BiasedSampling:

    def __init__(self, params=None):
        pass
    def fit(self, x):
        pass

    def score(self, x):
        pass

class multinomialSampling(BiasedSampling):

    def __init__(self, params=None):
        pass

    def fit(self, x):
        # TODO: document code
        mean = np.mean(x, axis=0)
        cov = np.cov(x, rowvar=0)
        self.var = multivariate_normal(mean=mean, cov=np.diag(np.diag(cov)))
        probs = self.var.pdf(x)


        return probs

    def score(self, x):
        return self.var.pdf(x)

class KDESampling(BiasedSampling):
    def __init__(self, params=None):
        if params is None:
            self.bandwidth = 0.2
            self.kernel = "epanechnikov"
        else:
            self.bandwidth = params['bandwidth']
            self.kernel = params['kernel']
        super().__init__(params)

    def fit(self, x):
        # TODO: Document code
        self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(x)
        scores = np.exp(self.kde.score_samples(x))

        return scores

    def score(self, x):
        return np.exp(self.kde.score_samples(x))

def calc_probability_distribution(x, y, pdf_method=None):
    '''

    :param x:
    :param y:
    :param pdf_method:
    :return:
    '''
    prob_dist = np.zeros(shape=(len(y),))
    indici_positives = y==1
    x_pos = x[indici_positives, :]
    prob_dist_pos = np.zeros(shape=(len(x_pos),))
    biasedSampler = None
    if pdf_method is None: # uniform distribution
        return None
    if pdf_method == "multinomial":
        biasedSampler = multinomialSampling()
        prob_dist_pos = biasedSampler.fit(x_pos)
    if pdf_method == "kde":
        params = {
            ''
        }
        biasedSampler = KDESampling()
        prob_dist_pos = biasedSampler.fit(x_pos)

    prob_dist[indici_positives] = prob_dist_pos

    return prob_dist

def biasedSampling(x, y, cols, biased_sampling_method, biased_sampling_params):
    '''

    :param x:
    :return:
    '''

    # 1. Select positive samples with given biased columns
    x_positive = x[y==1, :]
    x_negative = x[y==0, :]
    x_positive_biased = x_positive[:, cols]

    biasedSampler = None
    # 2. create biasedSampler
    if biased_sampling_method == "multinomial":
        biasedSampler = multinomialSampling(biased_sampling_params)
    elif biased_sampling_method == 'kde':
        biasedSampler = KDESampling(biased_sampling_params)

    # 2. Generate labels using sampler
    s_pos = biasedSampler.fit(x_positive_biased)
    s_neg = np.zeros(shape=(len(x_negative),), dtype=int)


    return np.concatenate((s_pos, s_neg))



