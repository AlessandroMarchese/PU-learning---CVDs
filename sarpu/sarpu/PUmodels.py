import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class BasePU:
    @staticmethod
    def _make_propensity_weighted_data(x, s, e, sample_weight=None):
        e[e == 0] = 1e-6
        weights_pos = s / e
        weights_neg = (1 - s) + s * (1 - 1 / e)
        if sample_weight is not None:
            weights_pos = sample_weight * weights_pos
            weights_neg = sample_weight * weights_neg

        Xp = np.concatenate([x, x])
        Yp = np.concatenate([np.ones_like(s), np.zeros_like(s)])
        Wp = np.concatenate([weights_pos, weights_neg])
        return Xp, Yp, Wp


class LogisticRegressionPU(LogisticRegression, BasePU):
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):
        LogisticRegression.__init__(self, penalty=penalty, dual=dual, tol=tol, C=C,
                                    fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                    class_weight=class_weight, random_state=random_state,
                                    solver=solver, max_iter=max_iter, multi_class=multi_class,
                                    verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, x, s, e=None, sample_weight=None):
        if e is None:
            super().fit(x, s, sample_weight)
        else:
            Xp, Yp, Wp = self._make_propensity_weighted_data(x, s, e, sample_weight)
            super().fit(Xp, Yp, Wp)


class SVCPU(SVC, BasePU):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True,
                 tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=100,
                 decision_function_shape='ovr', break_ties=False, random_state=None):
        SVC.__init__(self, C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
                     probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight,
                     verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape,
                     break_ties=break_ties, random_state=random_state)

    def fit(self, x, s, e=None, sample_weight=None):
        if e is None:
            super().fit(x, s, sample_weight)
        else:
            Xp, Yp, Wp = self._make_propensity_weighted_data(x, s, e, sample_weight)
            super().fit(Xp, Yp, Wp)


class RFPU(RandomForestClassifier, BasePU):
    def __init__(self,
                 n_estimators=100,
                 *,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None,
                 ):
        RandomForestClassifier.__init__(self,
                                        n_estimators=100,
                                        criterion="gini",
                                        max_depth=None,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        min_weight_fraction_leaf=0.0,
                                        max_features="auto",
                                        max_leaf_nodes=None,
                                        min_impurity_decrease=0.0,
                                        bootstrap=True,
                                        oob_score=False,
                                        n_jobs=None,
                                        random_state=None,
                                        verbose=0,
                                        warm_start=False,
                                        class_weight=None,
                                        ccp_alpha=0.0,
                                        max_samples=None,
                                        )

    def fit(self, x, s, e=None, sample_weight=None):
        if e is None:
            sample_weight[sample_weight == np.nan] = 1e-6
            super().fit(x, s, sample_weight)
        else:
            Xp, Yp, Wp = self._make_propensity_weighted_data(x, s, e, sample_weight)
            super().fit(Xp, Yp, Wp)

