"""Name to estimator class map."""

import inspect
import functools
from importlib import import_module


_ENSEMBLE = 'sklearn.ensemble'
_GAUS = 'sklearn.gaussian_process'
_LIN = 'sklearn.linear_model'
_SMV = 'sklearn.svm'
_NAIVE_BAYES = 'sklearn.naive_bayes'

_CLS_NAME_TO_PARAM_MAP = {

    # === Sklearn Submodules ===
    # Ensemble
    'AdaBoostClassifier': {
        'module': _ENSEMBLE,
        'names': ['AdaBoostClassifier', 'adaboostclassifier', 'AdaBoost',
                  'adaboost'],
    },
    'BaggingClassifier': {
        'module': _ENSEMBLE,
        'names': ['BaggingClassifier', 'baggingclassifier', 'Bagging',
                  'bagging'],
    },
    'ExtraTreesClassifier': {
        'module': _ENSEMBLE,
        'names': ['ExtraTreesClassifier', 'extratreesclassifier',
                  'ExtraTrees', 'extratrees'],
    },
    'GradientBoostingClassifier': {
        'module': _ENSEMBLE,
        'names': ['GradientBoostingClassifier', 'gradientboostingclassifier',
                  'GradientBoosting', 'gradientboosting'],
    },
    'RandomForestClassifier': {
        'module': _ENSEMBLE,
        'names': ['RandomForestClassifier', 'randomforestclassifier',
                  'RandomForest', 'randomforest'],
    },
    'VotingClassifier': {
        'module': _ENSEMBLE,
        'names': ['VotingClassifier', 'votingclassifier', 'Voting', 'voting'],
    },
    # Gaussian processes
    'GaussianProcessClassifier': {
        'module': _GAUS,
        'names': ['GaussianProcessClassifier', 'GaussianProcessClassifier',
                  'GaussianProcess', 'gaussianprocess'],
    },
    # Linear models
    'LogisticRegression': {
        'module': _LIN,
        'names': ['LogisticRegression', 'logisticregression', 'logreg', 'lr'],
    },
    'LogisticRegressionCV': {
        'module': _LIN,
        'names': ['LogisticRegressionCV', 'logisticregressioncv', 'logregcv',
                  'lrcv'],
    },
    'SGDClassifier': {
        'module': _LIN,
        'names': ['SGDClassifier', 'sgdclassifier', 'sgd'],
    },
    # SVM
    'SVC': {
        'module': _SMV,
        'names': ['SVC', 'svc', 'SVM', 'svm'],
    },
    # Naive Bayes
    'BernoulliNB': {
        'module': _NAIVE_BAYES,
        'names': ['BernoulliNB', 'bernoullinb', 'bnb'],
    },
    'GaussianNB': {
        'module': _NAIVE_BAYES,
        'names': ['GaussianNB', 'gaussiannb', 'gnb'],
    },
    'MultinomialNB': {
        'module': _NAIVE_BAYES,
        'names': ['MultinomialNB', 'multinomialnb', 'mnb'],
    },

    # === Others ===
    'XGBClassifier': {
        'module': 'xgboost.sklearn',
        'names': ['XGBClassifier', 'xgbclassifier', 'xgboost', 'xgb'],
    },
}

_NAME_TO_MODULE_N_CLS_MAP = {}

for cls_name in _CLS_NAME_TO_PARAM_MAP:
    params = _CLS_NAME_TO_PARAM_MAP[cls_name]
    for name in params['names']:
        _NAME_TO_MODULE_N_CLS_MAP[name] = (params['module'], cls_name)


@functools.lru_cache(maxsize=32)
def _get_module(submodule_path):
    return import_module(submodule_path)


def classifier_cls_by_name(name):
    """Get an sklearn classifier class by name.

    Parameters
    ----------
    name : str
        The name of the sklearn classifier class to get. Can also be lower-
        cased. Also, some shorthands are supported (e.g. svm for SVC, logreg
        and lr for LogisticRegression).

    Returns
    -------
    object
        The class object of the desired classifier.

    Example
    -------
    >>> classifier_cls_by_name('LogisticRegression')
    <class 'sklearn.linear_model.logistic.LogisticRegression'>
    >>> classifier_cls_by_name('logisticregression')
    <class 'sklearn.linear_model.logistic.LogisticRegression'>
    >>> classifier_cls_by_name('logreg')
    <class 'sklearn.linear_model.logistic.LogisticRegression'>
    >>> classifier_cls_by_name('lr')
    <class 'sklearn.linear_model.logistic.LogisticRegression'>
    """
    submodule_name, cls_name = _NAME_TO_MODULE_N_CLS_MAP[name]
    submodule = _get_module(submodule_name)
    # submodule = getattr(_sklearn(), submodule_name)
    return getattr(submodule, cls_name)


@functools.lru_cache(maxsize=128)
def _constructor_kwargs_by_class(klass):
    return inspect.getargspec(klass).args


# flake8: noqa: E501
def classifier_by_params(name, **kwargs):
    """Returns a classifier object by the given name and parameters.

    Parameters
    ----------
    name : str
        The name of the sklearn classifier class. Can also be lower-
        cased. Also, some shorthands are supported (e.g. svm for SVC, logreg
        and lr for LogisticRegression).
    **kwargs : Extra keyword arguments
        All keyword arguments supported by the consturctor of the class are
        forwared to it, while the rest are discarded.

    Returns
    -------
    object
        The class object of the desired classifier.

    Example
    -------
    >>> classifier_by_params('LogisticRegression', penalty='l1', ignore='a')
    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)
    """
    klass = classifier_cls_by_name(name)
    allowed_kwargs = _constructor_kwargs_by_class(klass)
    model_kwargs = {
        key: kwargs[key] for key in kwargs
        if key in allowed_kwargs and key != 'self'
    }
    return klass(**model_kwargs)
