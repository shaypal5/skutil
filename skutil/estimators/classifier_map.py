"""Name to estimator class map."""

from importlib import import_module

from decore import lazy_property


_LIN = 'linear_model'
_SMV = 'svm'
_ENSEMBLE = 'ensemble'
_NAIVE_BAYES = 'naive_bayes'

_CLS_NAME_TO_PARAM_MAP = {
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
    'SVC': {
        'module': _SMV,
        'names': ['SVC', 'svc', 'SVM', 'svm'],
    },
    'RandomForestClassifier': {
        'module': _ENSEMBLE,
        'names': ['RandomForestClassifier', 'randomforestclassifier',
                  'RandomForest', 'randomforest'],
    },
    'BernoulliNB': {
        'module': _NAIVE_BAYES,
        'names': ['BernoulliNB', 'bernoullinb', 'bnb'],
    },
    'GaussianNB': {
        'module': _NAIVE_BAYES,
        'names': ['GaussianNB', 'gaussiannb', 'gnb'],
    },
}

_NAME_TO_MODULE_N_CLS_MAP = {}

for cls_name in _CLS_NAME_TO_PARAM_MAP:
    params = _CLS_NAME_TO_PARAM_MAP[cls_name]
    for name in params['names']:
        _NAME_TO_MODULE_N_CLS_MAP[name] = (params['module'], cls_name)


@lazy_property
def _sklearn():
    return import_module('sklearn')


def classifier_cls_by_name(name):
    submodule_name, cls_name = _NAME_TO_MODULE_N_CLS_MAP[name]
    submodule = getattr(_sklearn(), submodule_name)
    return getattr(submodule, cls_name)
