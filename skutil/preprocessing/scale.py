"""Scaler-related stuff."""

import inspect
import functools
from importlib import import_module

from decore import lazy_property


@lazy_property
def _preprocessing():
    return import_module(submodule_path)


@functools.lru_cache(maxsize=128)
def scaler_cls_by_name(cls_name):
    """Get an sklearn scaler class by name.

    Parameters
    ----------
    cls_name : str
        The name of the sklearn scaler class to get.

    Returns
    -------
    object
        The class object of the desired scaler.

    Example
    -------
    >>> classifier_cls_by_name('RobustScaler')
    <class 'sklearn.preprocessing.RobustScaler'>
    >>> classifier_cls_by_name('Normalizer')
    <class 'sklearn.preprocessing.Normalizer'>
    """
    submodule = _preprocessing()
    return getattr(submodule, cls_name)


@functools.lru_cache(maxsize=128)
def _constructor_kwargs_by_class(klass):
    return inspect.getargspec(klass).args


# flake8: noqa: E501
def scaler_by_params(name, **kwargs):
    """Returns an sklearn scaler object by the given name and parameters.

    Parameters
    ----------
    name : str
        The name of the sklearn scaler class.
    **kwargs : Extra keyword arguments
        All keyword arguments supported by the consturctor of the class are
        forwared to it, while the rest are discarded.

    Returns
    -------
    object
        The class object of the desired scaler.

    Example
    -------
    >>> scaler_by_params('QuantileTransformer', n_quantiles=500)
    QuantileTransformer()
    """
    klass = scaler_cls_by_name(name)
    allowed_kwargs = _constructor_kwargs_by_class(klass)
    constructor_kwargs = {
        key: kwargs[key] for key in kwargs
        if key in allowed_kwargs and key != 'self'
    }
    return klass(**constructor_kwargs)
