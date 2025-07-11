"""Scaler-related stuff."""

import functools
import inspect
from importlib import import_module

from decore import lazy_property


@lazy_property
def _preprocessing():
    return import_module("sklearn.preprocessing")


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
    >>> scaler_cls_by_name('RobustScaler')
    <class 'sklearn.preprocessing._data.RobustScaler'>
    >>> scaler_cls_by_name('Normalizer')
    <class 'sklearn.preprocessing._data.Normalizer'>

    """
    submodule = _preprocessing()
    return getattr(submodule, cls_name)


@functools.lru_cache(maxsize=128)
def _constructor_kwargs_by_class(klass):
    sig = inspect.signature(klass)
    return list(sig.parameters.keys())


# flake8: noqa: E501
def scaler_by_params(name, **kwargs):
    """Get an sklearn scaler object by the given name and parameters.

    Parameters
    ----------
    name : str
        The name of the sklearn scaler class.
    **kwargs : Extra keyword arguments
        All keyword arguments supported by the constructor of the class are
        forward to it, while the rest are discarded.

    Returns
    -------
    object
        The class object of the desired scaler.

    Example
    -------
    >>> scaler_by_params('QuantileTransformer', n_quantiles=500)  # doctest: +SKIP
    QuantileTransformer(copy=True, ignore_implicit_zeros=False, n_quantiles=500,
              output_distribution='uniform', random_state=None,
              subsample=100000)

    """
    klass = scaler_cls_by_name(name)
    allowed_kwargs = _constructor_kwargs_by_class(klass)
    constructor_kwargs = {
        key: kwargs[key]
        for key in kwargs
        if key in allowed_kwargs and key != "self"
    }
    return klass(**constructor_kwargs)
