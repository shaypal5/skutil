"""Name to regressor class map."""

from typing import Dict

from sklearn import base
from sklearn.utils import all_estimators


def class_name_to_class_regressor_map() -> Dict[str, object]:
    """Returns a dict mapping sklearn regressor class names to class objects.
    """
    name2class = {}
    estimators = all_estimators()
    for name, class_ in estimators:
        if issubclass(class_, base.RegressorMixin):
            name2class[name] = class_
    return name2class


REG_CLS_NAME_2_CLS_MAP = None


def regressor_cls_by_name(name):
    """Get an sklearn regressor class by name.

    Parameters
    ----------
    name : str
        The name of the sklearn regressor class to get.

    Returns
    -------
    object
        The class object of the desired regressor. None if no matching class
        can be found.

    Example
    -------
    >>> regressor_cls_by_name('LinearRegression')
    <class 'sklearn.linear_model._base.LinearRegression'>
    """
    global REG_CLS_NAME_2_CLS_MAP
    try:
        return REG_CLS_NAME_2_CLS_MAP[name]
    except TypeError:  # the map has not been initialized
        REG_CLS_NAME_2_CLS_MAP = class_name_to_class_regressor_map()
        return regressor_cls_by_name(name)
    except KeyError:
        return None
