"""scikit-learn classifier wrapper ignoring some input columns."""

import re

import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import (
    check_array,
    check_X_y,
)


class _BaseColumnIgnoringClassifier(BaseEstimator, ClassifierMixin):
    """A base class for sklearn classifier wrappers that ignores input columns.
    """

    def __init__(self, clf):
        self.clf = clf
        if hasattr(self.clf, "decision_function"):
            setattr(self, 'decision_function', self._hidden_decision_function)
        if hasattr(self.clf, "predict_proba"):
            setattr(self, 'predict_proba', self._hidden_predict_proba)

    def fit(self, X, y, validate=True, sparse=False):
        """Fits the classifier

        Parameters
        ----------
        X : pandas.DataFrame, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.
        validate : bool, default True
            If set, input arrays type is validated to be ndarray.
        sparse : bool, default False
            If set, X is assumed to be a pandas.SparseDataFrame object.

        Returns
        -------
        self : object
            Returns self.
        """
        self.classes_ = unique_labels(y)
        # self.classes_ = np.unique(y)
        inner_X = self._transform_X(X)
        if validate:
            inner_X, y = check_X_y(inner_X, y)
        if sparse:
            inner_X = sp.sparse.csr_matrix(inner_X.to_coo())
            y = np.array(y)
        self.clf = self.clf.fit(inner_X, y)
        return self

    def predict(self, X):
        """Predict labels.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of int of shape = [n_samples]
            Predicted labels for the given inpurt samples.
        """
        inner_X = check_array(self._transform_X(X))
        return self.clf.predict(inner_X)

    def _hidden_predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute classes_.
        """
        inner_X = check_array(self._transform_X(X))
        return self.clf.predict_proba(inner_X)

    def _hidden_decision_function(self, X):
        """Predict confidence scores for samples.

        The confidence score for a sample is the signed distance of that sample
        to the hyperplane.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            onfidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """
        inner_X = check_array(self._transform_X(X))
        return self.clf.decision_function(inner_X)


class IxColIgnoringClassifier(_BaseColumnIgnoringClassifier):
    """A sklearn classifier wrapper that ignores input columns by index.

    Parameters
    ----------
    clf : classifier object implementing 'fit'
        The object to use to fit the data.
    col_ignore : list
        A list of indices of columns to ignore.
    """
    def __init__(self, clf, col_ignore):
        super(IxColIgnoringClassifier, self).__init__(clf=clf)
        self.col_ignore = col_ignore

    def _transform_X(self, X):
        X = np.array(X)
        col_keep = [
            i for i in range(X.shape[1]) if i not in self.col_ignore]
        return X[:, col_keep]


class ObjColIgnoringClassifier(_BaseColumnIgnoringClassifier):
    """A sklearn classifier wrapper that ignores object columns in dataframes.

    Parameters
    ----------
    clf : classifier object implementing 'fit'
        The object to use to fit the data.
    """
    def __init__(self, clf):
        super(ObjColIgnoringClassifier, self).__init__(clf=clf)
        self.col_to_drop = None

    def _transform_X(self, X):
        if self.col_to_drop is None:
            self.col_to_drop = [
                col_name for col_name, dtype in X.dtypes.items()
                if dtype == object
            ]
        return X.drop(self.col_to_drop, axis=1)


class PatternColIgnoringClassifier(_BaseColumnIgnoringClassifier):
    """A sklearn classifier wrapper that ignores columns in dataframes by name.

    Parameters
    ----------
    clf : classifier object implementing 'fit'
        The object to use to fit the data.
    pattern : str
        The name pattern to match column names with.
    exclude : bool, default True
        If set to True (default), all columns matching the pattern are ignored.
        Otherwise, all columns NOT matching the pattern are ignored.
    """
    def __init__(self, clf, pattern, exclude=False):
        super(PatternColIgnoringClassifier, self).__init__(clf=clf)
        self.pattern = pattern
        self.exclude = exclude
        if self.exclude:
            self.decision_func_ = lambda x: x
        else:
            self.decision_func_ = lambda x: not x

    def _transform_X(self, X):
        col_to_drop = [
            col_name for col_name in X.columns
            if self.decision_func_(re.match(self.pattern, col_name))
        ]
        return X.drop(col_to_drop, axis=1)
