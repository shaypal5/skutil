"""scikit-learn classifier wrapper ignoring some input columns."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import (
    check_array,
    check_X_y,
)


class ColumnIgnoringClassifier(BaseEstimator, ClassifierMixin):
    """A sklearn classifier wrapper that ignores some input columns.

    Parameters
    ----------
    clf : classifier object implementing 'fit'
        The object to use to fit the data.
    col_ignore : list
        A list of indices of columns to ignore.
    """
    def __init__(self, clf, col_ignore):
        self.clf = clf
        self.col_ignore = col_ignore

    def fit(self, X, y):
        """Fits the classifier

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        X = np.array(X)
        col_keep = [
            i for i in range(X.shape[1]) if i not in self.col_ignore]
        inner_X = X[:, col_keep]
        inner_X, y = check_X_y(inner_X, y)
        self.clf = self.clf.fit(inner_X, y)
        return self

    def _transform_X(self, X):
        X = np.array(X)
        col_keep = [
            i for i in range(X.shape[1]) if i not in self.col_ignore]
        inner_X = X[:, col_keep]
        return check_array(inner_X)

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
        return self.clf.predict(self._transform_X(X))

    def predict_proba(self, X):
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
        return self.clf.predict_proba(self._transform_X(X))
