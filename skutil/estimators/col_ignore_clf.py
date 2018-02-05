"""scikit-learn classifier wrapper ignoring some input columns."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import (
    check_array,
    check_X_y,
)


class ColumnIgnoringClassifier(BaseEstimator, ClassifierMixin):
    """A sklearn classifier wrapper that ignores input columns by index.

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

    def _transform_X(self, X):
        X = np.array(X)
        col_keep = [
            i for i in range(X.shape[1]) if i not in self.col_ignore]
        return X[:, col_keep]

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
        inner_X = self._transform_X(X)
        inner_X, y = check_X_y(inner_X, y)
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
        inner_X = check_array(self._transform_X(X))
        return self.clf.predict_proba(inner_X)


class ObjColIgnoringClassifier(BaseEstimator, ClassifierMixin):
    """A sklearn classifier wrapper that ignores object columns in dataframes.

    Parameters
    ----------
    clf : classifier object implementing 'fit'
        The object to use to fit the data.
    """
    def __init__(self, clf):
        self.clf = clf
        self.col_to_drop = None

    def _transform_X(self, X):
        if self.col_to_drop is None:
            self.col_to_drop = [
                col_name for col_name, dtype in X.dtypes.items()
                if dtype == object
            ]
        return X.drop(self.col_to_drop, axis=1)

    def fit(self, X, y):
        """Fits the classifier

        Parameters
        ----------
        X : pandas.DataFrame, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        inner_X = self._transform_X(X)
        inner_X, y = check_X_y(inner_X, y)
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
        inner_X = check_array(self._transform_X(X))
        return self.clf.predict_proba(inner_X)
