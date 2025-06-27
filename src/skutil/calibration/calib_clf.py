"""scikit-learn classifier wrapper calibrating after fit."""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
# from sklearn.calibration import CalibratedClassifierCV

from .calib_clf_cv import UnsafeCalibratedClassifierCV


class CalibratingCvClassifier(BaseEstimator, ClassifierMixin):
    """A sklearn classifier wrapper using part of the train set to calibrate.

    Parameters
    ----------
    clf : classifier object implementing 'fit'
        The object to use to fit the data.
    method : str, optional
        See the same parameter for sklearn.calibration.CalibratedClassifierCV.
    val_size : integer or float, optional
        The portion of the train set to take as a validation set to calibrate
        the model with. If a float, interpreted as the ratio of the train set
        to take, and so must be between 0 and 1. If integer, interpreted as
        the number of entries to take. In both cases, the appropriate number
        of items is randomly sampled. By default, set to 0.25.
    stratify : bool, default True
        If set to True, the validation set is sampled in a stratified way.
    """
    def __init__(self, clf, method=None, val_size=None, stratify=True):
        self.clf = clf
        self.method = method
        self.val_size = val_size
        self.stratify = stratify

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
        strat = None
        if self.stratify:
            strat = y
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_size, stratify=strat)
        self.clf.fit(X_train, y_train)
        self._calib = UnsafeCalibratedClassifierCV(
            base_estimator=self.clf, method=self.method, cv='prefit')
        self._calib.fit(X_val, y_val)
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
        return self._calib.predict(X)

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
        return self._calib.predict_proba(X)
