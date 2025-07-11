"""A percentile selector you can fit once and use for many percentiles."""

from warnings import warn

import numpy as np
from sklearn.feature_selection import (
    SelectPercentile,
)
from sklearn.feature_selection._univariate_selection import _clean_nans
from sklearn.utils import safe_mask
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
)


class MultiSelectPercentile(SelectPercentile):
    """A feature selector that can be used to select multiple percentiles.

    This is a wrapper around sklearn's SelectPercentile that allows you to
    fit the selector once and then use it to transform data for multiple
    percentiles. This is useful when you want to select features at different
    percentiles without having to refit the selector each time.

    Parameters
    ----------
    score_func : callable, default=None
        The function to use to compute the scores for each feature. If None,
        the default score function is used.
    percentile : int, default=10
        The initial percentile to select features from. This is used only for
        fitting the selector. After fitting, you can use
        `transform_by_percentile` to transform data for different percentiles.

    Attributes
    ----------
    scores_ : array of shape [n_features]
        The scores of the features after fitting the selector.
    pvalues_ : array of shape [n_features]
        p-values of feature scores, None if score_func returned only scores.
    n_features_in_ : int
        The number of features seen during fitting.
    feature_names_in_ : array of shape [n_features_in_]
        Names of features seen during fitting. Only available if the input
        data has feature names.

    """

    def _custom_support_mask(self, percentile):
        check_is_fitted(self, "scores_")

        # Cater for NaNs
        if percentile == 100:
            return np.ones(len(self.scores_), dtype=np.bool)
        elif percentile == 0:
            return np.zeros(len(self.scores_), dtype=np.bool)

        scores = _clean_nans(self.scores_)
        threshold = np.percentile(scores, 100 - percentile)
        # threshold = stats.scoreatpercentile(scores, 100 - percentile)
        mask = scores > threshold
        ties = np.where(scores == threshold)[0]
        if len(ties):
            max_feats = int(len(scores) * percentile / 100)
            kept_ties = ties[: max_feats - mask.sum()]
            mask[kept_ties] = True
        return mask

    def transform_by_percentile(self, X, percentile):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        percentile : int
            Percent of features to keep

        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.

        """
        X = check_array(X, accept_sparse="csr")
        mask = self._custom_support_mask(percentile=percentile)
        if not mask.any():
            warn(
                "No features were selected: either the data is"
                " too noisy or the selection test too strict.",
                UserWarning,
                stacklevel=2,
            )
            return np.empty(0).reshape((X.shape[0], 0))
        if len(mask) != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")
        return X[:, safe_mask(X, mask)]
