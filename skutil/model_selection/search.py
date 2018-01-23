"""Extends scikit-learn tools for hyper-parameter search."""

from sklearn.model_selection import ParameterGrid


class ConstrainedParameterGrid(ParameterGrid):
    """Grid of discrete-valued parameters with constraints.

    Parameters
    ----------
    param_grid : dict of string to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.

        An empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See the examples below.
    bad_comb : list of dicts, optional
        A list of bad parameter combinations to avoid, each given as a dict.
        All combinations containing a bad combinations will be avoided.

    Example
    -------
    >>> param = {'a': [1, 2], 'b': [3, 4, 5]}
    >>> grid = ConstrainedParameterGrid(param, [{'a': 1, 'b': 4}])
    >>> for params in grid: print(sorted(params.items()))
    [('a', 1), ('b', 3)]
    [('a', 1), ('b', 5)]
    [('a', 2), ('b', 3)]
    [('a', 2), ('b', 4)]
    [('a', 2), ('b', 5)]
    """

    def __init__(self, param_grid, bad_comb=None):
        super().__init__(param_grid)
        self.bad_comb = bad_comb

    def __iter__(self):
        """Iterate over the points in the grid.
        Returns
        -------
        params : iterator over dict of string to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        if self.bad_comb is None:
            for params in super().__iter__():
                yield params
        else:
            for params in super().__iter__():
                bad_found = False
                for bad_sub_param in self.bad_comb:
                    if bad_sub_param.items() <= params.items():
                        bad_found = True
                        break
                if not bad_found:
                    yield params
