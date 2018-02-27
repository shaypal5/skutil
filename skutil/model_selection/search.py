"""Extends scikit-learn tools for hyper-parameter search."""

import copy
from collections import Sequence

import numpy as np
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
        A list of grids defining bad parameter combinations to avoid, each
        given as a dict of string to sequence. All combinations containing a
        bad combinations will be avoided.

    Example
    -------
    >>> param = {'a': [1, 2], 'b': [3, 4, 5]}
    >>> grid = ConstrainedParameterGrid(param, [{'a': [1], 'b': [4]}])
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
        if bad_comb is not None:
            self.bad_grids = [ParameterGrid(bad_dict) for bad_dict in bad_comb]

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
            bad_param_set_list = []
            for bad_param_grid in self.bad_grids:
                for bad_param_set in bad_param_grid:
                    bad_param_set_list.append(bad_param_set.items())
            for params in super().__iter__():
                bad_found = False
                for bad_param_set in bad_param_set_list:
                    if bad_param_set <= params.items():
                        bad_found = True
                        break
                if not bad_found:
                    yield params

    def partial(self, assign_grid):
        """Returns a new parameter grid by the given partial assignment.

        Parameters
        ----------
        assign_grid : dict of string to object or sequence
            A, possibly partial, assignment to the parameters of the grid. Keys
            not appearing in this grid are ignored.

        Returns
        -------
        ConstrainedParameterGrid
            A new parameter grid induced by the given partial assignment.

        Example
        -------
        >>> param = {'a': [1, 2], 'b': [3, 4, 5]}
        >>> grid = ConstrainedParameterGrid(param, [{'a': [1], 'b': [4]}])
        >>> part_grid = grid.partial({'a': 1})
        >>> for params in part_grid: print(sorted(params.items()))
        [('a', 1), ('b', 3)]
        [('a', 1), ('b', 5)]
        >>> part_grid = grid.partial({'b': [3, 4]})
        >>> for params in part_grid: print(sorted(params.items()))
        [('a', 1), ('b', 3)]
        [('a', 2), ('b', 3)]
        [('a', 2), ('b', 4)]
        """
        new_params = [copy.deepcopy(grid) for grid in self.param_grid]
        for grid in new_params:
            for key in grid:
                if key in assign_grid:
                    val = assign_grid[key]
                    if not isinstance(val, (np.ndarray, Sequence)) or (
                            isinstance(val, str)):
                        val = [val]
                    grid[key] = val
        return ConstrainedParameterGrid(
            param_grid=new_params,
            bad_comb=self.bad_comb,
        )
