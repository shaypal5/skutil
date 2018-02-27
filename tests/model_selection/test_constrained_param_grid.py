"""Test the ConstainedParameterGrid class."""

from sklearn.model_selection import ParameterGrid

from skutil.model_selection import ConstrainedParameterGrid


PARAMS1 = {
    'a': [8, 9],
    'b': [1, 2, 3],
    'c': [5, 6],
}
CONSTRAINTS1 = [{'a': [8], 'b': [3]}]


def test_base():
    grid = ParameterGrid(PARAMS1)
    cgrid = ConstrainedParameterGrid(
        param_grid=PARAMS1,
        bad_comb=CONSTRAINTS1,
    )
    reg_param_sets = list(grid)
    const_param_sets = list(cgrid)
    assert len(reg_param_sets) == 12
    assert len(const_param_sets) == 10
    for param_set in const_param_sets:
        assert not (param_set['a'] == 8 and param_set['b'] == 3)


def test_no_constraints():
    grid = ParameterGrid(PARAMS1)
    cgrid = ConstrainedParameterGrid(
        param_grid=PARAMS1,
        bad_comb=None,
    )
    reg_param_sets = list(grid)
    const_param_sets = list(cgrid)
    assert len(reg_param_sets) == 12
    assert len(const_param_sets) == 12
    found_bad = False
    for param_set in const_param_sets:
        if param_set['a'] == 8 and param_set['b'] == 3:
            found_bad = True
    assert found_bad


PARAMS2 = {
    'a': [1, 2],
    'b': ['Xor', 'Yank'],
}


def test_partial_with_str_assignment():
    cgrid = ConstrainedParameterGrid(
        param_grid=PARAMS2,
        bad_comb=None,
    )
    partial_grid = cgrid.partial({'a': 1})
    param_sets = list(cgrid)
    partial_param_sets = list(partial_grid)
    assert len(param_sets) == 4
    assert len(partial_param_sets) == 2
    found_bad = False
    for param_set in partial_param_sets:
        if param_set['a'] == 2:
            found_bad = True
    assert not found_bad
