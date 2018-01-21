"""Test colunm-ignoring classifier wrapper."""

import pytest
import pandas as pd
from sklearn.svm import SVC
from pdutil.transform import x_y_by_col_lbl

from skutil.estimators import ColumnIgnoringClassifier


BASE_DATA = [
    [1, 'asd', 34, 0],
    [2, '98', 993, 1],
]


def test_base_case():
    clf = SVC(probability=True)
    df = pd.DataFrame(data=BASE_DATA, columns=['x1', 'x2', 'x3', 'y'])
    X, y = x_y_by_col_lbl(df=df, y_col_lbl='y')
    with pytest.raises(ValueError):
        clf.fit(X, y)

    ignore_clf = ColumnIgnoringClassifier(
        clf=clf,
        col_ignore=[list(df.columns).index('x2')],
    )
    fitted_ignore_clf = ignore_clf.fit(X, y)
    assert fitted_ignore_clf == ignore_clf

    test_data = [[1, '233', 30]]
    test_df = pd.DataFrame(test_data)
    res = fitted_ignore_clf.predict(test_df)
    assert len(res) == 1

    res2 = fitted_ignore_clf.predict_proba(test_df)
    assert len(res2) == 1
    assert len(res2[0] == 2)
    for proba in res2[0]:
        assert isinstance(proba, float)
        assert proba <= 1
        assert proba >= 0