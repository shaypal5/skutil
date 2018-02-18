"""Test classifier_cls_by_name."""


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


from skutil.estimators import classifier_cls_by_name


def test_base():
    assert classifier_cls_by_name('LogisticRegression') == LogisticRegression
    assert classifier_cls_by_name('logisticregression') == LogisticRegression
    assert classifier_cls_by_name('logreg') == LogisticRegression
    assert classifier_cls_by_name('SVC') == SVC
    assert classifier_cls_by_name('svc') == SVC
    assert classifier_cls_by_name('SVM') == SVC
    assert classifier_cls_by_name('svm') == SVC
    assert classifier_cls_by_name('MultinomialNB') == MultinomialNB
    assert classifier_cls_by_name('multinomialnb') == MultinomialNB
    assert classifier_cls_by_name('mnb') == MultinomialNB
