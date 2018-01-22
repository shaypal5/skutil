from .col_ignore_clf import ColumnIgnoringClassifier
from .classifier_map import classifier_cls_by_name

for name in [
    'col_ignore_clf', 'classifier_map', 'name',
]:
    try:
        globals().pop(name)
    except KeyError:
        pass
try:
    del name  # pylint: disable=W0631
except NameError:
    pass
