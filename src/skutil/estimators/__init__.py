from .classifier_map import (  # noqa: F401
    classifier_by_params,
    classifier_cls_by_name,
)
from .col_ignore_clf import (  # noqa: F401
    IxColIgnoringClassifier,
    ObjColIgnoringClassifier,
    PatternColIgnoringClassifier,
)
from .regressor_map import (  # noqa: F401
    regressor_cls_by_name,
)

for name in [
    "col_ignore_clf",
    "classifier_map",
    "name",
]:
    try:
        globals().pop(name)
    except KeyError:
        pass
try:
    del name  # pylint: disable=W0631
except NameError:
    pass
