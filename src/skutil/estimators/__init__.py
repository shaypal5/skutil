import contextlib

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
    # use `contextlib.suppress(KeyError)` instead of `try`-`except`-`pass
    with contextlib.suppress(KeyError):
        globals().pop(name)

with contextlib.suppress(NameError):
    del name
del contextlib
