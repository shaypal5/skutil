"""Utilities for pandas."""

import skutil.estimators

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
del _version

for name in ['get_versions', '_version', 'pdutil', 'skutil', 'name']:
    try:
        globals().pop(name)
    except KeyError:
        pass
