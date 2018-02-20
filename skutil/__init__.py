"""Utilities for pandas."""

import skutil.estimators  # noqa: F401

from ._version import get_versions
__version__ = get_versions()['version']

for name in ['get_versions', '_version', 'skutil', 'name']:
    try:
        globals().pop(name)
    except KeyError:
        pass
