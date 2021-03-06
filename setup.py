"""Setup for the skutil package."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import warnings
import setuptools
import versioneer


# Require Python 3.4 or higher
if sys.version_info.major < 3 or sys.version_info.minor < 5:
    warnings.warn("skutil requires Python 3.5 or higher!")
    sys.exit(1)


INSTALL_REQUIRES = ['numpy', 'decore']
TEST_REQUIRES = [
    'pytest', 'scikit-learn', 'coverage', 'pytest-cov', 'pandas', 'pdutil'
]

with open('README.rst', encoding="utf-8") as f:
    README = f.read()

setuptools.setup(
    author="Shay Palachy",
    author_email="shay.palachy@gmail.com",
    name='skutil',
    description='Utilities for scikit-learn.',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    long_description=README,
    url='https://github.com/shaypal5/skutil',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        INSTALL_REQUIRES
    ],
    extras_require={
        'test': TEST_REQUIRES + INSTALL_REQUIRES,
    },
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
)
