skutil
######
|PyPI-Status| |PyPI-Versions| |Build-Status| |Codecov| |LICENCE|

Utilities for scikit-learn.

.. code-block:: python

  from skutil.estimators import ColumnIgnoringClassifier
  # use a classifier that can't handle string data as 
  # an inner classifier in some stacked model, for example

.. contents::

.. section-numbering::


Installation
============

.. code-block:: bash

  pip install skutil


Basic Use
=========

``skutil`` is divided into several sub-modules by functionality:

estimators
----------

``ColumnIgnoringClassifier`` - An sklearn classifier wrapper that ignores input columns by index. 

``ObjColIgnoringClassifier`` - An sklearn classifier wrapper that ignores object columns in dataframes.

``classifier_cls_by_name`` - Get an sklearn classifier class by name. Also supports lowercasing and some shorthands (e.g. svm for SVC, logreg and lr for LogisticRegression).

model_selection
---------------

``ConstrainedParameterGrid`` - Grid of discrete-valued parameters with constraints.


Contributing
============

Package author and current maintainer is Shay Palachy (shay.palachy@gmail.com); You are more than welcome to approach him for help. Contributions are very welcomed.

Installing for development
----------------------------

Clone:

.. code-block:: bash

  git clone git@github.com:shaypal5/skutil.git


Install in development mode, and with test dependencies:

.. code-block:: bash

  cd skutil
  pip install -e ".[test]"


Running the tests
-----------------

To run the tests use:

.. code-block:: bash

  cd skutil
  pytest


Adding documentation
--------------------

The project is documented using the `numpy docstring conventions`_, which were chosen as they are perhaps the most widely-spread conventions that are both supported by common tools such as Sphinx and result in human-readable docstrings. When documenting code you add to this project, follow `these conventions`_.

.. _`numpy docstring conventions`: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _`these conventions`: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt


Credits
=======

Created by Shay Palachy (shay.palachy@gmail.com).


.. |PyPI-Status| image:: https://img.shields.io/pypi/v/skutil.svg
  :target: https://pypi.python.org/pypi/skutil

.. |PyPI-Versions| image:: https://img.shields.io/pypi/pyversions/skutil.svg
   :target: https://pypi.python.org/pypi/skutil

.. |Build-Status| image:: https://travis-ci.org/shaypal5/skutil.svg?branch=master
  :target: https://travis-ci.org/shaypal5/skutil

.. |LICENCE| image:: https://img.shields.io/github/license/shaypal5/skutil.svg
  :target: https://github.com/shaypal5/skutil/blob/master/LICENSE

.. |Codecov| image:: https://codecov.io/github/shaypal5/skutil/coverage.svg?branch=master
   :target: https://codecov.io/github/shaypal5/skutil?branch=master
