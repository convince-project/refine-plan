Installation Instructions
=========================

.. role:: bash(code)
   :language: bash

REFINE-PLAN has been tested on Ubuntu 22.04 with Python 3.10.12.

Installing Dependencies
-----------------------

COVERAGE-PLAN requires the following dependencies:

* `Numpy`_ (Tested with 26.4)
* `Sympy`_ (Tested with 1.12)
* `Pyeda`_ (Tested with 0.29.0)
* `PyAgrum`_ (Tested with 1.14.1)
* `PyMongo`_ (Tested with 4.8.0)
* `Pandas`_ (Tested with 2.2.1)
* `Stormpy`_ (Tested with 1.8.0) 
* `MongoDB`_ (Tested with 7.0.12) - only required for unit tests.

The first six dependencies can be installed via:

.. code-block:: bash

	pip install -r requirements.txt

:bash:`MongoDB` can be installed using the `official instructions <https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/>`_.

Installing :bash:`stormpy` is more involved. Please see below.

Installing Stormpy
^^^^^^^^^^^^^^^^^^

Stormpy requires the use of `Storm <https://www.stormchecker.org/>`_.
Instructions for installing stormpy alongside its dependencies can be found `here <https://moves-rwth.github.io/stormpy/installation.html#>`_.

REFINE-PLAN has been tested with the following software versions during the above installation steps:

* Storm - 1.8.1. Specifically we use this `commit <https://github.com/moves-rwth/storm/commit/5b662c76549558750938fdb980c5727b062d662d>`_.
* Carl - 14.27 (this is the :bash:`carl-storm` version of Carl, as mentioned in the Storm installation instructions)
* pycarl - 2.2.0
* stormpy - 1.8.0

As Storm is updated, you may want newer versions of these libraries. The current master of Stormpy and Storm should usually be compatible.

Installation for Users (Pip Install)
------------------------------------

After installing all dependencies, run the following in the root directory of this repository:

.. code-block:: bash

	pip install .


Installation for Developers (Update PYTHONPATH)
-----------------------------------------------

After installing all dependencies, run the following in the root directory of this repository:

.. code-block:: bash

	./setup_dev.sh

Run examples
------------

Small examples of REFINE-PLAN can be found in the :bash:`bin` directory.


Run the Unit Tests
------------------

To run all unit tests, run:

.. code-block:: bash

	cd tests
	python3 -m unittest discover --pattern=*.py

Build the Documentation
-----------------------
 
If you want to build the REFINE-PLAN documentation locally, do the following:


1. Install the required packages:

.. code-block:: bash

    pip install -r docs/requirements.txt

2. Install the package to be documented:

.. code-block:: bash

    pip install refine_plan/
    
Or add it to your Python path:
    
.. code-block:: bash

    ./setup_dev.sh

3. Build the documentation:

.. code-block:: bash

    cd docs
    make html

4. Look at the documentation:

.. code-block:: bash

    cd docs
    firefox build/html/index.html

Clean documentation build artifacts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to clean the documentation, you can run:

.. code-block:: bash
	
	cd docs
	make clean


.. _Numpy: https://numpy.org/
.. _Sympy: https://www.sympy.org/en/index.html
.. _Pyeda: https://pyeda.readthedocs.io/en/latest/
.. _PyAgrum: https://pyagrum.readthedocs.io/en/1.15.1/index.html
.. _PyMongo: https://pymongo.readthedocs.io/en/stable/index.html
.. _Pandas: https://pandas.pydata.org/
.. _Stormpy: https://moves-rwth.github.io/stormpy/index.html
.. _MongoDB: https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/
