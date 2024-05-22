# REFINE-PLAN

This repo contains the source code for REFINE-PLAN, an automated tool for refining hand-designed behaviour trees to achieve higher robustness to uncertainty.


## Installation

REFINE-PLAN has been tested on Ubuntu 22.04 with Python 3.10.12.

### Installing Dependencies

COVERAGE-PLAN requires the following dependencies:

* [Numpy](https://numpy.org/) (Tested with 26.4)
* [Sympy](https://www.sympy.org/en/index.html) (Tested with 1.12)
* [Pyeda](https://pyeda.readthedocs.io/en/latest/)  (Tested with 0.29.0)
* [Stormpy](https://moves-rwth.github.io/stormpy/index.html) (Tested with 1.8.0) 

The first three dependencies can be installed via:
```
pip install -r requirements.txt
```

Installing `stormpy` is more involved. Please see below.

### Installing Stormpy

Stormpy requires the use of [Storm](https://www.stormchecker.org/).
Instructions for installing stormpy alongside its dependencies can be found [here](https://moves-rwth.github.io/stormpy/installation.html#).

REFINE-PLAN has been tested with the following software versions during the above installation steps:

* Storm - 1.8.1. Specifically we use this [commit](https://github.com/moves-rwth/storm/commit/5b662c76549558750938fdb980c5727b062d662d).
* Carl - 14.27 (this is the `carl-storm` version of Carl, as mentioned in the Storm installation instructions)
* pycarl - 2.2.0
* stormpy - 1.8.0

As Storm is updated, you may want newer versions of these libraries. The current master of Stormpy and Storm should usually be compatible.

### Installation for Users (Pip Install)

After installing all dependencies, run the following in the root directory of this repository:

```bash
pip install .
```

### Installation for Developers (Update PYTHONPATH)

After installing all dependencies, run the following in the root directory of this repository:

```bash
./setup_dev.sh
```

## Run examples

Small examples of REFINE-PLAN can be found in the `bin` directory.


## Run the Unit Tests

To run all unit tests, run:

```bash
cd tests
python3 -m unittest discover --pattern=*.py
```


## Build the documentation

The REFINE-PLAN documentation can be found [here](https://convince-project.github.io/refine-plan). 
If you want to build it locally, do the following:


1. Install the required packages:

    ```bash
    pip install -r docs/requirements.txt
    ```

2. Install the package to be documented:

    ```bash
    pip install refine_plan/
    ```
    
    Or add it to your Python path:
    ```bash
    ./setup_dev.sh
    ```

3. Build the documentation:

    ```bash
    cd docs
    make html
    ```

4. Look at the documentation:

    ```bash
    cd docs
    firefox build/html/index.html
    ```

### Clean documentation build artifacts

If you want to clean the documentation, you can run:

```bash
cd docs
rm -r source/API
make clean
```
