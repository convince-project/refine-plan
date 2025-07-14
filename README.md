# REFINE-PLAN

This repo contains the source code for REFINE-PLAN, an automated tool for refining hand-designed behaviour trees to achieve higher robustness to uncertainty.

If you use this repository, please consider citing:

```
@inproceedings{street2025planning,
  title={Planning under Uncertainty from Behaviour Trees},
  author={Street, Charlie and Grubb, Oliver and Mansouri, Masoumeh},
  booktitle={Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2025}
}
```

## Installation

REFINE-PLAN has been tested on Ubuntu 22.04 with Python 3.10.12.

### Installing Dependencies

COVERAGE-PLAN requires the following dependencies:

* [Numpy](https://numpy.org/) (Tested with 26.4)
* [Sympy](https://www.sympy.org/en/index.html) (Tested with 1.12)
* [Pyeda](https://pyeda.readthedocs.io/en/latest/)  (Tested with 0.29.0)
* [PyAgrum](https://pyagrum.readthedocs.io/en/1.15.1/index.html) (Tested with 1.14.1)
* [PyMongo](https://pymongo.readthedocs.io/en/stable/index.html) (Tested with 4.8.0)
* [Pandas](https://pandas.pydata.org/) (Tested with 2.2.1)
* [Stormpy](https://moves-rwth.github.io/stormpy/index.html) (Tested with 1.8.0) 
* [MongoDB](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/) (Tested with 7.0.12) - only required for unit tests.

The first six dependencies can be installed via:
```
pip install -r requirements.txt
```

`MongoDB` can be installed using the [official instructions](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/).

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

## Running the experiments for 'Planning under Uncertainty from Behaviour Trees'

In case of code updates after paper submission/publication, please consider downloading the `iros-2025` release of this repository.

### Book Store Navigation Domain

The planning script for the book store experiment is found at `bin/bookstore_planning.py`.
This script processes the data collected in a MongoDB instance, learns the Bayesian networks, and synthesises the policy.
The dataset, Bayesian networks, and refined policy are already generated and can be found in `data/bookstore/`.

If you wish to generate them yourself, run the following in `bin/bookstore_planning.py` by uncommenting the corresponding line in lines 225-227:
* `write_mongodb_to_yaml(sys.argv[1])` writes the data from a MongoDB instance to a YAML file. `sys.argv[1]` should be a MongoDB instance address.
* `learn_options()` learns the Bayesian networks from the YAML dataset.
* `run_planner()` uses the learned Bayesian networks to build an MDP and synthesise a policy.

To execute the initial BT or refined policy in Gazebo, please install [turtlebot_bookstore_sim](https://github.com/HyPAIR/turtlebot_bookstore_sim) and read the instructions in the README.

### Vacuum Cleaner Search Domain

The planning script for the vacuum cleaner search experiment is found at `bin/house_planning.py`.
This script behaves similarly to `bin/bookstore_planning.py`, with identical function names and behaviours.
The dataset, Bayesian networks, and refined policy are already generated and can be found in `data/house/`.
To generate these yourself, run `write_mongodb_to_yaml(sys.argv[1])`, `learn_options()`, or `run_planner()` in `bin/house_planning.py` by uncommenting the corresponding line in lines 249-251.
See the book store instructions above for the expected behaviour of these functions.


To execute the initial BT or refined policy in Gazebo, please install [turtlebot_house_sim](https://github.com/HyPAIR/turtlebot_house_sim) and read the instructions in the README.


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
make clean
```

## Maintainer

This repository is maintained by:

| | | |
|:---:|:---:|:---:|
| Charlie Street | [@charlie1329](https://github.com/charlie1329) |[c.l.street@bham.ac.uk](mailto:c.l.street@bham.ac.uk?subject=[GitHub]%20Refine%20Plan)|
