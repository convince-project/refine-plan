Welcome to the REFINE-PLAN documentation!
===========================================

Welcome to the documentation for REFINE-PLAN, an automatic refinement tool for behaviour trees (BTs).
The final release will convert an input BT into a refined BT which can be executed with BehaviorTree.cpp.
The current release converts a set of `options`_ into a refined BT which requires some adjustments to execute with BehaviorTree.cpp.

Installation instructions can be found :doc:`here <../installation>`.
The :doc:`tutorials <../tutorials>` page contains an in-depth tutorial describing how to synthesise a refined BT given a set of options, and also how to learn a set of options from data.
These two tutorials can be combined to synthesise a refined BT given a dataset describing the execution of each action node in the input BT.

REFINE-PLAN still requires code to automatically extract a state space from the input/initial BT using the CONVINCE modelling formalisms.
It also requires code to connect the refined BT to the appropriate plugins defined in the CONVINCE system model.
This functionality will appear in the next software release.

Contents
--------

.. toctree::
   :maxdepth: 2
   
   installation
   tutorials
   api
   contacts

.. _options: https://www.sciencedirect.com/science/article/pii/S0004370299000521
.. _here: https://github.com/convince-project/refine-plan/blob/main/README.md
