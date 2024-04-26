Tutorials
=========

How to synthesise a refined BT from a set of options
----------------------------------------------------

For this tutorial, we will consider a basic graph search problem, where robot actions
are navigation actions along edges.

The source code for this tutorial can be found `here`_.

Consider the following undirected graph, where values along edges represent the cost of traversal:

|Graph|

Let's say that we want to synthesise a BT which navigates the robot to node 5 along the shortest path.
For an edge (v,v'), the robot reaches v' with probability 0.7. The remainder of the probability is uniformly distributed among the other possible successors from v.
The expected cost of edge (v,v') is the weighted average cost according to the edge weights and transition probabilities.

**In this tutorial, we will construct an option for each edge action, and use this to synthesise a BT.**

We begin by importing all necessary code from ``refine_plan``. We will cover these imports throughout the tutorial.

.. code-block:: python

    from refine_plan.algorithms.refine import synthesise_bt_from_options
    from refine_plan.models.condition import Label, EqCondition
    from refine_plan.models.state_factor import IntStateFactor
    from refine_plan.models.option import Option

Our first step in building the options is to define our state space.
We do this by defining a *state factor* which capture the robot's current node.
In this instance, the nodes are integers, and so we create an ``IntStateFactor`` called ``node_sf`` which can take values 0 through 6 inclusive.

.. code-block:: python

    # Step 1: Build our state factor for the graph
    # The state factor captures the robot's current node
    node_sf = IntStateFactor("node_sf", 0, 6)

To define the probabilistic transitions and reward function, we must now capture the connectivity of the above graph.
This can be achieved through a nested dictionary:

.. code-block:: python

    # Step 2: Define the connectivity of our graph as a dictionary
    # graph[v][v'] = cost
    graph = {
        0: {1: 5, 3: 7},
        1: {0: 5, 2: 6},
        2: {1: 6, 5: 4},
        3: {0: 7, 4: 3},
        4: {3: 3, 5: 2, 6: 4},
        5: {2: 4, 4: 2, 6: 8},
        6: {4: 4, 5: 8},
    }

With the graph and our state space defined, we can now compute the probabilistic transitions and 
rewards. In ``refine_plan``, we define these using *conditions*.

For this graph search example, we use ``EqConditions``, or equality conditions. These conditions are 
satisfied when a state holds a particular value for a state factor.

With this, we represent probabilistic transitions as a (pre-condition, probabilistic post-conditions) pair.
The pre-condition defines the states where this edge/option can be navigated.
The probabilistic post-conditions is a dictionary from post-condition to probability.
This defines the effects of an edge/option, and the probability of these effects.

We can compute these transitions as follows:

.. code-block:: python

    # Step 3: Define our transition function for an edge
    # We say that for edge (v,v') there is a 0.7 chance of reaching v
    # and a 0.3 chance of that being split evenly across the other possible successor nodes
    def trans_probs(src, dst):
        """Compute a (pre_cond, prob_post_conds) pair for a given edge.

        pre_cond is the guard for an edge.
        prob_post_conds is a dictionary from post conditions to probabilities

        Args:
            src: The start node of an edge
            dst: The destination node of an edge

        Returns:
            A (pre_cond, prob_post_conds) pair
        """
        pre_cond = EqCondition(node_sf, src)
        prob_post_conds = {}

        for succ in graph[src]:
            post_cond = EqCondition(node_sf, succ)
            prob = 0.7 if succ == dst else 0.3 / (len(graph[src]) - 1.0)
            prob_post_conds[post_cond] = prob

        return (pre_cond, prob_post_conds)


The rewards can be computed in a similar way. In ``refine-plan``, rewards for an edge/option are given as a 
(pre-condition, reward) pair. 
The pre-condition defines when this reward is given.
The reward for an edge can be written as: 

.. code-block:: python

    # Step 4: Define our reward function for an edge
    # Here, we define our reward function to be the expected cost of the edge action
    def reward(src, dst):
        """Compute a (pre_cond, reward) pair for a given edge.

        pre_cond is the guard for an edge.
        reward is the expected cost of an edge

        Args:
            src: The start node of an edge
            dst: The destination node of an edge

        Returns:
            A (pre_cond, reward) pair
        """
        pre_cond = EqCondition(node_sf, src)
        reward = 0.0

        for succ in graph[src]:
            # Get our transition probability again
            prob = 0.7 if succ == dst else 0.3 / (len(graph[src]) - 1.0)
            edge_weight = graph[src][succ]
            # Compute the weighted average
            reward += prob * edge_weight

        return (pre_cond, reward)

With the transitions and rewards defined, we can now define the *options*.
Here, we construct an option for each edge the robot can navigate on.
The resulting options are very simple, but can be expanded through more complex transitions and rewards.
An ``Option`` requires:

1. A name, e.g. ``e01`` for the edge between node 0 and 1.

2. A list of transitions, i.e. a list of (pre-condition, probabilistic post-conditions) pairs.

3. A list of rewards, i.e. a list of (pre-condition, reward) pairs.

We can implement this as follows:

.. code-block:: python

    # Step 5: Create an option for each edge
    # The options correspond to single robot actions but in practice
    # they can capture more complex behaviour
    options = []
    for src in graph:
        for dst in graph[src]:
            options.append(
                Option(
                    "e{}{}".format(src, dst), [trans_probs(src, dst)], [reward(src, dst)]
                )
            )

The final step before synthesising our BT is to define our goal condition for planning.
Here, we want the robot to reach node 5. 
We can encode this using *labels*, which are named conditions.
We can create a ``Label`` object which has the name ``goal`` and which holds when the robot reaches node 5:

.. code-block:: python

    # Step 6: Create our goal label
    # The goal label captures reaching node 5
    goal = Label("goal", EqCondition(node_sf, 5))

With this, we can now synthesise a BT using ``synthesise_bt_from_options``, which takes:

1. A list of state factors (only one is required here)

2. A list of options

3. A list of labels

4. An optional initial state. In most problems, this can be set to None

5. A planning objective specified in the `PRISM modelling language`_. Here we minimise the total reward for the robot to reach the goal state, according to our ``goal`` label.

6. A default action. Our planner may not synthesise an action for some states, e.g. the goal state. A default action can be provided for these states.

7. An output file for the BT. The BT is outputted in the `XML format used by BehaviorTree.cpp`.

.. code-block:: python

    # Step 7: Bring everything together and synthesise the refined bt
    synthesise_bt_from_options(
        [node_sf],
        options,
        [goal],
        initial_state=None,
        prism_prop='Rmin=?[F "goal"]',
        default_action="idle",
        out_file="/tmp/bt.xml",
    )

This concludes the tutorial. The BT output by ``synthesise_bt_from_options`` cannot be directly executed BehaviorTree.cpp currently, but should give enough information as to how this could be achieved.
Executable BT XML files will be addressed in the next release.

.. |Graph| image:: images/graph.png
  :width: 400
  :alt: A graph with seven nodes. Connectivity is defined in the pseudocode below.

.. _here: https://github.com/convince-project/refine-plan/blob/main/bin/graph_example.py
.. _PRISM modelling language: https://www.prismmodelchecker.org/manual/PropertySpecification/Introduction
.. _XML format used by BehaviorTree.cpp: https://www.behaviortree.dev/docs/3.8/intro