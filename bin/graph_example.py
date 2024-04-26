#!/usr/bin/env python3
""" A small graph search BT refinement example as documentation.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.algorithms.refine import synthesise_bt_from_options
from refine_plan.models.condition import Label, EqCondition
from refine_plan.models.state_factor import IntStateFactor
from refine_plan.models.option import Option


# Step 1: Build our state factor for the graph
# The state factor captures the robot's current node
node_sf = IntStateFactor("node_sf", 0, 6)

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

# Step 6: Create our goal label
# The goal label captures reaching node 5
goal = Label("goal", EqCondition(node_sf, 5))

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
