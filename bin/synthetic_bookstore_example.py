#!/usr/bin/env python3
""" A synthetic version of the bookstore example to test the full REFINE-PLAN pipeline.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.condition import Label, EqCondition, AndCondition, OrCondition
from refine_plan.learning.option_learning import mongodb_to_yaml, learn_dbns
from refine_plan.algorithms.semi_mdp_solver import synthesise_policy
from refine_plan.algorithms.refine import synthesise_bt_from_options
from refine_plan.models.state_factor import StateFactor
from refine_plan.models.dbn_option import DBNOption
from refine_plan.models.semi_mdp import SemiMDP
from refine_plan.models.option import Option
from refine_plan.models.state import State
from scipy.stats import mannwhitneyu
from pymongo import MongoClient
from datetime import datetime
import numpy as np
import itertools
import random
import copy

# Global map setup

GRAPH = {
    "v1": {"e12": "v2", "e13": "v3", "e14": "v4"},
    "v2": {"e12": "v1", "e23": "v3", "e25": "v5", "e26": "v6"},
    "v3": {
        "e13": "v1",
        "e23": "v2",
        "e34": "v4",
        "e35": "v5",
        "e36": "v6",
        "e37": "v7",
    },
    "v4": {"e14": "v1", "e34": "v3", "e46": "v6", "e47": "v7"},
    "v5": {"e25": "v2", "e35": "v3", "e56": "v6", "e58": "v8"},
    "v6": {
        "e26": "v2",
        "e36": "v3",
        "e46": "v4",
        "e56": "v5",
        "e67": "v7",
        "e68": "v8",
    },
    "v7": {
        "e37": "v3",
        "e47": "v4",
        "e67": "v6",
        "e78": "v8",
    },
    "v8": {"e58": "v5", "e68": "v6", "e78": "v7"},
}

CORRESPONDING_DOOR = {
    "e12": None,
    "e14": None,
    "e58": "v5",
    "e78": "v7",
    "e13": None,
    "e36": "v3",
    "e68": "v6",
    "e25": "v2",
    "e47": "v4",
    "e26": "v2",
    "e35": "v3",
    "e46": "v4",
    "e37": "v3",
    "e23": None,
    "e34": None,
    "e56": None,
    "e67": None,
}

EDGE_MEANS = {
    "e12": 8,
    "e14": 8,
    "e58": 8,
    "e78": 8,
    "e13": 6,
    "e36": 6,
    "e68": 6,
    "e25": 6,
    "e47": 6,
    "e26": 7,
    "e35": 7,
    "e46": 7,
    "e37": 7,
    "e23": 3,
    "e34": 3,
    "e56": 3,
    "e67": 3,
}

# In door_dists, the distribution is [closed, open]
DOOR_OPEN_COST = 6
DOOR_CHECK_COST = 0
DOOR_DISTS = {
    "v2": [0.4, 0.6],
    "v3": [0.8, 0.2],
    "v4": [0.6, 0.4],
    "v5": [0.4, 0.6],
    "v6": [0.8, 0.2],
    "v7": [0.6, 0.4],
}

# Problem Setup
INITIAL_LOC = "v1"
GOAL_LOC = "v8"


def _log_data(logs, mongo_collection):
    """Log the simulation data for option learning.

    Args:
        logs: A list of (s, a, s', r) tuples
        mongo_collection: A mongoDB collection for the data
    """

    docs = []
    run_id = random.getrandbits(32)

    for log in logs:
        state, option, next_state, cost = log

        doc = {}
        doc["run_id"] = run_id
        doc["option"] = option
        doc["duration"] = cost
        doc["_meta"] = {"inserted_at": datetime.now()}

        for sf in state._state_dict:
            doc["{}0".format(sf)] = state[sf]

        for sf in next_state._state_dict:
            doc["{}t".format(sf)] = next_state[sf]

        docs.append(doc)

    # Insert into the DB
    mongo_collection.insert_many(docs)


def _create_initial_state():
    """Creates the initial state for the simulation.

    Returns:
        The initial state
    """
    loc_sf = StateFactor("location", ["v{}".format(i) for i in range(1, 9)])
    door_sfs = [
        StateFactor("v2_door", ["unknown", "closed", "open"]),
        StateFactor("v3_door", ["unknown", "closed", "open"]),
        StateFactor("v4_door", ["unknown", "closed", "open"]),
        StateFactor("v5_door", ["unknown", "closed", "open"]),
        StateFactor("v6_door", ["unknown", "closed", "open"]),
        StateFactor("v7_door", ["unknown", "closed", "open"]),
    ]

    state_dict = {loc_sf: INITIAL_LOC}
    for sf in door_sfs:
        state_dict[sf] = "unknown"

    return State(state_dict)


def _step_forward(state, option):
    """Run forward one step of the simulation.

    Non-enabled actions stay in the same state with cost 0.

    Args:
        state: The current state
        option: The option to execute

    Returns:
        The next state and immediate cost
    """
    doors_at = ["v2", "v3", "v4", "v5", "v6", "v7"]

    if option == "check_door":
        for door_loc in doors_at:
            door_sf = "{}_door".format(door_loc)
            if state["location"] == door_loc and state[door_sf] == "unknown":
                # Sample closed open
                closed = np.random.choice(
                    ["closed", "open"], p=DOOR_DISTS[state["location"]]
                )
                next_state_dict = copy.deepcopy(state._state_dict)
                next_state_dict[door_sf] = closed
                next_state_dict = {
                    state._sf_dict[sf]: next_state_dict[sf] for sf in next_state_dict
                }
                # Update state
                return State(next_state_dict), DOOR_CHECK_COST
        return copy.deepcopy(state), 0.0

    elif option == "open_door":
        for door_loc in doors_at:
            door_sf = "{}_door".format(door_loc)
            if state["location"] == door_loc and state[door_sf] == "closed":
                next_state_dict = copy.deepcopy(state._state_dict)
                next_state_dict[door_sf] = "open"
                next_state_dict = {
                    state._sf_dict[sf]: next_state_dict[sf] for sf in next_state_dict
                }
                # Update state
                return State(next_state_dict), DOOR_OPEN_COST
        return copy.deepcopy(state), 0.0

    else:  # Edge option
        if option in GRAPH[state["location"]]:
            # Check if door needs to be opened
            can_nav = True
            if CORRESPONDING_DOOR[option] != None:
                door_sf = "{}_door".format(CORRESPONDING_DOOR[option])
                if state[door_sf] != "open":
                    can_nav = False
            if can_nav:  # Update location and sample cost
                next_state_dict = copy.deepcopy(state._state_dict)
                next_state_dict["location"] = GRAPH[state["location"]][option]
                next_state_dict = {
                    state._sf_dict[sf]: next_state_dict[sf] for sf in next_state_dict
                }
                return State(next_state_dict), EDGE_MEANS[option] + np.random.uniform(
                    -0.5, 0.5
                )

        return copy.deepcopy(state), 0.0


def run_sim(
    policy_fn,
    max_timesteps=100,
    mongo_collection=None,
    print_info=False,
    stop_at_goal=True,
):
    """Run a simulation until the robot reaches the goal or max timesteps exceeded.

    Args:
        policy_fn: A function which takes a state and returns an action.
        max_timesteps: The maximum timesteps to run the simulation
        mongo_collection: A MongoDB collection if logging required
        print_info: Whether to print out the s,a,s',r transitions
        stop_at_goal: Should the sim stop when the goal is reached?

    Returns:
        A flag stating whether the goal was reached and the cumulative cost
    """

    current_state = _create_initial_state()
    t = 0
    total_cost = 0
    logs = []

    while (
        not stop_at_goal or current_state["location"] != GOAL_LOC
    ) and t < max_timesteps:

        option = policy_fn(current_state)
        # print("STATE: {}; OPTION: {}".format(current_state, option))
        next_state, cost = _step_forward(current_state, option)

        logs.append((current_state, option, next_state, cost))
        total_cost += cost

        if print_info:
            print(
                "S: {}; A: {}; S': {}; R: {}".format(
                    current_state, option, next_state, cost
                )
            )

        current_state = next_state
        t += 1

    if mongo_collection is not None:
        _log_data(logs, mongo_collection)

    return current_state["location"] == GOAL_LOC, total_cost


def initial_bt(state):
    """A spoof of the initial BT.

    The initial BT follows the shortest path and goes through any door it sees.

    Args:
        state: The current state of the system
    """

    if state["location"] == "v1":
        return "e13"
    elif state["location"] == "v3" and state["v3_door"] == "unknown":
        return "check_door"
    elif state["location"] == "v3" and state["v3_door"] == "closed":
        return "open_door"
    elif state["location"] == "v3" and state["v3_door"] == "open":
        return "e36"
    elif state["location"] == "v6" and state["v6_door"] == "unknown":
        return "check_door"
    elif state["location"] == "v6" and state["v6_door"] == "closed":
        return "open_door"
    elif state["location"] == "v6" and state["v6_door"] == "open":
        return "e68"
    else:
        return None


def _generate_exact_options(sf_list):
    """Generate the exact options for the exact model.

    Args:
        sf_list: The list of state factors

    Returns:
        A list of options
    """
    sf_dict = {sf.get_name(): sf for sf in sf_list}
    option_list = []

    # Check door and open door
    check_trans = []
    open_trans = []
    open_rewards = [(_get_enabled_cond(sf_list, "open_door"), DOOR_OPEN_COST)]
    door_locs = ["v{}".format(i) for i in range(2, 8)]
    for door in door_locs:
        loc_cond = EqCondition(sf_dict["location"], door)
        unknown = EqCondition(sf_dict["{}_door".format(door)], "unknown")
        closed = EqCondition(sf_dict["{}_door".format(door)], "closed")
        d_open = EqCondition(sf_dict["{}_door".format(door)], "open")

        open_trans.append((AndCondition(loc_cond, closed), {d_open: 1.0}))
        check_trans.append(
            (
                AndCondition(loc_cond, unknown),
                {closed: DOOR_DISTS[door][0], d_open: DOOR_DISTS[door][1]},
            )
        )

    option_list.append(Option("check_door", check_trans, []))
    option_list.append(Option("open_door", open_trans, open_rewards))

    edge_trans = {}

    for node in GRAPH:
        for edge in GRAPH[node]:
            pre_cond = EqCondition(sf_dict["location"], node)
            post_cond = EqCondition(sf_dict["location"], GRAPH[node][edge])
            door = CORRESPONDING_DOOR[edge]
            if door != None:
                door_cond = EqCondition(sf_dict["{}_door".format(door)], "open")
                pre_cond = AndCondition(pre_cond, door_cond)

            if edge not in edge_trans:
                edge_trans[edge] = []
            edge_trans[edge].append((pre_cond, {post_cond: 1.0}))

    for edge in edge_trans:
        r_list = [(_get_enabled_cond(sf_list, edge), EDGE_MEANS[edge])]
        option_list.append(Option(edge, edge_trans[edge], r_list))

    return option_list


def generate_optimal_bt():
    """Create the true semi-MDP and solve it to obtain an optimal policy.

    Returns:
        The optimal BT (policy)
    """
    loc_sf = StateFactor("location", ["v{}".format(i) for i in range(1, 9)])
    door_sfs = [
        StateFactor("v2_door", ["unknown", "closed", "open"]),
        StateFactor("v3_door", ["unknown", "closed", "open"]),
        StateFactor("v4_door", ["unknown", "closed", "open"]),
        StateFactor("v5_door", ["unknown", "closed", "open"]),
        StateFactor("v6_door", ["unknown", "closed", "open"]),
        StateFactor("v7_door", ["unknown", "closed", "open"]),
    ]
    sf_list = [loc_sf] + door_sfs

    labels = [Label("goal", EqCondition(loc_sf, "v8"))]

    init_state_dict = {sf: "unknown" for sf in door_sfs}
    init_state_dict[loc_sf] = "v1"
    init_state = State(init_state_dict)

    option_list = _generate_exact_options(sf_list)

    semi_mdp = SemiMDP(sf_list, option_list, labels, initial_state=init_state)

    print("Synthesising Optimal Policy...")
    return synthesise_policy(semi_mdp, prism_prop='Rmin=?[F "goal"]')


def _enabled_actions(state):
    """Return the enabled actions in a state.

    Args:
        state: The current state

    Returns:
        A list of enabled actions
    """
    enabled_actions = set([])

    door_locs = ["v{}".format(i) for i in range(2, 8)]
    current_loc = state["location"]

    # Door actions
    for loc in door_locs:
        if current_loc == loc:
            if state["{}_door".format(loc)] == "closed":
                enabled_actions.add("open_door")
            elif state["{}_door".format(loc)] == "unknown":
                enabled_actions.add("check_door")

    # Navigation
    for edge in GRAPH[current_loc]:
        if CORRESPONDING_DOOR[edge] == None:  # No door to worry about
            enabled_actions.add(edge)
        elif state["{}_door".format(CORRESPONDING_DOOR[edge])] == "open":
            enabled_actions.add(edge)

    return list(enabled_actions)


def _get_enabled_cond(sf_list, option):
    """Get the enabled condition for an option.

    Args:
        sf_list: The list of state factors
        option: The option we want the condition for

    Returns:
        The enabled condition for the option
    """
    sf_dict = {sf.get_name(): sf for sf in sf_list}

    door_locs = ["v{}".format(i) for i in range(2, 8)]

    if option == "check_door" or option == "open_door":
        enabled_cond = OrCondition()
        door_status = "unknown" if option == "check_door" else "closed"
        for door in door_locs:
            enabled_cond.add_cond(
                AndCondition(
                    EqCondition(sf_dict["location"], door),
                    EqCondition(sf_dict["{}_door".format(door)], door_status),
                )
            )
        return enabled_cond
    else:  # edge navigation option
        enabled_cond = OrCondition()
        for node in GRAPH:
            if option in GRAPH[node]:
                enabled_cond.add_cond(EqCondition(sf_dict["location"], node))
        door = CORRESPONDING_DOOR[option]
        if door != None:
            enabled_cond = AndCondition(
                enabled_cond, EqCondition(sf_dict["{}_door".format(door)], "open")
            )
        return enabled_cond


def enabled_cond_test():
    """Test that the enabled actions function matches the enabled conditions."""
    loc_sf = StateFactor("location", ["v{}".format(i) for i in range(1, 9)])
    door_sfs = [
        StateFactor("v2_door", ["unknown", "closed", "open"]),
        StateFactor("v3_door", ["unknown", "closed", "open"]),
        StateFactor("v4_door", ["unknown", "closed", "open"]),
        StateFactor("v5_door", ["unknown", "closed", "open"]),
        StateFactor("v6_door", ["unknown", "closed", "open"]),
        StateFactor("v7_door", ["unknown", "closed", "open"]),
    ]
    sf_list = [loc_sf] + door_sfs

    option_names = [
        "e12",
        "e14",
        "e58",
        "e78",
        "e13",
        "e36",
        "e68",
        "e25",
        "e47",
        "e26",
        "e35",
        "e46",
        "e37",
        "e23",
        "e34",
        "e56",
        "e67",
        "check_door",
        "open_door",
    ]

    state_values = [sf.get_valid_values() for sf in sf_list]

    for option in option_names:
        print("Testing Option: {}".format(option))

        enabled_cond = _get_enabled_cond(sf_list, option)
        print("Enabled Condition: {}".format(enabled_cond))

        for state_tuple in itertools.product(*state_values):
            state = State({sf_list[i]: state_tuple[i] for i in range(len(sf_list))})

            fn_result = option in _enabled_actions(state)
            cond_result = enabled_cond.is_satisfied(state)

            if fn_result != cond_result:
                raise Exception(
                    "INCONSISTENCY AT STATE {} - FN: {}, COND: {}.".format(
                        state, fn_result, cond_result
                    )
                )

    print("***Enabled conditions consistent with enabled actions function***")


def random_policy(state):
    """A random policy for data collection.

    Args:
        state: The current state of the system
    """
    return np.random.choice(_enabled_actions(state))


def run_data_collection():
    """Run the data collection using a random data collection policy."""
    client = MongoClient("localhost:27017")
    collection = client["refine-plan"]["synthetic-bookstore"]

    i = 1
    while collection.estimated_document_count() < 10000:
        print("Starting Simulation Run: {}".format(i))
        run_sim(random_policy, mongo_collection=collection, stop_at_goal=False)
        i += 1


def write_mongodb_to_yaml():
    """Learn the DBNOptions from the database."""

    loc_sf = StateFactor("location", ["v{}".format(i) for i in range(1, 9)])
    door_sfs = [
        StateFactor("v2_door", ["unknown", "closed", "open"]),
        StateFactor("v3_door", ["unknown", "closed", "open"]),
        StateFactor("v4_door", ["unknown", "closed", "open"]),
        StateFactor("v5_door", ["unknown", "closed", "open"]),
        StateFactor("v6_door", ["unknown", "closed", "open"]),
        StateFactor("v7_door", ["unknown", "closed", "open"]),
    ]

    print("Writing mongo database to yaml file")
    mongodb_to_yaml(
        "localhost:27017",
        "refine-plan",
        "synthetic-bookstore",
        [loc_sf] + door_sfs,
        "../data/synthetic_bookstore/dataset.yaml",
    )


def learn_options():
    """Learn the options from the YAML file."""
    dataset_path = "../data/synthetic_bookstore/dataset.yaml"
    output_dir = "../data/synthetic_bookstore/"

    loc_sf = StateFactor("location", ["v{}".format(i) for i in range(1, 9)])
    door_sfs = [
        StateFactor("v2_door", ["unknown", "closed", "open"]),
        StateFactor("v3_door", ["unknown", "closed", "open"]),
        StateFactor("v4_door", ["unknown", "closed", "open"]),
        StateFactor("v5_door", ["unknown", "closed", "open"]),
        StateFactor("v6_door", ["unknown", "closed", "open"]),
        StateFactor("v7_door", ["unknown", "closed", "open"]),
    ]

    learn_dbns(dataset_path, output_dir, [loc_sf] + door_sfs)


def run_planner():
    """Run refine-plan and synthesise a BT.

    Returns:
        The refined BT
    """

    loc_sf = StateFactor("location", ["v{}".format(i) for i in range(1, 9)])
    door_sfs = [
        StateFactor("v2_door", ["unknown", "closed", "open"]),
        StateFactor("v3_door", ["unknown", "closed", "open"]),
        StateFactor("v4_door", ["unknown", "closed", "open"]),
        StateFactor("v5_door", ["unknown", "closed", "open"]),
        StateFactor("v6_door", ["unknown", "closed", "open"]),
        StateFactor("v7_door", ["unknown", "closed", "open"]),
    ]
    sf_list = [loc_sf] + door_sfs

    labels = [Label("goal", EqCondition(loc_sf, "v8"))]

    option_names = [
        "e12",
        "e14",
        "e58",
        "e78",
        "e13",
        "e36",
        "e68",
        "e25",
        "e47",
        "e26",
        "e35",
        "e46",
        "e37",
        "e23",
        "e34",
        "e56",
        "e67",
        "check_door",
        "open_door",
    ]

    assert len(set(option_names)) == 19  # Quick safety check

    init_state_dict = {sf: "unknown" for sf in door_sfs}
    init_state_dict[loc_sf] = "v1"
    init_state = State(init_state_dict)

    option_list = []
    for option in option_names:
        print("Reading in option: {}".format(option))
        t_path = "../data/synthetic_bookstore/{}_transition.bifxml".format(option)
        r_path = "../data/synthetic_bookstore/{}_reward.bifxml".format(option)
        option_list.append(
            DBNOption(
                option, t_path, r_path, sf_list, _get_enabled_cond(sf_list, option)
            )
        )

    print("Synthesising BT...")
    return synthesise_bt_from_options(
        sf_list,
        option_list,
        labels,
        prism_prop='Rmin=?[F "goal"]',
        initial_state=init_state,
        bt_converter="simple",
        out_file="synthetic_bookstore_bt.xml",
    )


def initial_optimal_refined_comparison(refined_bt):
    """Compare the initial BT to the refined BT and the optimal BT.

    Args:
        refined_bt: The refined BT
    """
    initial_bt_costs = []
    optimal_bt_costs = []
    refined_bt_costs = []

    optimal_bt = generate_optimal_bt()

    for i in range(10000):
        print("Experiment Repeat {}".format(i))

        print("WITH INITIAL BT")
        goal_reached, cost = run_sim(initial_bt)
        if not goal_reached:
            print("GOAL NOT REACHED INITIAL BT")
        else:
            initial_bt_costs.append(cost)

        print("WITH OPTIMAL BT")
        goal_reached, cost = run_sim(optimal_bt.get_action)
        if not goal_reached:
            print("GOAL NOT REACHED OPTIMAL BT")
        else:
            optimal_bt_costs.append(cost)

        print("WITH REFINED BT")
        goal_reached, cost = run_sim(refined_bt.tick_at_state)
        if not goal_reached:
            print("GOAL NOT REACHED REFINED BT")
        else:
            refined_bt_costs.append(cost)

    print(
        "INITIAL BT SUCCESSES: {}; AVG COST: {}; VARIANCE: {}".format(
            len(initial_bt_costs), np.mean(initial_bt_costs), np.var(initial_bt_costs)
        )
    )
    print(
        "OPTIMAL BT SUCCESSES: {}; AVG COST: {}; VARIANCE: {}".format(
            len(optimal_bt_costs), np.mean(optimal_bt_costs), np.var(optimal_bt_costs)
        )
    )
    print(
        "REFINED BT SUCCESSES: {}; AVG COST: {}; VARIANCE: {}".format(
            len(refined_bt_costs), np.mean(refined_bt_costs), np.var(refined_bt_costs)
        )
    )

    non_none = 0
    for s in optimal_bt._state_action_dict:
        if optimal_bt[s] != None:
            non_none += 1
    print("OPTIMAL NON NONE: {}".format(non_none))
    print("OPTIMAL POLICY SIZE: {}".format(len(optimal_bt._state_action_dict)))

    # non_none = 0
    # for s in refined_bt._state_action_dict:
    #     if refined_bt[s] != None:
    #         non_none += 1
    # print("REFINED NON NONE: {}".format(non_none))
    # print("REFINED POLICY SIZE: {}".format(len(refined_bt._state_action_dict)))

    print("*** STATISTICAL SIGNIFICANCE RESULTS ***")
    p = mannwhitneyu(
        refined_bt_costs,
        initial_bt_costs,
        alternative="less",
    )[1]
    print(
        "REFINED BT BETTER THAN INITIAL BT: p = {}, stat sig better = {}".format(
            p, p < 0.05
        )
    )

    p = mannwhitneyu(
        optimal_bt_costs,
        refined_bt_costs,
        alternative="less",
    )[1]
    print(
        "OPTIMAL BT BETTER THAN REFINED BT: p = {}, stat sig better = {}".format(
            p, p < 0.05
        )
    )


if __name__ == "__main__":

    # enabled_cond_test()
    # run_data_collection()
    # write_mongodb_to_yaml()
    # learn_options()
    bt = run_planner()
    initial_optimal_refined_comparison(bt)
