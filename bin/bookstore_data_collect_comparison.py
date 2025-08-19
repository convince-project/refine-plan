#!/usr/bin/env python3
"""An alternative data collection comparison on the bookstore domain.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.condition import Label, EqCondition, AndCondition, OrCondition
from refine_plan.learning.option_learning import mongodb_to_yaml, learn_dbns
from refine_plan.algorithms.explore import synthesise_exploration_policy
from refine_plan.algorithms.semi_mdp_solver import synthesise_policy
from refine_plan.models.state_factor import StateFactor
from refine_plan.models.dbn_option import DBNOption
from refine_plan.models.semi_mdp import SemiMDP
from refine_plan.models.policy import Policy
from refine_plan.models.state import State
import matplotlib.pyplot as plt
from pymongo import MongoClient
from datetime import datetime
import numpy as np
import tempfile
import yaml
import copy
import sys

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


def build_mongo_logs(logs, episode_id):
    """Build the MongoDB logs for a run.

    Args:
        logs: A list of (s, a, s', r) tuples
        episode_id: The episode ID
    """

    docs = []
    for log in logs:
        state, option, next_state, cost = log

        doc = {}
        doc["run_id"] = episode_id
        doc["option"] = option
        doc["duration"] = cost
        doc["_meta"] = {"inserted_at": datetime.now()}

        for sf in state._state_dict:
            doc["{}0".format(sf)] = state[sf]

        for sf in next_state._state_dict:
            doc["{}t".format(sf)] = next_state[sf]

        docs.append(doc)

    return docs


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
    episode_id,
    max_timesteps=100,
    print_info=True,
    stop_at_goal=True,
):
    """Run a simulation until the robot reaches the goal or max timesteps exceeded.

    Args:
        policy_fn: A function which takes a state and returns an action.
        max_timesteps: The maximum timesteps to run the simulation
        print_info: Whether to print out the s,a,s',r transitions
        stop_at_goal: Should the sim stop when the goal is reached?

    Returns:
        The logs for that run
    """

    current_state = _create_initial_state()
    t = 0
    total_cost = 0
    logs = []

    while (
        not stop_at_goal or current_state["location"] != GOAL_LOC
    ) and t < max_timesteps:

        option = policy_fn(current_state, t)
        if option is None:
            break
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

    return build_mongo_logs(logs, episode_id)


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


def run_random_data_collection(connection_str):
    """Run 100 episodes of random data collection."""

    client = MongoClient(connection_str)
    db = client["refine-plan-v2"]
    collection = db["fake-bookstore-random-data"]

    total_logs = 0
    episode_id = 0

    def rand_policy(s, t):
        return np.random.choice(_enabled_actions(s))

    while total_logs < 10000:
        print(
            "RANDOM DATA COLLECTION, EPISODE: {}, LOGS: {}".format(
                episode_id + 1, total_logs
            )
        )
        logs = run_sim(rand_policy, episode_id, stop_at_goal=False)
        collection.insert_many(logs)
        episode_id += 1
        total_logs += len(logs)


def run_informed_data_collection(connection_str):
    """Run 3 episodes of random data collection and 97 of informed."""

    client = MongoClient(connection_str)
    db = client["refine-plan-v2"]
    collection = db["fake-bookstore-informed-data"]

    total_logs = 0
    episode_id = 0

    def rand_policy(s, t):
        return np.random.choice(_enabled_actions(s))

    while total_logs < 10000:
        print(
            "INFORMED DATA COLLECTION, EPISODE: {}, LOGS: {}".format(
                episode_id + 1, total_logs
            )
        )
        if total_logs < 300:
            logs = run_sim(rand_policy, episode_id, stop_at_goal=False)
        else:
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

            assert len(set(option_names)) == 19  # Quick safety check

            init_state_dict = {sf: "unknown" for sf in door_sfs}
            init_state_dict[loc_sf] = "v1"
            init_state = State(init_state_dict)

            enabled_conds = {o: _get_enabled_cond(sf_list, o) for o in option_names}

            policy = synthesise_exploration_policy(
                connection_str,
                "refine-plan-v2",
                "fake-bookstore-informed-data",
                sf_list,
                option_names,
                5,
                100,
                enabled_conds,
                initial_state=init_state,
            )
            if policy[init_state, 0] is None:  # Stop infinite loops
                print("EXPLORATION NO LONGER NEEDED")
                break
            logs = run_sim(policy.get_action, episode_id, stop_at_goal=False)
        collection.insert_many(logs)
        episode_id += 1
        total_logs += len(logs)


def build_random_policies(mongo_connection_str):
    """Build the policies for the random data.

    Args:
        mongo_connection_str: The MongoDB conenction string"""

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

    for i in range(500, 10001, 500):
        tmp_file = tempfile.NamedTemporaryFile()
        print("Writing Mongo Database to Yaml File for {} Items - Random".format(i))
        mongodb_to_yaml(
            mongo_connection_str,
            "refine-plan-v2",
            "fake-bookstore-random-data",
            sf_list,
            tmp_file.name,
            query={"run_id": {"$lte": i / 100}},
        )
        print("YAML Dataset Created")

        output_dir = "../data/fake_bookstore/random_{}".format(i)
        print("Learning DBNs")
        learn_dbns(tmp_file.name, output_dir, sf_list)

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
            t_path = "../data/fake_bookstore/random_{}/{}_transition.bifxml".format(
                i, option
            )
            r_path = "../data/fake_bookstore/random_{}/{}_reward.bifxml".format(
                i, option
            )
            option_list.append(
                DBNOption(
                    option, t_path, r_path, sf_list, _get_enabled_cond(sf_list, option)
                )
            )

        print("Creating MDP...")
        semi_mdp = SemiMDP(sf_list, option_list, labels, initial_state=init_state)
        print("Synthesising Policy...")
        policy = synthesise_policy(semi_mdp, prism_prop='Rmin=?[F "goal"]')
        policy.write_policy("../data/fake_bookstore/random_{}/policy.yaml".format(i))


def build_informed_policies(mongo_connection_str):
    """Build the policies for the informed data.

    Args:
        mongo_connection_str: The MongoDB conenction string"""

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

    for i in range(500, 4501, 500):
        tmp_file = tempfile.NamedTemporaryFile()
        print("Writing Mongo Database to Yaml File for {} Items - Informed".format(i))
        mongodb_to_yaml(
            mongo_connection_str,
            "refine-plan-v2",
            "fake-bookstore-informed-data",
            sf_list,
            tmp_file.name,
            query={"run_id": {"$lte": i / 100}},
        )
        print("YAML Dataset Created")

        output_dir = "../data/fake_bookstore/informed_{}".format(i)
        print("Learning DBNs")
        learn_dbns(tmp_file.name, output_dir, sf_list)

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
            t_path = "../data/fake_bookstore/informed_{}/{}_transition.bifxml".format(
                i, option
            )
            r_path = "../data/fake_bookstore/informed_{}/{}_reward.bifxml".format(
                i, option
            )
            option_list.append(
                DBNOption(
                    option, t_path, r_path, sf_list, _get_enabled_cond(sf_list, option)
                )
            )

        print("Creating MDP...")
        semi_mdp = SemiMDP(sf_list, option_list, labels, initial_state=init_state)
        print("Synthesising Policy...")
        policy = synthesise_policy(semi_mdp, prism_prop='Rmin=?[F "goal"]')
        policy.write_policy("../data/fake_bookstore/informed_{}/policy.yaml".format(i))


def collect_experiment_data():
    """Collect data for each of the checkpointed policies (random and informed)."""

    results = {"random": {}, "informed": {}}
    success_rates = {"random": {}, "informed": {}}

    for method in ["random", "informed"]:
        for checkpoint in range(500, 10001, 500):
            print("EXPERIMENTS: {}; {}".format(method, checkpoint))
            if method == "informed" and checkpoint > 4500:  # TODO: TEMPORARY
                results[method][checkpoint] = [0] * 100
                success_rates[method][checkpoint] = 1.0
                continue
            results[method][checkpoint] = []
            policy = Policy(
                {},
                {},
                policy_file="../data/fake_bookstore/{}_{}/policy.yaml".format(
                    method, checkpoint
                ),
            )
            success_rate = 0
            for i in range(100):
                logs = run_sim(lambda s, t: policy.get_action(s), 0, stop_at_goal=True)
                if len(logs) < 100:
                    success_rate += 1
                    time_to_goal = sum([l["duration"] for l in logs])
                    results[method][checkpoint].append(time_to_goal)
            success_rates[method][checkpoint] = success_rate / 100

    result_file = "../data/fake_bookstore/results.yaml"
    with open(result_file, "w") as yaml_out:
        yaml.dump(results, yaml_out)

    succ_file = "../data/fake_bookstore/success_rates.yaml"
    with open(succ_file, "w") as yaml_out:
        yaml.dump(success_rates, yaml_out)


def set_box_colors(bp):

    count_1 = 0
    count_2 = 0

    for _ in range(40):
        colour = "tab:blue" if count_1 % 2 == 0 else "tab:red"

        plt.setp(bp["boxes"][count_1], color=colour, linewidth=4.0)
        plt.setp(bp["caps"][count_2], color=colour, linewidth=4.0)
        plt.setp(bp["caps"][count_2 + 1], color=colour, linewidth=4.0)
        plt.setp(bp["whiskers"][count_2], color=colour, linewidth=4.0)
        plt.setp(bp["whiskers"][count_2 + 1], color=colour, linewidth=4.0)
        plt.setp(bp["medians"][count_1], color=colour, linewidth=4.0)
        count_1 += 1
        count_2 += 2


def plot_results():
    """Plot the results of the random vs informed data collection experiment

    Args:
        results: method to env to results list
        env: The environment name to plot
    """

    results_file = "../data/fake_bookstore/results.yaml"
    with open(results_file, "r") as yaml_in:
        results = yaml.load(yaml_in, Loader=yaml.FullLoader)

    checks = list(range(500, 10001, 500))

    results_list = []
    for check in checks:
        for method in ["random", "informed"]:
            print(
                "{} ITEMS, {} DATA COLLECTION, MEAN: {}".format(
                    check, method, np.mean(results[method][check])
                )
            )
            results_list.append(results[method][check])

    box = plt.boxplot(
        results_list,
        whis=[0, 100],
        positions=list(range(1, 41)),
        widths=0.6,
    )
    set_box_colors(box)

    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=True,  # labels along the bottom edge are offcd
        labelsize=24,
    )
    plt.tick_params(
        axis="y",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=True,  # labels along the bottom edge are offcd
        labelsize=24,
    )
    plt.ylabel("Time to Find Wire (s)", fontsize=40)

    plt.xticks(
        [x + 0.5 for x in range(1, 40, 2)],
        checks,
    )
    plt.xlabel("Data Items Collected", fontsize=40)

    (hB,) = plt.plot([], color="tab:blue", linewidth=6.0)
    (hR,) = plt.plot([], color="tab:red", linewidth=6.0)
    plt.legend(
        (hB, hR),
        ("Random Collection", "Informed Collection"),
        loc="upper right",
        prop={"size": 24},
    )

    plt.show()


if __name__ == "__main__":
    # connection_str = sys.argv[1]
    # run_random_data_collection(connection_str)
    # run_informed_data_collection(connection_str)
    # build_random_policies(connection_str)
    # build_informed_policies(connection_str)
    collect_experiment_data()
    plot_results()
