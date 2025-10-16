#!/usr/bin/env python3
"""A synthetic example of the house wire search domain to compare data collection approaches.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.condition import AndCondition, OrCondition, EqCondition, Label
from refine_plan.learning.option_learning import mongodb_to_yaml, learn_dbns
from refine_plan.algorithms.explore import synthesise_exploration_policy
from refine_plan.algorithms.refine import synthesise_policy
from refine_plan.models.state_factor import StateFactor
from refine_plan.models.dbn_option import DBNOption
from refine_plan.models.semi_mdp import SemiMDP
from refine_plan.models.policy import Policy
from refine_plan.models.state import State
from pymongo import MongoClient
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import yaml
import copy
import sys


GRAPH = {
    "v1": {"e12": "v2", "e13": "v3", "e14": "v4"},
    "v2": {"e12": "v1", "e24": "v4"},
    "v3": {"e13": "v1", "e34": "v4"},
    "v4": {"e14": "v1", "e24": "v2", "e34": "v3", "e45": "v5"},
    "v5": {"e45": "v4", "e56": "v6", "e57": "v7", "e58": "v8", "e59": "v9"},
    "v6": {"e56": "v5", "e67": "v7"},
    "v7": {"e57": "v5", "e67": "v6", "e78": "v8"},
    "v8": {"e58": "v5", "e78": "v7"},
    "v9": {"e59": "v5", "e910": "v10", "e911": "v11"},
    "v10": {"e910": "v9"},
    "v11": {"e911": "v9"},
}


class WireSearchSim(object):
    """Discrete event simulator for the wire search domain.

    Attributes:
        _episode_id: The episode ID for the simulator
        _state: The current state of the system
        _timestep: The current timestep of the simulation
        _mode: Can be 'data' or 'refined'
        _wire_loc: The node the wire is located at in this simulation
        _graph: The topological map
        _costs: The cost dictionary for edge actions
    """

    def __init__(self, episode_id, mode, wire_loc=None):
        """Initialise attributes.

        Args:
            episode_id: The episode ID for the simulator
            mode: The mode of execution
            wire_loc: Overwrite the wire location
        """
        self._episode_id = episode_id
        self._state = self._create_initial_state()
        self._timestep = 0
        self._mode = mode
        self._wire_loc = (
            np.random.choice(["v2", "v7", "v10", "v11"], p=[0.1, 0.15, 0.5, 0.25])
            if wire_loc is None
            else wire_loc
        )
        self._graph = self._create_graph()
        self._costs = self._create_costs()

    def _create_graph(self):
        """Create the topological map."""
        return GRAPH

    def _create_costs(self):
        """Create the cost dictionary."""
        return {
            "e58": 18.961600862336393,
            "e67": 18.941210814744007,
            "e56": 37.31873743187375,
            "e14": 43.032482078839614,
            "e78": 36.859459457837836,
            "e910": 12.813378664507947,
            "e12": 42.52491981040685,
            "e24": 11.611713209891622,
            "e45": 8.615894349436978,
            "e57": 45.636605914695664,
            "e13": 11.870823252449165,
            "e911": 14.433811540653732,
            "e59": 23.447509934830247,
            "e34": 40.17821765050201,
        }

    def _goal_reached(self):
        """Goal termination condition.

        Returns:
            Whether or not the simulation goal has been reached
        """
        goal_reached = (
            self._state["wire_at_v2"] == "yes"
            or self._state["wire_at_v7"] == "yes"
            or self._state["wire_at_v10"] == "yes"
            or self._state["wire_at_v11"] == "yes"
        )
        if self._mode == "data":
            return self._timestep >= 100
        else:
            return goal_reached or self._timestep >= 100

    def _is_enabled(self, state, action):
        """Tests whether an action is enabled in state.

        Args:
            state: The current state
            action: The action to execute

        Returns:
            Whether the action is enabled
        """
        if action == "check_for_wire":
            return (
                state["location"] in ["v2", "v7", "v10", "v11"]
                and state["wire_at_{}".format(state["location"])] == "unknown"
            )
        else:
            return action in self._graph[state["location"]]

    def _create_initial_state(self):
        """Creates the initial state for the simulation.

        Returns:
            The initial state
        """
        loc_sf = StateFactor("location", ["v{}".format(i) for i in range(1, 12)])
        door_sfs = [
            StateFactor("wire_at_v2", ["unknown", "no", "yes"]),
            StateFactor("wire_at_v7", ["unknown", "no", "yes"]),
            StateFactor("wire_at_v10", ["unknown", "no", "yes"]),
            StateFactor("wire_at_v11", ["unknown", "no", "yes"]),
        ]

        state_dict = {loc_sf: "v1"}
        for sf in door_sfs:
            state_dict[sf] = "unknown"

        return State(state_dict)

    def _build_mongo_log(self, state, action, cost, next_state):
        """Build a mongo log for a timestep.

        Args:
            state: The predecessor state
            action: The action executed
            cost: The action cost
            next_state: The next state

        Returns:
            The Mongo log
        """
        doc = {}
        doc["run_id"] = self._episode_id
        doc["option"] = action
        doc["duration"] = cost
        doc["_meta"] = {"inserted_at": datetime.now()}

        for sf in state._state_dict:
            doc["{}0".format(sf)] = state[sf]

        for sf in next_state._state_dict:
            doc["{}t".format(sf)] = next_state[sf]
        return doc

    def _step_forward(self, action):
        """Step forward the simulation one step.

        Args:
            action: The action to execute

        Returns:
            A Mongo log for this step
        """

        if not self._is_enabled(self._state, action):
            raise Exception("Can't execute {} in {}".format(action, self._state))

        new_state_dict = copy.deepcopy(self._state._state_dict)
        cost = 0.0
        if action == "check_for_wire":
            if self._state["location"] == self._wire_loc:
                for wire_loc in ["v2", "v7", "v10", "v11"]:
                    if wire_loc == self._state["location"]:
                        new_state_dict["wire_at_{}".format(wire_loc)] = "yes"
                    else:
                        new_state_dict["wire_at_{}".format(wire_loc)] = "no"
            else:
                new_state_dict["wire_at_{}".format(self._state["location"])] = "no"
                unknowns = []
                for wire_loc in ["v2", "v7", "v10", "v11"]:
                    if new_state_dict["wire_at_{}".format(wire_loc)] == "unknown":
                        unknowns.append("wire_at_{}".format(wire_loc))
                if len(unknowns) == 1:  # Process of elimination, we've found it
                    new_state_dict[unknowns[0]] = "yes"
        else:
            new_loc = self._graph[self._state["location"]][action]
            new_state_dict["location"] = new_loc
            cost = self._costs[action] + np.random.uniform(-0.5, 0.5)

        next_state = {}
        for sf in new_state_dict:
            next_state[self._state._sf_dict[sf]] = new_state_dict[sf]
        next_state = State(next_state)

        log = self._build_mongo_log(self._state, action, cost, next_state)
        # print(
        #    "STATE: {}; TIME: {}; ACTION: {}; COST: {}; NEXT STATE: {}".format(
        #        self._state, self._timestep, action, cost, next_state
        #    )
        # )
        self._state = next_state
        return log

    def run_sim(self, policy_fn):
        """Run a simulation until a goal is satisfied.

        The goal is dependent on self._mode

        Args:
            policy_fn: A function of state and time that returns an action

        Returns:
            The mongo logs for this run
        """
        logs = []
        while not self._goal_reached():
            action = policy_fn(self._state, self._timestep)
            if action is None:
                break
            logs.append(self._step_forward(action))
            self._timestep += 1
        return logs


def get_enabled_cond(sf_list, option):
    """Get the enabled condition for an option.

    Args:
        sf_list: The list of state factors
        option: The option we want the condition for

    Returns:
        The enabled condition for the option
    """
    sf_dict = {sf.get_name(): sf for sf in sf_list}
    wire_locs = ["v{}".format(v) for v in [2, 7, 10, 11]]

    if option == "check_for_wire":
        enabled_cond = OrCondition()

        for loc in wire_locs:
            enabled_cond.add_cond(
                AndCondition(
                    EqCondition(sf_dict["location"], loc),
                    EqCondition(sf_dict["wire_at_{}".format(loc)], "unknown"),
                )
            )
        return enabled_cond
    else:  # Edge navigation action
        enabled_cond = OrCondition()
        for node in GRAPH:
            if option in GRAPH[node]:
                enabled_cond.add_cond(EqCondition(sf_dict["location"], node))
        return enabled_cond


def get_enabled_actions(state):
    """Get the enabled actions for a state.

    Args:
        state: The state to check

    Returns:
        The list of enabled actions
    """
    out_edges = {
        "v1": ["e12", "e13", "e14"],
        "v2": ["e12", "e24"],
        "v3": ["e13", "e34"],
        "v4": ["e14", "e24", "e34", "e45"],
        "v5": ["e45", "e56", "e57", "e58", "e59"],
        "v6": ["e56", "e67"],
        "v7": ["e57", "e67", "e78"],
        "v8": ["e58", "e78"],
        "v9": ["e59", "e910", "e911"],
        "v10": ["e910"],
        "v11": ["e911"],
    }
    loc = state["location"]
    enabled = out_edges[loc]

    if (
        loc in ["v2", "v7", "v10", "v11"]
        and state["wire_at_{}".format(loc)] == "unknown"
    ):
        enabled.append("check_for_wire")

    return enabled


def run_random_data_collection(connection_str):
    """Run 100 episodes of random data collection."""

    client = MongoClient(connection_str)
    db = client["refine-plan-v2"]
    collection = db["fake-wire-random-data"]

    total_logs = 0
    episode_id = 0

    def rand_policy(s, t):
        return np.random.choice(get_enabled_actions(s))

    while total_logs < 10000:
        print(
            "RANDOM DATA COLLECTION, EPISODE: {}, LOGS: {}".format(
                episode_id + 1, total_logs
            )
        )
        sim = WireSearchSim(episode_id, "data")
        logs = sim.run_sim(rand_policy)
        collection.insert_many(logs)
        episode_id += 1
        total_logs += len(logs)


def run_informed_data_collection(connection_str):
    """Run 3 episodes of random data collection and 97 of informed."""

    client = MongoClient(connection_str)
    db = client["refine-plan-v2"]
    collection = db["fake-wire-informed-data"]

    total_logs = 0
    episode_id = 0

    def rand_policy(s, t):
        return np.random.choice(get_enabled_actions(s))

    while total_logs < 10000:
        print(
            "INFORMED DATA COLLECTION, EPISODE: {}, LOGS: {}".format(
                episode_id + 1, total_logs
            )
        )
        sim = WireSearchSim(episode_id, "data")
        if total_logs < 300:
            logs = sim.run_sim(rand_policy)
        else:

            loc_sf = StateFactor("location", ["v{}".format(i) for i in range(1, 12)])
            wire_sfs = [
                StateFactor("wire_at_v2", ["unknown", "no", "yes"]),
                StateFactor("wire_at_v7", ["unknown", "no", "yes"]),
                StateFactor("wire_at_v10", ["unknown", "no", "yes"]),
                StateFactor("wire_at_v11", ["unknown", "no", "yes"]),
            ]
            sf_list = [loc_sf] + wire_sfs

            option_names = set([])
            for src in GRAPH:
                option_names.update(list(GRAPH[src].keys()))
            option_names.add("check_for_wire")

            enabled_conds = {o: get_enabled_cond(sf_list, o) for o in option_names}

            policy = synthesise_exploration_policy(
                connection_str,
                "refine-plan-v2",
                "fake-wire-informed-data",
                sf_list,
                option_names,
                5,
                100,
                enabled_conds,
                initial_state=sim._state,
            )
            if policy[sim._state, 0] is None:  # Stop infinite loops
                break
            logs = sim.run_sim(policy.get_action)
        collection.insert_many(logs)
        episode_id += 1
        total_logs += len(logs)


def build_random_policies(mongo_connection_str):
    """Build the policies for the random data.

    Args:
        mongo_connection_str: The MongoDB conenction string"""

    loc_sf = StateFactor("location", ["v{}".format(i) for i in range(1, 12)])
    wire_sfs = [
        StateFactor("wire_at_v2", ["unknown", "no", "yes"]),
        StateFactor("wire_at_v7", ["unknown", "no", "yes"]),
        StateFactor("wire_at_v10", ["unknown", "no", "yes"]),
        StateFactor("wire_at_v11", ["unknown", "no", "yes"]),
    ]
    sf_list = [loc_sf] + wire_sfs

    for i in range(500, 10001, 500):
        tmp_file = tempfile.NamedTemporaryFile()
        print("Writing Mongo Database to Yaml File for {} Items - Random".format(i))
        mongodb_to_yaml(
            mongo_connection_str,
            "refine-plan-v2",
            "fake-wire-random-data",
            sf_list,
            tmp_file.name,
            query={"run_id": {"$lte": i / 100}},
        )
        print("YAML Dataset Created")

        output_dir = "../data/fake_house/random_{}".format(i)
        print("Learning DBNs")
        learn_dbns(tmp_file.name, output_dir, [loc_sf] + wire_sfs)

        goal_cond = OrCondition()
        wire_locs = ["v2", "v7", "v10", "v11"]
        for j in range(len(wire_locs)):
            goal_cond.add_cond(
                AndCondition(
                    EqCondition(loc_sf, wire_locs[j]), EqCondition(wire_sfs[j], "yes")
                )
            )

        labels = [Label("goal", goal_cond)]

        option_names = set([])
        for src in GRAPH:
            option_names.update(list(GRAPH[src].keys()))
        option_names.add("check_for_wire")

        assert len(set(option_names)) == 15  # Quick safety check

        init_state_dict = {sf: "unknown" for sf in wire_sfs}
        init_state_dict[loc_sf] = "v1"
        init_state = State(init_state_dict)

        option_list = []
        for option in option_names:
            print("Reading in option: {}".format(option))
            t_path = "../data/fake_house/random_{}/{}_transition.bifxml".format(
                i, option
            )
            r_path = "../data/fake_house/random_{}/{}_reward.bifxml".format(i, option)
            option_list.append(
                DBNOption(
                    option, t_path, r_path, sf_list, get_enabled_cond(sf_list, option)
                )
            )

        print("Creating MDP...")
        semi_mdp = SemiMDP(sf_list, option_list, labels, initial_state=init_state)
        print("Synthesising Policy...")
        policy = synthesise_policy(semi_mdp, prism_prop='Rmin=?[F "goal"]')
        policy.write_policy("../data/fake_house/random_{}/policy.yaml".format(i))


def build_informed_policies(mongo_connection_str):
    """Build the policies for the informed data.

    Args:
        mongo_connection_str: The MongoDB conenction string"""

    loc_sf = StateFactor("location", ["v{}".format(i) for i in range(1, 12)])
    wire_sfs = [
        StateFactor("wire_at_v2", ["unknown", "no", "yes"]),
        StateFactor("wire_at_v7", ["unknown", "no", "yes"]),
        StateFactor("wire_at_v10", ["unknown", "no", "yes"]),
        StateFactor("wire_at_v11", ["unknown", "no", "yes"]),
    ]
    sf_list = [loc_sf] + wire_sfs

    for i in range(500, 10001, 500):
        tmp_file = tempfile.NamedTemporaryFile()
        print("Writing Mongo Database to Yaml File for {} Items - Informed".format(i))
        mongodb_to_yaml(
            mongo_connection_str,
            "refine-plan-v2",
            "fake-wire-informed-data",
            sf_list,
            tmp_file.name,
            sort_by="_meta.inserted_at",
            limit=i,
        )
        print("YAML Dataset Created")

        output_dir = "../data/fake_house/informed_{}".format(i)
        print("Learning DBNs")
        learn_dbns(tmp_file.name, output_dir, [loc_sf] + wire_sfs)

        goal_cond = OrCondition()
        wire_locs = ["v2", "v7", "v10", "v11"]
        for j in range(len(wire_locs)):
            goal_cond.add_cond(
                AndCondition(
                    EqCondition(loc_sf, wire_locs[j]), EqCondition(wire_sfs[j], "yes")
                )
            )

        labels = [Label("goal", goal_cond)]

        option_names = set([])
        for src in GRAPH:
            option_names.update(list(GRAPH[src].keys()))
        option_names.add("check_for_wire")

        assert len(set(option_names)) == 15  # Quick safety check

        init_state_dict = {sf: "unknown" for sf in wire_sfs}
        init_state_dict[loc_sf] = "v1"
        init_state = State(init_state_dict)

        option_list = []
        for option in option_names:
            print("Reading in option: {}".format(option))
            t_path = "../data/fake_house/informed_{}/{}_transition.bifxml".format(
                i, option
            )
            r_path = "../data/fake_house/informed_{}/{}_reward.bifxml".format(i, option)
            option_list.append(
                DBNOption(
                    option, t_path, r_path, sf_list, get_enabled_cond(sf_list, option)
                )
            )

        print("Creating MDP...")
        semi_mdp = SemiMDP(sf_list, option_list, labels, initial_state=init_state)
        print("Synthesising Policy...")
        policy = synthesise_policy(semi_mdp, prism_prop='Rmin=?[F "goal"]')
        policy.write_policy("../data/fake_house/informed_{}/policy.yaml".format(i))


def collect_experiment_data():
    """Collect data for each of the checkpointed policies (random and informed)."""

    results = {"random": {}, "informed": {}}
    success_rates = {"random": {}, "informed": {}}

    wire_locs = []
    for _ in range(100):
        wire_locs.append(
            np.random.choice(["v2", "v7", "v10", "v11"], p=[0.1, 0.15, 0.5, 0.25])
        )

    for method in ["random", "informed"]:
        for checkpoint in range(500, 10001, 500):
            print("EXPERIMENTS: {}; {}".format(method, checkpoint))
            results[method][checkpoint] = []
            policy = Policy(
                {},
                {},
                policy_file="../data/fake_house/{}_{}/policy.yaml".format(
                    method, checkpoint
                ),
            )
            success_rate = 0
            for i in range(100):
                sim = WireSearchSim(0, "refined", wire_loc=wire_locs[i])
                logs = sim.run_sim(lambda s, t: policy.get_action(s))
                if len(logs) < 100:
                    success_rate += 1
                    time_to_goal = sum([l["duration"] for l in logs])
                    results[method][checkpoint].append(time_to_goal)
            success_rates[method][checkpoint] = success_rate / 100

    result_file = "../data/fake_house/results.yaml"
    with open(result_file, "w") as yaml_out:
        yaml.dump(results, yaml_out)

    succ_file = "../data/fake_house/success_rates.yaml"
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

    results_file = "../data/fake_house/results.yaml"
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
    # collect_experiment_data()
    plot_results()
