#!/usr/bin/env python3
""" A script to run REFINE-PLAN in the house domain.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.condition import Label, EqCondition, AndCondition, OrCondition
from refine_plan.learning.option_learning import mongodb_to_yaml, learn_dbns
from refine_plan.algorithms.semi_mdp_solver import synthesise_policy
from refine_plan.models.state_factor import StateFactor
from refine_plan.models.dbn_option import DBNOption
from refine_plan.models.semi_mdp import SemiMDP
from refine_plan.models.state import State
import sys

# Global map setup
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

INITIAL_LOC = "v1"


def _get_enabled_cond(sf_list, option):
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


def write_mongodb_to_yaml(mongo_connection_str):
    """Get the dataset from the database.

    Args:
        mongo_connection_str: The MongoDB conenction string"""

    loc_sf = StateFactor("location", ["v{}".format(i) for i in range(1, 12)])
    wire_sfs = [
        StateFactor("wire_at_v2", ["unknown", "no", "yes"]),
        StateFactor("wire_at_v7", ["unknown", "no", "yes"]),
        StateFactor("wire_at_v10", ["unknown", "no", "yes"]),
        StateFactor("wire_at_v11", ["unknown", "no", "yes"]),
    ]

    print("Writing Mongo Database to Yaml File")
    mongodb_to_yaml(
        mongo_connection_str,
        "refine-plan",
        "house-data",
        [loc_sf] + wire_sfs,
        "../data/house/dataset.yaml",
    )
    print("YAML Dataset Created")


def learn_options():
    """Learn the options from the YAML file."""
    dataset_path = "../data/house/dataset.yaml"
    output_dir = "../data/house/"

    loc_sf = StateFactor("location", ["v{}".format(i) for i in range(1, 12)])
    wire_sfs = [
        StateFactor("wire_at_v2", ["unknown", "no", "yes"]),
        StateFactor("wire_at_v7", ["unknown", "no", "yes"]),
        StateFactor("wire_at_v10", ["unknown", "no", "yes"]),
        StateFactor("wire_at_v11", ["unknown", "no", "yes"]),
    ]

    learn_dbns(dataset_path, output_dir, [loc_sf] + wire_sfs)


def run_planner():
    """Run REFINE-PLAN and synthesise a policy.

    Returns:
        The refined BT
    """

    loc_sf = StateFactor("location", ["v{}".format(i) for i in range(1, 12)])
    wire_sfs = [
        StateFactor("wire_at_v2", ["unknown", "no", "yes"]),
        StateFactor("wire_at_v7", ["unknown", "no", "yes"]),
        StateFactor("wire_at_v10", ["unknown", "no", "yes"]),
        StateFactor("wire_at_v11", ["unknown", "no", "yes"]),
    ]
    sf_list = [loc_sf] + wire_sfs

    goal_cond = OrCondition()
    wire_locs = ["v2", "v7", "v10", "v11"]
    for i in range(len(wire_locs)):
        goal_cond.add_cond(
            AndCondition(
                EqCondition(loc_sf, wire_locs[i]), EqCondition(wire_sfs[i], "yes")
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
        t_path = "../data/house/{}_transition.bifxml".format(option)
        r_path = "../data/house/{}_reward.bifxml".format(option)
        option_list.append(
            DBNOption(
                option, t_path, r_path, sf_list, _get_enabled_cond(sf_list, option)
            )
        )

    print("Creating MDP...")
    semi_mdp = SemiMDP(sf_list, option_list, labels, initial_state=init_state)
    print("Synthesising Policy...")
    policy = synthesise_policy(semi_mdp, prism_prop='Rmin=?[F "goal"]')
    policy.write_policy("../data/house/house_refined_policy.yaml")


if __name__ == "__main__":

    write_mongodb_to_yaml(sys.argv[1])
    # learn_options()
    # run_planner()
