#!/usr/bin/env python3
"""A script to run REFINE-PLAN in the CONVINCE overarching demo domain.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.condition import Label, EqCondition, AndCondition, OrCondition
from refine_plan.learning.option_learning import mongodb_to_yaml, learn_dbns
from refine_plan.algorithms.semi_mdp_solver import synthesise_policy
from refine_plan.models.state_factor import StateFactor
from refine_plan.models.dbn_option import DBNOption
from refine_plan.models.semi_mdp import SemiMDP
from refine_plan.models.policy import Policy
from refine_plan.models.state import State
import sys

# Node to nodes we can directly reach from that node
GRAPH = {
    "hall": ["dining_table", "fridge"],
    "dining_table": ["hall", "side_table"],
    "side_table": ["dining_table", "kitchen_table"],
    "kitchen_table": ["side_table", "fridge"],
    "fridge": ["kitchen_table", "hall"],
}

INITIAL_LOC = "hall"
BREAD_LOCS = ["side_table", "dining_table", "kitchen_table", "fridge"]
NAV_LOCS = ["hall"] + BREAD_LOCS


def build_sfs():
    loc_sf = StateFactor("location", NAV_LOCS)
    bread_sfs = []
    for bread_loc in BREAD_LOCS:
        bread_sfs.append(
            StateFactor("bread_at_{}".format(bread_loc), ["unknown", "no", "yes"])
        )

    return [loc_sf] + bread_sfs


def _get_enabled_cond(sf_list, option):
    """Get the enabled condition for an option.

    Args:
        sf_list: The list of state factors
        option: The option we want the condition for

    Returns:
        The enabled condition for the option
    """
    sf_dict = {sf.get_name(): sf for sf in sf_list}

    if option == "detect":
        enabled_cond = OrCondition()

        for loc in BREAD_LOCS:
            enabled_cond.add_cond(
                AndCondition(
                    EqCondition(sf_dict["location"], loc),
                    EqCondition(sf_dict["bread_at_{}".format(loc)], "unknown"),
                )
            )
        return enabled_cond
    else:  # Edge navigation action
        start_loc = option.split("TO")[0]
        return EqCondition(sf_dict["location"], start_loc)


def write_mongodb_to_yaml(mongo_connection_str):
    """Get the dataset from the database.

    Args:
        mongo_connection_str: The MongoDB conenction string"""

    print("Writing Mongo Database to Yaml File")
    mongodb_to_yaml(
        mongo_connection_str,
        "refine-plan-demo",
        "demo-data",
        build_sfs(),
        "../data/overarching_demo/dataset.yaml",
    )
    print("YAML Dataset Created")


def learn_options():
    """Learn the options from the YAML file."""
    dataset_path = "../data/overarching_demo/dataset.yaml"
    output_dir = "../data/overarching_demo/"
    learn_dbns(dataset_path, output_dir, build_sfs())


def run_planner():
    """Run REFINE-PLAN and synthesise a policy.

    Returns:
        The refined BT
    """
    sf_list = build_sfs()

    goal_cond = OrCondition()
    for i in range(len(BREAD_LOCS)):
        goal_cond.add_cond(
            AndCondition(
                EqCondition(sf_list[0], BREAD_LOCS[i]),
                EqCondition(sf_list[i + 1], "yes"),
            )
        )

    labels = [Label("goal", goal_cond)]

    option_names = set([])
    for src in GRAPH:
        for dst in GRAPH[src]:
            option_names.add(f"{src}TO{dst}")
    option_names.add("detect")

    assert len(set(option_names)) == 11  # Quick safety check

    init_state_dict = {sf: "unknown" for sf in sf_list[1:]}
    init_state_dict[sf_list[0]] = INITIAL_LOC
    init_state = State(init_state_dict)

    option_list = []
    for option in option_names:
        print("Reading in option: {}".format(option))
        t_path = "../data/overarching_demo/{}_transition.bifxml".format(option)
        r_path = "../data/overarching_demo/{}_reward.bifxml".format(option)
        option_list.append(
            DBNOption(
                option, t_path, r_path, sf_list, _get_enabled_cond(sf_list, option)
            )
        )

    print("Creating MDP...")
    semi_mdp = SemiMDP(sf_list, option_list, labels, initial_state=init_state)
    print("Synthesising Policy...")
    policy = synthesise_policy(semi_mdp, prism_prop='Rmin=?[F "goal"]')
    policy.write_policy("../data/overarching_demo/refined_policy.yaml")


if __name__ == "__main__":

    # write_mongodb_to_yaml(sys.argv[1])
    # learn_options()
    run_planner()
