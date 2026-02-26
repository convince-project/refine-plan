#!/usr/bin/env python3
"""A script to run REFINE-PLAN in the house domain.

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

# Global map setup
GRAPH = {
    "v2": {"e24": "v4"},
    "v3": {"e34": "v4"},
    "v4": {"e45": "v5"},
    "v5": {"e45": "v4"},
}

INITIAL_LOC = "v2"


def _get_enabled_cond(loc_sf, option):
    """Get the enabled condition for an option.

    Args:
        loc_sf: The location state factor
        option: The option we want the condition for

    Returns:
        The enabled condition for the option
    """

    enabled_cond = OrCondition()
    for node in GRAPH:
        if option in GRAPH[node]:
            enabled_cond.add_cond(EqCondition(loc_sf, node))
    return enabled_cond


def write_mongodb_to_yaml(mongo_connection_str):
    """Get the dataset from the database.

    Args:
        mongo_connection_str: The MongoDB conenction string"""

    loc_sf = StateFactor("location", ["v{}".format(i) for i in range(2, 6)])

    print("Writing Mongo Database to Yaml File")
    mongodb_to_yaml(
        mongo_connection_str,
        "refine-plan",
        "house-data",
        [loc_sf],
        "../data/t3.4/dataset.yaml",
    )
    print("YAML Dataset Created")


def learn_options():
    """Learn the options from the YAML file."""
    dataset_path = "../data/t3.4/dataset.yaml"
    output_dir = "../data/t3.4/"

    loc_sf = StateFactor("location", ["v{}".format(i) for i in range(1, 12)])
    learn_dbns(dataset_path, output_dir, [loc_sf])


def run_planner():
    """Run REFINE-PLAN and synthesise a policy.

    Returns:
        The refined BT
    """

    loc_sf = StateFactor("location", ["v{}".format(i) for i in range(1, 12)])

    sf_list = [loc_sf]

    goal_cond = EqCondition(loc_sf, "v5")

    labels = [Label("goal", goal_cond)]

    option_names = set([])
    for src in GRAPH:
        option_names.update(list(GRAPH[src].keys()))

    assert len(option_names) == 3  # Quick safety check

    init_state_dict = {loc_sf: INITIAL_LOC}
    init_state = State(init_state_dict)

    option_list = []
    for option in option_names:
        print("Reading in option: {}".format(option))
        t_path = "../data/t3.4/{}_transition.bifxml".format(option)
        r_path = "../data/t3.4/{}_reward.bifxml".format(option)
        option_list.append(
            DBNOption(
                option, t_path, r_path, sf_list, _get_enabled_cond(loc_sf, option)
            )
        )

    print("Creating MDP...")
    semi_mdp = SemiMDP(sf_list, option_list, labels, initial_state=init_state)
    semi_mdp.to_scxml_file(
        f"../data/t3.4/mdp_{INITIAL_LOC}.scxml", "House_Policy", name="House"
    )
    print("Synthesising Policy...")
    policy = synthesise_policy(semi_mdp, prism_prop='Rmin=?[F "goal"]')
    policy.to_scxml(
        f"../data/t3.4/policy_{INITIAL_LOC}.scxml",
        model_name="House",
        initial_state=init_state,
        name="House_Policy",
    )


if __name__ == "__main__":

    # write_mongodb_to_yaml(sys.argv[1])
    # learn_options()
    run_planner()
