#!/usr/bin/env python3
"""A script to build the bookstore MDP and output the corresponding SCXML file.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.condition import Label, EqCondition, AndCondition, OrCondition
from refine_plan.algorithms.semi_mdp_solver import synthesise_policy
from refine_plan.models.state_factor import StateFactor
from refine_plan.models.dbn_option import DBNOption
from refine_plan.models.semi_mdp import SemiMDP
from refine_plan.models.state import State
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

# Problem Setup
INITIAL_LOC = "v1"
GOAL_LOC = "v8"


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


def generate_mdp_scxml():
    """Generate the bookstore MDP and write out the corresponding SCXML file."""

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
        t_path = "../data/bookstore/{}_transition.bifxml".format(option)
        r_path = "../data/bookstore/{}_reward.bifxml".format(option)
        option_list.append(
            DBNOption(
                option, t_path, r_path, sf_list, _get_enabled_cond(sf_list, option)
            )
        )

    print("Creating MDP...")
    semi_mdp = SemiMDP(sf_list, option_list, labels, initial_state=init_state)
    print("Writing SCXML file")
    semi_mdp.to_scxml_file("../data/bookstore/bookstore_mdp.scxml", name="Bookstore")
    print("Synthesising Policy...")
    policy = synthesise_policy(semi_mdp, prism_prop='Rmin=?[F "goal"]')
    policy.to_scxml(
        "../data/bookstore/bookstore_policy.scxml",
        model_name="Bookstore",
        initial_state=init_state,
        name="Bookstore_Policy",
    )
    # policy.write_policy("../data/bookstore/bookstore_refined_policy.yaml")


if __name__ == "__main__":
    generate_mdp_scxml()
