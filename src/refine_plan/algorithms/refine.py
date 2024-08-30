#!/usr/bin/env python3
""" Functions which bring the stages of BT refinement together.

There are two key stages to refinement:
1. State space extraction and option learning.
2. Semi-MDP construction and solution, and policy->BT conversion.

These two stages are represented with different functions.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.algorithms.semi_mdp_solver import synthesise_policy
from refine_plan.algorithms.policy_to_bt import PolicyBTConverter
from refine_plan.models.semi_mdp import SemiMDP


def synthesise_bt_from_options(
    sf_list,
    option_list,
    labels,
    initial_state=None,
    prism_prop='Rmax=?[F "goal"]',
    default_action="None",
    out_file=None,
):
    """Synthesise a BT using options learned from the initial BT.

    Args:
        sf_list: A list of StateFactor objects for the semi-MDP
        option_list: A list of Option objects which capture robot capabilities
        labels: A list of Label objects within the semi-MDP
        initial_state: The semi-MDP initial state (if there is one)
        prism_prop: The PRISM property to solve for (passed to Storm)
        default_action: The action which replaces any None actions in a policy
        out_file: Optional. The path for the final BT XML file

    Returns:
        The final refined BT
    """
    semi_mdp = SemiMDP(sf_list, option_list, labels, initial_state=initial_state)
    print("Synthesising Policy...")
    policy = synthesise_policy(semi_mdp, prism_prop=prism_prop)

    print("Converting Policy to BT...")
    converter = PolicyBTConverter(default_action=default_action)
    return converter.convert_policy(policy, out_file)
