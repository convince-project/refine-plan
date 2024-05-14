#!/usr/bin/env python3
""" Implements the stochastic grid environment problem from Russell & Norvig.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.condition import Label, AndCondition, EqCondition, OrCondition
from refine_plan.algorithms.refine import synthesise_bt_from_options
from refine_plan.algorithms.policy_to_bt import PolicyBTConverter
from refine_plan.models.state_factor import IntStateFactor
from refine_plan.models.semi_mdp import SemiMDP
from refine_plan.models.policy import Policy
from refine_plan.models.option import Option
from refine_plan.models.state import State
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams.update({"font.size": 40})


def in_bounds(x, y):
    """Returns whether (x,y) is in bounds.

    Args:
        x: The x value
        y: The y value

    Returns:
        Is the cell in bounds?
    """
    x_in = x >= 0 and x < 4
    y_in = y >= 0 and y < 3
    not_obs = (x, y) != (1, 1)
    return x_in and y_in and not_obs


def get_succ_list(x, y, option_name):
    """Get the successor locations for a given option.

    Args:
        x: The x location
        y: The y location
        option_name: The option name

    Returns:
        The list of possible successors
    """
    up = (x, y + 1) if in_bounds(x, y + 1) else (x, y)
    down = (x, y - 1) if in_bounds(x, y - 1) else (x, y)
    left = (x - 1, y) if in_bounds(x - 1, y) else (x, y)
    right = (x + 1, y) if in_bounds(x + 1, y) else (x, y)

    if option_name == "up":
        return [up, left, right]
    elif option_name == "down":
        return [down, left, right]
    elif option_name == "left":
        return [left, up, down]
    elif option_name == "right":
        return [right, up, down]


def xy_to_cond(x_val, y_val, x_sf, y_sf):
    """Converts (x,y) into a Condition.

    Args:
        x_val: The x value
        y_val: The y value
        x_sf: The x state factor
        y_sf: The y state factor

    Returns:
        The corresponding condition
    """
    return AndCondition(EqCondition(x_sf, x_val), EqCondition(y_sf, y_val))


def build_option(option_name):
    """Creates a single option for one direction.

    Args:
        option_name: The option mame

    Returns:
        The corresponding Option object
    """
    x_sf = IntStateFactor("x", 0, 3)
    y_sf = IntStateFactor("y", 0, 2)
    trans = []
    reward = []

    terminal = [(1, 1), (3, 2), (3, 1)]

    for x in range(4):
        for y in range(3):
            if (x, y) in terminal:
                continue

            # first element is successful, rest are at right angles
            succ_list = get_succ_list(x, y, option_name)
            assert len(succ_list) == 3

            pre_cond = xy_to_cond(x, y, x_sf, y_sf)

            prob_post_conds = {succ_list[0]: 0.8}
            for i in range(1, len(succ_list)):
                if succ_list[i] in prob_post_conds:
                    prob_post_conds[succ_list[i]] += 0.1
                else:
                    prob_post_conds[succ_list[i]] = 0.1

            for succ in prob_post_conds:
                if succ == (3, 2):
                    reward.append((pre_cond, prob_post_conds[succ] * 1))
                elif succ == (3, 1):
                    reward.append((pre_cond, prob_post_conds[succ] * -1))

            prob_post_conds = {
                xy_to_cond(xy[0], xy[1], x_sf, y_sf): prob_post_conds[xy]
                for xy in prob_post_conds
            }

            trans.append((pre_cond, prob_post_conds))

    return Option(option_name, trans, reward)


def create_components():
    """Creates the navigation problem components.

    Returns:
        - A list of StateFactor objects
        - A list of Option objects
        - A list of Label objects
    """
    sf_list = [IntStateFactor("x", 0, 3), IntStateFactor("y", 0, 2)]
    option_list = [build_option(move) for move in ["up", "down", "left", "right"]]
    labels = [Label("goal", xy_to_cond(3, 2, sf_list[0], sf_list[1]))]

    return sf_list, option_list, labels


def synthesise_refined_bt():
    """Synthesise the refined BT.

    Returns:
        The refined bt
    """

    sf_list, option_list, labels = create_components()

    return synthesise_bt_from_options(
        sf_list, option_list, labels, None, 'Pmax=?[F "goal"]', "idle"
    )


def run_simulation(semi_mdp, bt, initial_state):
    """Run a simulation through a semi-MDP.

    Args:
        semi_mdp: The semi-MDP
        bt: The behaviour tree
        initial_state: The initial state

    Returns:
        reward: The total reward
    """
    x_sf = semi_mdp.get_state_factors()["x"]
    y_sf = semi_mdp.get_state_factors()["y"]

    current_state = initial_state

    good_goal = State({x_sf: 3, y_sf: 2})
    bad_goal = State({x_sf: 3, y_sf: 1})

    while current_state not in [good_goal, bad_goal]:
        action = bt.tick_at_state(current_state)
        succ_list = get_succ_list(current_state["x"], current_state["y"], action)

        unique_succ = set(succ_list)
        unique_succ = [State({x_sf: xy[0], y_sf: xy[1]}) for xy in unique_succ]
        probs = [
            semi_mdp.get_transition_prob(current_state, action, succ)
            for succ in unique_succ
        ]
        current_state = np.random.choice(a=unique_succ, p=probs)

    if current_state == good_goal:
        return 1
    elif current_state == bad_goal:
        return 0
    else:
        raise Exception("Invalid terminal state reached")


def generate_initial_bt():
    """Generate the initial BT.

    Returns:
        The initial BT
    """
    x_sf = IntStateFactor("x", 0, 3)
    y_sf = IntStateFactor("y", 0, 2)

    state_action_map = {
        (0, 0): "up",
        (0, 1): "up",
        (0, 2): "right",
        (1, 0): "right",
        (1, 1): None,
        (1, 2): "right",
        (2, 0): "up",
        (2, 1): "up",
        (2, 2): "right",
        (3, 0): "left",
        (3, 1): None,
        (3, 2): None,
    }

    state_action_map = {
        State({x_sf: xy[0], y_sf: xy[1]}): state_action_map[xy]
        for xy in state_action_map
    }

    policy = Policy(state_action_map)

    return PolicyBTConverter("idle").convert_policy(policy)


def run_comparison():
    """Run a comparison between the initial BT and refined BT."""
    print("CREATING INITIAL BT")
    initial_bt = generate_initial_bt()
    print("CREATING REFINED BT")
    refined_bt = synthesise_refined_bt()

    sf_list, option_list, labels = create_components()
    semi_mdp = SemiMDP(sf_list, option_list, labels)

    x_sf = sf_list[0]
    y_sf = sf_list[1]

    init_states = []
    for x in range(4):
        for y in range(3):
            if (x, y) == (1, 1) or (x, y) == (3, 2) or (x, y) == (3, 1):
                continue
            else:
                init_states.append(State({x_sf: x, y_sf: y}))
                print(init_states[-1], refined_bt.tick_at_state(init_states[-1]))

    initial_results = []
    refined_results = []

    for init in init_states:
        print("RUNNING SIMS FOR init state: {}".format(init))
        for i in range(100):
            initial_results.append(run_simulation(semi_mdp, initial_bt, init))
            refined_results.append(run_simulation(semi_mdp, refined_bt, init))

    init_mean = np.average(initial_results)
    init_std = np.std(initial_results)
    refined_mean = np.average(refined_results)
    refined_std = np.std(refined_results)
    print("INITIAL MEAN: {}".format(init_mean))
    print("INITIAL STD: {}".format(init_std))
    print("REFINED MEAN: {}".format(refined_mean))
    print("REFINED STD: {}".format(refined_std))

    # box = plt.boxplot(
    #    [initial_results, refined_results],
    #    whis=[0, 100],
    #    positions=[1, 2],
    #    widths=0.6,
    # )
    # plt.bar(
    #    [1, 2],
    #    [init_mean, refined_mean],
    #    color="tab:blue",
    # )
    # plt.errorbar(
    #    [1, 2],
    #    [init_mean, refined_mean],
    #    yerr=[init_var, refined_var],
    #    fmt="o",
    #    color="tab:red",
    # )
    # plt.xticks([1, 2], ["Initial BT", "Refined BT"])
    # plt.show()


if __name__ == "__main__":
    run_comparison()
