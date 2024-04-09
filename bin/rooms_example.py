#!/usr/bin/env python3
""" Implements the rooms example from Sutton's semi-MDP paper.

This problem involves a grid world with 4 rooms, and is an example
semi-MDP problem.

We use it here to test the option->BT pipeline.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.algorithms.refine import synthesise_bt_from_options
from refine_plan.models.state_factor import IntStateFactor
from refine_plan.models.condition import Label
from refine_plan.models.state import State


def get_transition(s, good_hall_state, bad_hall_state):
    """Get the probabilistic transition for an option froma given state.

    Args:
        s: The current state
        good_hall_state: The intended goal of the option
        bad_hall_state: The other possible terminating state of the option

    Return:
        trans_pair: (pre_cond, prob_post_cond)
    """

    good_man_dist = abs(s["x"] - good_hall_state["x"])
    good_man_dist += abs(s["y"] - good_hall_state["y"])

    bad_man_dist = abs(s["x"] - bad_hall_state["x"])
    bad_man_dist += abs(s["y"] - bad_hall_state["y"])

    if good_man_dist == 0:  # If we're there, we succeed
        return (s.to_and_cond(), {good_hall_state.to_and_cond(): 1.0})
    elif bad_man_dist == 0:  # In this case, just hand code a probability
        return (
            s.to_and_cond(),
            {good_hall_state.to_and_cond(): 0.6, bad_hall_state.to_and_cond(): 0.4},
        )

    good_comp = 2.0 / good_man_dist
    bad_comp = 1.0 / bad_man_dist

    good_prob = good_comp / (good_comp + bad_comp)
    bad_prob = bad_comp / (good_comp + bad_comp)

    return (
        s.to_and_cond(),
        {
            good_hall_state.to_and_cond(): good_prob,
            bad_hall_state.to_and_cond(): bad_prob,
        },
    )


def create_components(goal_location):
    """Creates the rooms problem components.

    Args:
        goal_location: The destination. Must be a hallway location.

    Returns:
        sf_list: A list of StateFactor objects
        option_list: A list of Option objects
        labels: A list of Label objects
    """
    sf_list = [IntStateFactor("x", 0, 10), IntStateFactor("y", 0, 10)]

    goal_state = State({sf_list[0]: goal_location[0], sf_list[1]: goal_location[1]})

    labels = [Label("goal", goal_state.to_and_cond())]

    option_list = []

    room_1_states = []
    for i in range(5):
        for j in range(5):
            room_1_states.append(State({sf_list[0]: i, sf_list[0]: j}))
    room_1_states.append(State({sf_list[0]: 1, sf_list[1]: 5}))
    room_1_states.append(State({sf_list[0]: 5, sf_list[1]: 2}))

    room_2_states = []
    for i in range(6, 11):
        for j in range(6):
            room_2_states.append(State({sf_list[0]: i, sf_list[0]: j}))
    room_2_states.append(State({sf_list[0]: 5, sf_list[1]: 2}))
    room_2_states.append(State({sf_list[0]: 8, sf_list[1]: 6}))

    room_3_states = []
    for i in range(5):
        for j in range(6, 11):
            room_3_states.append(State({sf_list[0]: i, sf_list[0]: j}))
    room_3_states.append(State({sf_list[0]: 1, sf_list[1]: 5}))
    room_3_states.append(State({sf_list[0]: 5, sf_list[1]: 9}))

    room_4_states = []
    for i in range(6, 11):
        for j in range(7, 11):
            room_4_states.append(State({sf_list[0]: i, sf_list[0]: j}))
    room_4_states.append(State({sf_list[0]: 8, sf_list[1]: 6}))
    room_4_states.append(State({sf_list[0]: 5, sf_list[1]: 9}))

    # TODO: Build options!

    return sf_list, option_list, labels


def output_bt(bt):
    """Output the BT's decision making, i.e. the action at each state.

    Args:
        bt: A behaviour tree
    """

    x = IntStateFactor("x", 0, 10)
    y = IntStateFactor("y", 0, 10)

    for i in range(11):
        for j in range(11):
            state = State({x: i, y: j})
            print("{}: {}".format(state, bt.tick_at_state(state)))


if __name__ == "__main__":

    sf_list, option_list, labels = create_components((8, 6))

    bt = synthesise_bt_from_options(
        sf_list,
        option_list,
        labels,
        prism_prop='Rmin=?[F "goal"]',
        none_replacer="dead",
    )

    output_bt(bt)
