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
from refine_plan.models.option import Option
from refine_plan.models.state import State


def manhattan_distance(s1, s2):
    """Get the manhattan distance between two states representing grid cells.

    Args:
        s1: The first state
        s2: The second state

    Returns:
        dist: The manhattan distance
    """
    return abs(s1["x"] - s2["x"]) + abs(s1["y"] - s2["y"])


def get_transition(s, good_hall_state, bad_hall_state):
    """Get the probabilistic transition for an option froma given state.

    Args:
        s: The current state
        good_hall_state: The intended goal of the option
        bad_hall_state: The other possible terminating state of the option

    Return:
        trans_pair: (pre_cond, prob_post_cond)
    """

    good_man_dist = manhattan_distance(s, good_hall_state)

    bad_man_dist = manhattan_distance(s, bad_hall_state)

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


def create_option(name, states, succ_state, fail_state, goal_state):
    """Creates an option for navigating to a hallway state.

    Args:
        name: The option name
        states: The states in a room
        succ_state: The successful end state for an option
        fail_state: The failure state
        goal_state: The planning goal (used for the reward function)

    Returns:
        option: The option for naivgating to a hallway
    """
    transition_list = []
    reward_list = []
    for state in states:
        trans = get_transition(state, succ_state, fail_state)
        succ_probs = trans[1]
        prob_to_succ = 0.0
        for succ_cond in succ_probs:
            if succ_cond.is_satisfied(succ_state):
                prob_to_succ = succ_probs[succ_cond]

        transition_list.append(trans)

        avg_r = prob_to_succ * manhattan_distance(state, succ_state)
        avg_r += (1 - prob_to_succ) * manhattan_distance(state, fail_state)
        reward_list.append((state.to_and_cond(), avg_r))

    return Option(name, transition_list, reward_list)


def create_components(goal_location):
    """Creates the rooms problem components.

    Args:
        goal_location: The destination. Must be a hallway location.

    Returns:
        - A list of StateFactor objects
        - A list of Option objects
        - A list of Label objects
    """
    sf_list = [IntStateFactor("x", 0, 10), IntStateFactor("y", 0, 10)]

    goal_state = State({sf_list[0]: goal_location[0], sf_list[1]: goal_location[1]})

    labels = [Label("goal", goal_state.to_and_cond())]

    option_list = []

    hall_1_5 = State({sf_list[0]: 1, sf_list[1]: 5})
    hall_5_2 = State({sf_list[0]: 5, sf_list[1]: 2})
    hall_8_6 = State({sf_list[0]: 8, sf_list[1]: 6})
    hall_5_9 = State({sf_list[0]: 5, sf_list[1]: 9})

    room_1_states = []
    for i in range(5):
        for j in range(5):
            room_1_states.append(State({sf_list[0]: i, sf_list[1]: j}))
    room_1_states.append(hall_1_5)
    room_1_states.append(hall_5_2)

    room_2_states = []
    for i in range(6, 11):
        for j in range(6):
            room_2_states.append(State({sf_list[0]: i, sf_list[1]: j}))
    room_2_states.append(hall_5_2)
    room_2_states.append(hall_8_6)

    room_3_states = []
    for i in range(5):
        for j in range(6, 11):
            room_3_states.append(State({sf_list[0]: i, sf_list[1]: j}))
    room_3_states.append(hall_1_5)
    room_3_states.append(State({sf_list[0]: 5, sf_list[1]: 9}))

    room_4_states = []
    for i in range(6, 11):
        for j in range(7, 11):
            room_4_states.append(State({sf_list[0]: i, sf_list[1]: j}))
    room_4_states.append(hall_8_6)
    room_4_states.append(hall_5_9)

    option_list.append(
        create_option("room_1_to_1_5", room_1_states, hall_1_5, hall_5_2, goal_state)
    )
    option_list.append(
        create_option("room_1_to_5_2", room_1_states, hall_5_2, hall_1_5, goal_state)
    )
    option_list.append(
        create_option("room_2_to_5_2", room_2_states, hall_5_2, hall_8_6, goal_state)
    )
    option_list.append(
        create_option("room_2_to_8_6", room_2_states, hall_8_6, hall_5_2, goal_state)
    )
    option_list.append(
        create_option("room_3_to_1_5", room_3_states, hall_1_5, hall_5_9, goal_state)
    )
    option_list.append(
        create_option("room_3_to_5_9", room_3_states, hall_5_9, hall_1_5, goal_state)
    )
    option_list.append(
        create_option("room_4_to_5_9", room_4_states, hall_5_9, hall_8_6, goal_state)
    )
    option_list.append(
        create_option("room_4_to_8_6", room_4_states, hall_8_6, hall_5_9, goal_state)
    )

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
        default_action="dead",
        out_file="/tmp/test_bt.xml",
    )

    output_bt(bt)
