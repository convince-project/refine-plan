#!/usr/bin/env python3
"""Functions for synthesising an exploration policy using the approach in:

Shyam, P., Jaśkowski, W. and Gomez, F., 2019, May. Model-based active exploration.
In International conference on machine learning (pp. 5779-5788). PMLR.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.algorithms.semi_mdp_solver import synthesise_policy
from refine_plan.models.dbn_option_ensemble import DBNOptionEnsemble
from refine_plan.learning.option_learning import mongodb_to_dict
from refine_plan.models.condition import Label, EqCondition
from refine_plan.models.state_factor import IntStateFactor
from refine_plan.models.policy import TimeDependentPolicy
from multiprocessing import Process, SimpleQueue
from refine_plan.models.semi_mdp import SemiMDP
from refine_plan.models.state import State
from itertools import product
import numpy as np


def _build_state_idx_map(sf_list):
    """Build a mapping from states to numerical IDs.

    Args:
        sf_list: The list of state factors

    Return:
        A dictionary from states to numerical IDs
    """
    sf_vals = [sf.get_valid_values() for sf in sf_list]
    state_idx_map = {}
    state_id = 0

    for state_vals in product(*sf_vals):
        state_dict = {}
        for i in range(len(state_vals)):
            state_dict[sf_list[i]] = state_vals[i]
        state_idx_map[State(state_dict)] = state_id
        state_id += 1

    return state_idx_map


def _build_options(
    option_names,
    dataset,
    ensemble_size,
    horizon,
    sf_list,
    enabled_conds,
    state_idx_map,
    compute_prism_str=False,
    motion_params=None,
):
    """Build a set of DBNOptionEnsemble objects.

    Args:
        option_names: A list of option names
        dataset: The dataset for DBN learning
        ensemble_size: How many DBNs are in an ensemble
        horizon: The planning horizon length
        sf_list: The state factors used for planning
        enabled_conds: A dictionary from option name to enabled condition
        state_idx_map: A mapping from states to matrix indices
        compute_prism_str: If True, build the PRISM strings
        motion_params: A dictionary from option names to a list of params for that option

    Returns:
        A list of DBNOptionEnsemble objects
    """
    option_list = []

    for opt in option_names:
        params = [None] if motion_params is None else motion_params[opt]
        for param in params:
            if param is None:
                full_name = opt
            else:
                full_name = "{}.{}".format(opt, param)
            option_list.append(
                DBNOptionEnsemble(
                    full_name,
                    dataset[full_name],
                    ensemble_size,
                    horizon,
                    sf_list,
                    enabled_conds[opt],
                    state_idx_map,
                    compute_prism_str=compute_prism_str,
                )
            )

    return option_list


def solve_finite_horizon_mdp(mdp, state_idx_map, horizon, mat_type=np.float32):
    """Synthesise a policy for a finite horizon MDP.

    This can be done through one backwards Bellman backup through time.

    Args:
        mdp: The MDP (with DBNOptionEnsemble options)
        state_idx_map: The state to matrix indice mapping
        horizon: The planning horizon
        mat_type: The dtype for the matrices

    Returns:
        A TimeDependentPolicy
    """
    state_action_dicts, value_dicts = [None] * horizon, [None] * horizon

    idx_opt_map, opt_id = {}, 0
    trans_to_stack, rew_to_stack = [], []
    for option in mdp._options:
        idx_opt_map[opt_id] = option
        trans_to_stack.append(mdp._options[option]._sampled_transition_mat)
        rew_to_stack.append(mdp._options[option]._reward_mat)
        opt_id += 1
    transition_mat = np.stack(trans_to_stack, axis=0, dtype=mat_type)
    reward_mat = np.stack(rew_to_stack, axis=0, dtype=mat_type)

    # All states have value 0 at horizon
    current_value = np.zeros(len(state_idx_map), dtype=mat_type)
    for timestep in range(horizon - 1, -1, -1):  # Move backwards through time

        print("Solving MDP for timestep: {}".format(timestep))
        q_vals = reward_mat + np.matmul(transition_mat, current_value)
        current_value = np.max(q_vals, axis=0)
        policy_actions = np.argmax(q_vals, axis=0)

        state_action_dict, value_dict = {}, {}
        for state in state_idx_map:
            if current_value[state_idx_map[state]] != 0.0:
                state_action_dict[state] = idx_opt_map[
                    policy_actions[state_idx_map[state]]
                ]
                value_dict[state] = current_value[state_idx_map[state]]
        state_action_dicts[timestep] = state_action_dict
        value_dicts[timestep] = value_dict

    return TimeDependentPolicy(state_action_dicts, value_dicts)


def synthesise_exploration_policy(
    connection_str,
    db_name,
    collection_name,
    sf_list,
    option_names,
    ensemble_size,
    horizon,
    enabled_conds,
    initial_state=None,
    use_storm=False,
    motion_params=None,
):
    """Synthesises an exploration policy for the current episode.

    Args:
        connection_str: The mongodb connection string
        db_name: The Mongo database name
        collection_name: The collection within the database
        sf_list: The list of state factors used for planning
        option_names: The list of option (action) names
        ensemble_size: The size of the ensemble model for each option
        horizon: The length of the planning horizon
        enabled_conds: A dictionary from option name to enabled Condition
        initial_state: The initial state of the exploration MDP
        use_storm: If True, use Storm instead of the local solver
        motion_params: A dictionary from option names to a list of params for that option

    Returns:
        The exploration policy
    """
    print("USING STORM: {}".format(use_storm))

    # Step 1: Retrieve the data from mongodb
    print("Reading data from MongoDB...")
    dataset = mongodb_to_dict(
        connection_str,
        db_name,
        collection_name,
        sf_list,
        sort_by="_meta.inserted_at",
        split_by_motion=(motion_params is not None),
    )
    assert len(dataset) == len(option_names)

    # Step 2: Build the state to idx mapping
    print("Building state to idx mapping...")
    state_idx_map = _build_state_idx_map(sf_list)

    # Step 2: Build the DBNOptionEnsemble objects
    print("Building option ensembles...")
    option_list = _build_options(
        option_names,
        dataset,
        ensemble_size,
        horizon,
        sf_list,
        enabled_conds,
        state_idx_map,
        use_storm,
        motion_params=motion_params,
    )

    # Step 3: Build the MDP
    print("Building MDP...")
    labels = []
    if use_storm:
        time_sf = IntStateFactor("time", 0, horizon)
        sf_list = sf_list + [time_sf]
        labels.append(Label("horizon", EqCondition(time_sf, horizon)))

    mdp = SemiMDP(sf_list, option_list, labels, initial_state=initial_state)

    # Step 4: Solve the MDP and return the policy
    print("Synthesising Policy...")
    if use_storm:
        return synthesise_policy(mdp, prism_prop='Rmax=?[F "horizon"]')
    else:
        return solve_finite_horizon_mdp(mdp, state_idx_map, horizon)
