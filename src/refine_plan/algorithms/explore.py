#!/usr/bin/env python3
"""Functions for synthesising an exploration policy using the approach in:

Shyam, P., Jaśkowski, W. and Gomez, F., 2019, May. Model-based active exploration.
In International conference on machine learning (pp. 5779-5788). PMLR.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.dbn_option_ensemble import DBNOptionEnsemble
from refine_plan.learning.option_learning import mongodb_to_dict
from multiprocessing import Process, SimpleQueue
from refine_plan.models.semi_mdp import SemiMDP
from refine_plan.models.state import State
from itertools import product


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


def _create_ensemble_for_option(
    opt,
    data,
    ensemble_size,
    horizon,
    sf_list,
    enabled_cond,
    state_idx_map,
    queue,
):
    """Auxiliary function for creating an ensemble model.

    This will be called in separate processes for concurrency.

    Args:
        opt: The option name
        data: The dataset for this option
        ensemble_size: The number of DBNs in the ensemble
        horizon: The planning horizon length
        sf_list: The state factors for planning
        enabled_cond: The enabled condition for this option
        state_idx_map: A mapping from states to matrix indices
        queue: A thread-safe queue for return values
    """
    queue.put(
        DBNOptionEnsemble(
            opt, data, ensemble_size, horizon, sf_list, enabled_cond, state_idx_map
        )
    )


def _build_options(
    option_names,
    dataset,
    ensemble_size,
    horizon,
    sf_list,
    enabled_conds,
    state_idx_map,
):
    """Build a set of DBNOptionEnsemble objects in parallel using processes.

    Args:
        option_names: A list of option names
        dataset: The dataset for DBN learning
        ensemble_size: How many DBNs are in an ensemble
        horizon: The planning horizon length
        sf_list: The state factors used for planning
        enabled_conds: A dictionary from option name to enabled condition
        state_idx_map: A mapping from states to matrix indices

    Returns:
        A list of DBNOptionEnsemble objects
    """
    queue = SimpleQueue()
    procs, option_list = [], []
    for opt in option_names:
        procs.append(
            Process(
                target=_create_ensemble_for_option,
                args=(
                    opt,
                    dataset[opt],
                    ensemble_size,
                    horizon,
                    sf_list,
                    enabled_conds[opt],
                    state_idx_map,
                    queue,
                ),
            )
        )
        procs[-1].start()

    while len(option_list) != len(option_names):  # Get all complete ensembles
        option_list.append(queue.get())

    for p in procs:  # Possibly not necessary, but do to be safe
        p.join()

    return option_list


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

    Returns:
        The exploration policy
    """
    # Step 1: Retrieve the data from mongodb
    print("Reading data from MongoDB...")
    dataset = mongodb_to_dict(
        connection_str, db_name, collection_name, sf_list, sort_by="_meta.inserted_at"
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
    )

    # Step 3: Build the MDP
    print("Building MDP...")
    mdp = SemiMDP(sf_list, option_list, [], initial_state=initial_state)

    # Step 4: Solve the MDP and return the policy
    print("Synthesising Policy...")
    return solve_finite_horizon_mdp(mdp, horizon)
