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
from multiprocessing import Process, SimpleQueue
from refine_plan.models.semi_mdp import SemiMDP


def _create_ensemble_for_option(
    opt, data, ensemble_size, horizon, sf_list, enabled_cond, queue
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
        queue: A thread-safe queue for return values
    """
    queue.put(
        DBNOptionEnsemble(opt, data, ensemble_size, horizon, sf_list, enabled_cond)
    )


def _build_options(
    option_names, dataset, ensemble_size, horizon, sf_list, enabled_conds
):
    """Build a set of DBNOptionEnsemble objects in parallel using processes.

    Args:
        option_names: A list of option names
        dataset: The dataset for DBN learning
        ensemble_size: How many DBNs are in an ensemble
        horizon: The planning horizon length
        sf_list: The state factors used for planning
        enabled_conds: A dictionary from option name to enabled condition

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
    dataset = mongodb_to_dict(
        connection_str, db_name, collection_name, sf_list, sort_by="_meta.inserted_at"
    )
    assert len(dataset) == len(option_names)

    # Step 2: Build the DBNOptionEnsemble objects
    option_list = _build_options(
        option_names, dataset, ensemble_size, horizon, sf_list, enabled_conds
    )

    # Step 3: Build the MDP
    time_sf = IntStateFactor("time", 0, horizon)
    labels = [Label("horizon", EqCondition(time_sf, horizon))]
    mdp = SemiMDP(sf_list + [time_sf], option_list, labels, initial_state=initial_state)

    # Step 4: Solve the MDP and return the policy
    return synthesise_policy(mdp, prism_prop='Rmax=?[F "horizon"]')
