#!/usr/bin/env python3
"""A class for an ensemble of DBNOption models.

This is used for active exploration.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.dbn_option import DBNOption
from refine_plan.learning.option_learning import (
    _initialise_dict_for_option,
    learn_bns_for_one_option,
)
from refine_plan.models.option import Option
import random


class DBNOptionEnsemble(Option):
    """A class containing an ensemble of DBNOptions for active exploration.

    Each DBNOption in the ensemble is trained on a different subset of the data.

    Attributes:
        Same as superclass, plus:
        _ensemble_size: The size of the ensemble
        _horizon: Number of steps in the planning horizon
        _sf_list: The list of state factors that make up the state space
        _enabled_cond: A Condition which is satisfied in states where the option is enabled
        _dbns: The ensemble (list) of DBNOptions
        _transition_mats: The corresponding flat transition matrices for each DBNOption.
    """

    def __init__(self, name, data, ensemble_size, horizon, sf_list, enabled_cond):
        """Initialise attributes.

        Args:
            name: The option's name
            data: A dictionary of data items to build the ensemble from
            ensemble_size: The number of DBNs in the ensemble
            horizon: The planning horizon length
            sf_list: The list of state factors that make up the state space
            enabled_cond: A Condition which is satisfied in states where the option is enabled
        """
        super(DBNOptionEnsemble, self).__init__(name, [], [])
        self._ensemble_size = ensemble_size
        self._horizon = horizon
        self._sf_list = sf_list
        self._enabled_cond = enabled_cond
        self._dbns = [None] * self._ensemble_size
        self._transition_mats = [None] * self._ensemble_size
        self._setup_ensemble(data)

    def get_transition_prob(self, state, next_state):
        """Return the exploration probability for a (s,s') pair.

        This is sampled uniformly from one of the ensemble models

        Args:
            state: The first state
            next_state: The next state

        Returns:
            The transition probability
        """
        return random.choice(self._dbns).get_transition_prob(state, next_state)

    def get_reward(self, state):
        """Return the reward for executing this option in a state.

        The reward is the entropy of the average minus the average entropy.

        Args:
            state: The state we want to check

        Returns:
            The reward for the state
        """
        # TODO: Fill in
        pass

    def get_scxml_transitions(self, sf_names, policy_name):
        """Return a list of SCXML transition elements for this option.

        Args:
            sf_names: The list of state factor names
            policy_name: The name of the policy in SCXML

        Returns:
            A list of SCXML transition elements
        """
        # TODO: Fill in
        pass

    def get_transition_prism_string(self):
        """Write out the PRISM string with all (sampled) transitions.

        Returns:
            The transition PRISM string
        """
        # TODO: Fill in
        pass

    def get_reward_prism_string(self):
        """Write out the PRISM string with all exploration rewards.

        The reward is the entropy of the average minus the average entropy.

        Returns:
            The reward PRISM string
        """
        # TODO: Fill in
        pass

    def _create_datasets(self, data):
        """Create the datasets for each model in the ensemble for DBN learning.

        Args:
            data: A dictionary of data items to learn from

        Returns:
            datasets: A list of length self._ensemble_size with each dataset
        """
        datasets = [{}] * self._ensemble_size
        for i in range(self._ensemble_size):  # Initialise datasets
            _initialise_dict_for_option(datasets[i], self._name, self._sf_list)
            datasets[i] = datasets[i][self._name]

        for i in range(len(data["reward"]["r"])):
            dbn_idx = i % self._ensemble_size

            for sf in self._sf_list:
                sf_name = sf.get_name()
                sf_0_name = "{}0".format(sf_name)
                sf_t_name = "{}t".format(sf_name)
                datasets[dbn_idx]["transition"][sf_0_name].append(
                    data["transition"][sf_0_name][i]
                )
                datasets[dbn_idx]["transition"][sf_t_name].append(
                    data["transition"][sf_t_name][i]
                )
                datasets[dbn_idx]["reward"][sf_name].append(data["reward"][sf_name][i])

            datasets[dbn_idx]["reward"]["r"].append(data["reward"]["r"][i])

        return datasets

    def _learn_dbn_options(self, datasets):
        """Function for learning DBN options given individual datasets.

        Args:
            datasets: A list of length self._ensemble_size with each dataset.
        """
        for i in range(len(datasets)):
            trans_dbn, reward_bn = learn_bns_for_one_option(datasets[i], self._sf_list)
            self._dbns[i] = DBNOption(
                self._name,
                None,
                None,
                self._sf_list,
                self._enabled_cond,
                prune_dists=True,
                transition_dbn=trans_dbn,
                reward_dbn=reward_bn,
            )

    def _setup_ensemble(self, data):
        """Set up the ensemble using the available data.

        Args:
            data: A dictionary of new data items to learn from
        """
        # Step 1: Split data evenly amongst ensemble
        datasets = self._create_datasets(data)
        # Step 2: Learn new DBNs
        self._learn_dbn_options()
        # Step 3: Do processing for transition/reward computation
        # TODO: Fill in
