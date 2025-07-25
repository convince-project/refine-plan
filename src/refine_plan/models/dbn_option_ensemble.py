#!/usr/bin/env python3
"""A class for an ensemble of DBNOption models.

This is used for active exploration.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.learning.option_learning import _initialise_dict_for_option
from refine_plan.models.dbn_option import DBNOption
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
        _datasets: The current datasets being used for each DBNOption.
    """

    def __init__(self, name, ensemble_size, horizon, sf_list, enabled_cond):
        """Initialise attributes.

        Args:
            name: The option's name
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
        self._datasets = [{}] * self._ensemble_size
        for i in range(self._ensemble_size):  # Initialise datasets
            _initialise_dict_for_option(self._datasets[i], self._name, sf_list)
            self._datasets[i] = self._datasets[i][self._name]

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

    def _update_datasets(self, new_data):
        """Update the datasets for DBN learning.

        Args:
            new_data: A dictionary of new data items to learn from
        """
        for i in range(len(new_data["reward"]["r"])):
            dbn_idx = i % self._ensemble_size

            for sf in self._sf_list:
                sf_name = sf.get_name()
                sf_0_name = "{}0".format(sf_name)
                sf_t_name = "{}t".format(sf_name)
                self._datasets[dbn_idx]["transition"][sf_0_name].append(
                    new_data["transition"][sf_0_name][i]
                )
                self._datasets[dbn_idx]["transition"][sf_t_name].append(
                    new_data["transition"][sf_t_name][i]
                )
                self._datasets[dbn_idx]["reward"][sf_name].append(
                    new_data["reward"][sf_name][i]
                )

            self._datasets[dbn_idx]["reward"]["r"].append(new_data["reward"]["r"][i])

    def update_ensemble(self, new_data):
        """Update the ensemble using new data.

        Args:
            new_data: A dictionary of new data items to learn from
        """
        # Step 1: Split data evenly amongst ensemble
        self._update_datasets(new_data)
        # Step 2: Learn new DBNs
        # TODO: Fill in
        # Step 3: Do processing for transition/reward computation
        # TODO: Fill in
