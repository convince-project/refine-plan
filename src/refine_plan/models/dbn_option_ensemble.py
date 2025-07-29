#!/usr/bin/env python3
"""A class for an ensemble of DBNOption models.

This is used for active exploration.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.condition import LtCondition, AddCondition, AndCondition
from refine_plan.models.state_factor import IntStateFactor
from refine_plan.models.dbn_option import DBNOption
from refine_plan.learning.option_learning import (
    _initialise_dict_for_option,
    learn_bns_for_one_option,
)
from refine_plan.models.option import Option
from math import log
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
        _transition_dicts: The corresponding transition dicts for each DBNOption.
        _sampled_transition_dict: The sampled transitions
        _reward_dict: The reward dictionary containing information gain values
        _transition_prism_str: The transition PRISM string, cached
        _reward_prism_str: The reward PRISM string, cached
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
        self._transition_dicts = [None] * self._ensemble_size
        self._sampled_transition_dict = {}
        self._reward_dict = {}
        self._transition_prism_str = None
        self._reward_prism_str = None
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
        # TODO: Make consistent with rest of code
        return random.choice(self._dbns).get_transition_prob(state, next_state)

    def get_reward(self, state):
        """Return the reward for executing this option in a state.

        The reward is the entropy of the average minus the average entropy.

        Args:
            state: The state we want to check

        Returns:
            The reward for the state
        """
        return self._reward_dict[state]

    def get_scxml_transitions(self, sf_names, policy_name):
        """Return a list of SCXML transition elements for this option.

        The time state factor is not included here, that is only for PRISM to
        facilitate the finite horizon planning objective.

        Args:
            sf_names: The list of state factor names
            policy_name: The name of the policy in SCXML

        Returns:
            A list of SCXML transition elements
        """
        transitions = []

        for state in self._sampled_transition_dict:
            pre_cond = state.to_and_cond()
            scxml_trans = self._build_single_scxml_transition(
                pre_cond, self._sampled_transition_dict[state], sf_names, policy_name
            )
            transitions.append(scxml_trans)

        return transitions

    def get_transition_prism_string(self):
        """Write out the PRISM string with all (sampled) transitions.

        Returns:
            The transition PRISM string
        """
        assert self._transition_prism_str is not None
        return self._transition_prism_str

    def get_reward_prism_string(self):
        """Write out the PRISM string with all exploration rewards.

        The reward is the entropy of the average minus the average entropy.

        Returns:
            The reward PRISM string
        """
        assert self._reward_prism_str is not None
        return self._reward_prism_str

    def _compute_entropy(self, dist):
        """Compute the Shannon entropy for a distribution.

        Args:
            dist: A dictionary from event to probability

        Returns:
            The Shannon entropy
        """
        entropy = 0
        for event in dist:
            entropy += dist[event] * log(dist[event], 2)
        return -1 * entropy

    def _compute_avg_dist(self, state):
        """Compute the average transition distribution for a state.

        Args:
            state: The state to compute the average for

        Returns:
            The average distribution (from post cond to prob)
        """
        avg_dist = {}
        for i in range(self._ensemble_size):
            for post_cond in self._transition_dicts[i][state]:
                if post_cond not in avg_dist:
                    avg_dist[post_cond] = 0
                avg_dist[post_cond] += (  # Do division incrementally to save time
                    self._transition_dicts[i][state][post_cond] / self._ensemble_size
                )

        return avg_dist

    def _compute_info_gain(self, state):
        """Compute the information gain for a state.

        Args:
            state: The state to compute the info gain for

        Returns:
            The information gain (approximation of the Jensen-Shannon Divergence)
        """
        entropy_of_avg = self._compute_entropy(self._compute_avg_dist(state))

        avg_entropy = 0
        for i in range(self._ensemble_size):
            avg_entropy += self._compute_entropy(self._transition_dicts[i][state])
        avg_entropy /= self._ensemble_size

        return entropy_of_avg - avg_entropy

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

    def _precompute_prism_strings(self):
        """Precompute the transition and reward PRISM strings.

        For the PRISM model, we introduce a new state factor time to facilitate
        a finite horizon model.
        """
        time_sf = IntStateFactor("time", 0, self._horizon)
        time_guard = LtCondition(time_sf, self._horizon)
        time_inc = AddCondition(time_sf, 1)
        self._transition_prism_str, self._reward_prism_str = "", ""
        assert len(self._sampled_transition_dict) == len(self._reward_dict)

        for state in self._sampled_transition_dict:
            pre_cond = state.to_and_cond()
            pre_cond.add_cond(time_guard)

            # Add to transition PRISM string
            self._transition_prism_str += "[{}] {} -> ".format(
                self.get_name(), pre_cond.to_prism_string()
            )

            for post_cond in self._sampled_transition_dict[state]:
                if not isinstance(post_cond, AndCondition):
                    post_cond = AndCondition(post_cond)
                post_cond.add_cond(time_inc)

                self._transition_prism_str += "{}:{} + ".format(
                    self._sampled_transition_dict[state][post_cond],
                    post_cond.to_prism_string(is_post_cond=True),
                )

            self._transition_prism_str = self._transition_prism_str[:-3] + ";\n"

            # Add to reward PRISM string
            self._reward_prism_str += "[{}] {}: {};\n".format(
                self.get_name(), pre_cond.to_prism_string(), self._reward_dict[state]
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
        # Step 4: Precompute PRISM strings
        self._precompute_prism_strings()
