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
from multiprocessing import Process, SimpleQueue
from refine_plan.models.option import Option
from refine_plan.models.state import State
from math import log
import numpy as np
import itertools
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
        _state_idx_map: A map from states to matrix indices
        _sampled_transition_mat: _sampled_transition_dict as a matrix
        _reward_mat: _reward_dict as a matrix
    """

    def __init__(
        self,
        name,
        data,
        ensemble_size,
        horizon,
        sf_list,
        enabled_cond,
        state_idx_map,
    ):
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
        self._state_idx_map = state_idx_map
        self._sampled_transition_mat = None
        self._reward_mat = None
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
        for post_cond in self._sampled_transition_dict[state]:
            if state.apply_post_cond(post_cond) == next_state:
                return self._sampled_transition_dict[state][post_cond]

        return 0.0

    def get_reward(self, state):
        """Return the reward for executing this option in a state.

        The reward is the entropy of the average minus the average entropy.

        Args:
            state: The state we want to check

        Returns:
            The reward for the state
        """
        return self._reward_dict[state] if state in self._reward_dict else 0.0

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

    def _build_transition_dict_for_dbn(self, dbn_idx, queue):
        """Build the transition dictionary for the DBN at dbn_idx.

        Args:
            dbn_idx: The DBN index
            queue: The queue to add the transition dictionary too
        """
        transition_dict = {}
        # Return partial predecessor states, not preconditions
        pre_post_pairs = self._dbns[dbn_idx].get_pre_post_cond_pairs(True)

        unused_sfs, unused_vals = [], []
        for sf in self._sf_list:
            if sf.get_name() + "0" not in self._dbns[dbn_idx]._transition_dbn.names():
                unused_sfs.append(sf)
                unused_vals.append(sf.get_valid_values())

        for pre_state, prob_post_conds in pre_post_pairs:
            for rest_of_state in itertools.product(*unused_vals):
                full_state_dict = {}
                for i in range(len(rest_of_state)):
                    full_state_dict[unused_sfs[i]] = rest_of_state[i]
                for sf_name in pre_state._state_dict:
                    full_state_dict[pre_state._sf_dict[sf_name]] = pre_state[sf_name]
                full_state = State(full_state_dict)

                if self._enabled_cond.is_satisfied(full_state):
                    transition_dict[full_state] = prob_post_conds

        queue.put((dbn_idx, transition_dict))

    def _build_transition_dicts(self):
        """Build all transition dictionaries for each model in parallel."""

        procs = []
        queue = SimpleQueue()
        # Create and start processes for each model in the ensemble
        for i in range(self._ensemble_size):
            procs.append(
                Process(target=self._build_transition_dict_for_dbn, args=(i, queue))
            )
            print("{}: Starting process for DBN {}".format(self.get_name(), i))
            procs[-1].start()

        # Update self._transition_dicts based on queue
        for _ in range(self._ensemble_size):
            idx, transition_dict = queue.get()
            self._transition_dicts[idx] = transition_dict
            print("{}: Received dict for DBN {}".format(self.get_name(), idx))

        # Wait for all parallel processes to finish
        for i in range(self._ensemble_size):
            procs[i].join()

    def _compute_sampled_transitions_and_info_gain(self):
        """Compute the transition and reward functions for the exploration MDP."""
        # TODO: This might break if each transition_dict has a different length
        # With enough data this would be fine as all variable values would be covered
        # in the DBN. However, in the early stages this may not be the case, causing
        # failures in this function (and those it calls)
        for state in self._transition_dicts[0]:
            self._sampled_transition_dict[state] = random.choice(
                self._transition_dicts
            )[state]
            info_gain = self._compute_info_gain(state)
            if not np.isclose(info_gain, 0.0):
                self._reward_dict[state] = info_gain

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
                trans_prob = self._sampled_transition_dict[state][post_cond]
                if not isinstance(post_cond, AndCondition):
                    post_cond = AndCondition(post_cond, time_inc)
                else:
                    post_cond = AndCondition(*(post_cond._cond_list + [time_inc]))

                self._transition_prism_str += "{}:{} + ".format(
                    trans_prob,
                    post_cond.to_prism_string(is_post_cond=True),
                )

            self._transition_prism_str = self._transition_prism_str[:-3] + ";\n"

            # Add to reward PRISM string
            self._reward_prism_str += "[{}] {}: {};\n".format(
                self.get_name(), pre_cond.to_prism_string(), self._reward_dict[state]
            )

    def _build_matrices(self):
        """Build the sampled transition matrix and reward vector."""
        num_states = len(self._state_idx_map)
        self._sampled_transition_mat = np.zeros((num_states, num_states))
        self._reward_mat = np.zeros(num_states)

        for state in self._state_idx_map:
            state_id = self._state_idx_map[state]
            if state in self._reward_dict:
                self._reward_mat[state_id] = self._reward_dict[state]

            if state in self._sampled_transition_dict:
                prob_post_conds = self._sampled_transition_dict[state]

                for post_cond in prob_post_conds:
                    next_state = state.apply_post_cond(post_cond)
                    prob = prob_post_conds[post_cond]
                    next_id = self._state_idx_map[next_state]
                    self._sampled_transition_mat[state_id, next_id] = prob

    def _setup_ensemble(self, data):
        """Set up the ensemble using the available data.

        Args:
            data: A dictionary of new data items to learn from
        """
        # Step 1: Split data evenly amongst ensemble
        print("{}: Splitting up datasets...".format(self.get_name()))
        datasets = self._create_datasets(data)
        # Step 2: Learn new DBNs
        print("{}: Learning DBNS...".format(self.get_name()))
        self._learn_dbn_options(datasets)
        # Step 3: Do preprocessing for transition/reward computation
        print("{}: Building transition dicts for each DBN".format(self.get_name()))
        self._build_transition_dicts()
        # Step 4: Compute sampled transition function and info gain reward function
        print("{}: Sampling transitions and computing reward".format(self.get_name()))
        self._compute_sampled_transitions_and_info_gain()
        # Step 5: Build transition and reward matrices
        print("{}: Building transition/reward matrices".format(self.get_name()))
        self._build_matrices()
