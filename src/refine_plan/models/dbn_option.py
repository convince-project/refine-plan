#!/usr/bin/env python3
""" Subclass of Option for options represented using dynamic Bayesian networks.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import BoolStateFactor, IntStateFactor
from refine_plan.models.option import Option
from refine_plan.models.state import State
import pyAgrum as gum
import numpy as np
import itertools


class DBNOption(Option):
    """A subclass of options represented using dynamic Bayesian networks.

    DBNs are passed in as files using the .bifxml format used by pyAgrum.
    The DBNs assume all state factor values are strings.
    This is to stop lookup errors with integers.
    States can be passed in without considering this however.
    The transition function uses up to two variables for each state factor sf: sf0 and sft.
    The reward function should just use one variable per state factor, and one more called r
    which represents the reward.
    Some state factors may not appear in the transition or reward DBNs.

    Attributes:
        Same as superclass, plus:
        _transition_dbn: The transition function DBN.
        _reward_dbn: The reward function DBN.
        _sf_list: The list of state factors that make up the state space
        _is_enabled: A function which returns whether the option is enabled in a state
    """

    def __init__(
        self,
        name,
        transition_dbn_path,
        reward_dbn_path,
        sf_list,
        is_enabled,
        prune_dists=True,
    ):
        """Initialise attributes.

        Args:
            name: The option's name
            transition_dbn_path: A path to a .bifxml file for the transition DBN.
            reward_dbn_path: A path to a .bifxml file for the reward function DBN.
            sf_list: The list of state factors that make up the state space
            is_enabled: A function which returns whether the option is enabled in a state
            prune_dists: If True, remove small probs in transition DBN and renormalise
        """
        super(DBNOption, self).__init__(name, [], [])
        self._transition_dbn = gum.loadBN(transition_dbn_path)
        self._reward_dbn = gum.loadBN(reward_dbn_path)
        self._sf_list = sf_list
        self._is_enabled = is_enabled
        self._check_valid_dbns()

        if prune_dists:
            self._prune_dists()

    def _check_valid_dbns(self):
        """Check the DBNs have values for all state factors.

        Raises:
            invalid_dbn_exception: Raised if invalid DBNs are provided
        """
        sf_names = [sf.get_name() for sf in self._sf_list]

        # Step 1: Check transition function has correct state factors
        zero_names, t_names = [], []
        for var in self._transition_dbn.names():
            if var[-1] == "0":
                zero_names.append(var[:-1])
            elif var[-1] == "t":
                t_names.append(var[:-1])
            else:
                raise Exception("Invalid variable in transition DBN: {}".format(var))

        if not (set(zero_names) <= set(sf_names) and set(t_names) <= set(sf_names)):
            raise Exception("Transition variables do not match with state factors")

        # Step 2: Check the reward function has the correct variables
        r_found, r_names = False, []
        for var in self._reward_dbn.names():
            if var == "r":
                r_found = True
            else:
                r_names.append(var)

        if not r_found:
            raise Exception("No r variable in reward DBN")

        if not (set(r_names) <= set(sf_names)):
            raise Exception("Reward variables do not match with state factors")

        # Step 3: Check the values for each state variable
        # The values for each DBN node should be a subset of the state factor values
        for i in range(len(sf_names)):
            name = sf_names[i]
            vals = set([str(v) for v in self._sf_list[i].get_valid_values()])
            name_0 = "{}0".format(name)
            name_t = "{}t".format(name)

            if name_0 in self._transition_dbn.names():
                trans_0_values = set(self._transition_dbn[name_0].labels())
            else:
                trans_0_values = set([])

            if name_t in self._transition_dbn.names():
                trans_t_values = set(self._transition_dbn[name_t].labels())
            else:
                trans_t_values = set([])

            if name in self._reward_dbn.names():
                r_values = set(self._reward_dbn[name].labels())
            else:
                r_values = set([])

            # Python sets can use <= for subset checks
            if not (
                trans_0_values <= vals and trans_t_values <= vals and r_values <= vals
            ):
                raise Exception("State factor values don't match with those in DBNs")

    def _prune_dists(self, prune_threshold=1e-4):
        """Prune small probabilities from transition DBN and renormalise.

        Args:
            prune_threshold: The point at which to set probabilities to zero.
        """
        sf_names = list(filter(lambda x: x[-1] == "t", self._transition_dbn.names()))
        for var in sf_names:
            parents = [
                self._transition_dbn[p].name()
                for p in self._transition_dbn.parents(var)
            ]
            instance = gum.Instantiation()
            instance.addVarsFromModel(self._transition_dbn, parents)

            while not instance.end():
                dist = self._transition_dbn.cpt(var)[instance.todict()]
                norm_dist = [0.0 if v <= prune_threshold else v for v in dist]
                norm_dist = [v / np.sum(norm_dist) for v in norm_dist]
                self._transition_dbn.cpt(var)[instance.todict()] = norm_dist
                instance.inc()

    def _expected_val_fn(self, x):
        """The auxiliary function for computing the expected reward value.

        Args:
            x: A dictionary from state factor name to value index

        Returns:
            The corresponding state factor value for r (a numerical value)
        """
        return float(self._reward_dbn["r"].labels()[x["r"]])

    def _get_independent_groups(self):
        """Compute the set of independent transition DBN variable groups.

        The variables considered are the successor variables suffixed with t.
        Variables between groups are independent.
        Variables within a group are dependent.

        DEPRECATED: FUNCTION NOT IN USE

        Returns:
            A list of sets of variable names
        """
        succ_vars = set([var for var in self._transition_dbn.names() if var[-1] == "t"])
        pre_vars = [var for var in self._transition_dbn.names() if var[-1] == "0"]

        groups = []
        current_group = set([])

        while len(succ_vars) > 0:
            for var in succ_vars:
                if current_group == []:
                    current_group.add(var)
                elif not self._transition_dbn.isIndependent(
                    current_group[0], var, pre_vars
                ):
                    current_group.add(var)

            groups.append(current_group)
            succ_vars = succ_vars.difference(current_group)

        return groups

    def _get_parents_for_groups(self, groups):
        """Get the previous state variables for each group of next state variables.

        Args:
            groups: A list of sets of variable names

        DEPRECATED: FUNCTION NOT IN USE

        Returns:
            A list of sets of variable names (the parents)
        """
        succ_vars = set([var for var in self._transition_dbn.names() if var[-1] == "t"])
        parents = []

        for group in groups:
            parent_set = set([])
            for var in group:
                parent_set.update(self._transition_dbn.parents(var))

            parent_set = set([[self._transition_dbn[p].name() for p in parent_set]])

            # Remove any 't' variables
            parents.append(parent_set.difference(succ_vars))

        return parents

    def _str_to_sf_vals(self, str_sf_vals, sfs_used):
        """Converts a list of list of strings into appropriate state factor values.

        PyAgrum converts all values into strings.
        We need to convert them back to the correct state factor values (e.g. bool/int).

        Args:
            str_sf_vals: A list of lists of values. The list must follow the order of self._sf_list.
            sfs_used: A list of state factors. sfs_used[i] is the state factor for str_sf_vals[i].

        Returns:
            A list of lists of state factor values
        """
        sf_vals = []

        assert len(str_sf_vals) == len(sfs_used)

        for i in range(len(str_sf_vals)):
            if isinstance(sfs_used[i], BoolStateFactor) or isinstance(
                sfs_used[i], IntStateFactor
            ):
                sf_vals.append([eval(v) for v in str_sf_vals[i]])
            else:  # StateFactor -> string values (so eval doesn't work)
                sf_vals.append([str(v) for v in str_sf_vals[i]])
        return sf_vals

    def get_transition_prob(self, state, next_state):
        """Return the transition probability for an (s,s') pair.

        Args:
            state: The first state
            next_state: The next state

        Returns:
            The transition probability
        """

        # Check option is enabled in state
        if not self._is_enabled(state):
            return 0.0

        # The state is the prior - only use state factors in the transition DBN
        evidence = {}
        for sf in state._state_dict:
            name_0 = "{}0".format(sf)
            if name_0 in self._transition_dbn.names():
                evidence[name_0] = str(state[sf])

        # Find the target variables and those which should remain unchanged
        # SFs without a {}t variable in the DBN should remain unchanged.
        target = set([])
        unchanged = []
        for sf in next_state._state_dict:
            name_t = "{}t".format(sf)
            if name_t in self._transition_dbn.names():
                target.add(name_t)
            else:
                unchanged.append(sf)

        # Check unchanged variables are unchanged
        for sf in unchanged:
            if state[sf] != next_state[sf]:
                return 0.0

        # Create the inference object, set evidence, and set the posterior target
        inf_eng = gum.LazyPropagation(self._transition_dbn)
        inf_eng.setEvidence(evidence)
        target = set(["{}t".format(sf[:-1]) for sf in evidence.keys()])
        inf_eng.addJointTarget(target)

        posterior = inf_eng.jointPosterior(target)

        # The successor state needs to be written as a pyAgrum Instantiation
        pyagrum_next_state = gum.Instantiation()
        pyagrum_next_state.addVarsFromModel(self._transition_dbn, list(target))
        next_state_dict = {
            "{}t".format(sf): str(next_state[sf]) for sf in next_state._state_dict
        }
        pyagrum_next_state.fromdict(next_state_dict)

        return posterior.get(pyagrum_next_state)

    def get_reward(self, state):
        """Return the reward for executing an option in a given state.

        Args:
            state: The state we want the reward for

        Returns:
            The reward at the state
        """
        # Check option is enabled in state
        if not self._is_enabled(state):
            return 0.0

        # The state is the prior in the reward DBN
        evidence = {}
        for sf in state._state_dict:
            if sf in self._reward_dbn.names():  # Only use SFs that appear in the DBN
                evidence[sf] = str(state[sf])

        # Create the inference engine and set the evidence
        inf_eng = gum.LazyPropagation(self._reward_dbn)
        inf_eng.setEvidence(evidence)

        return inf_eng.jointPosterior(set("r")).expectedValue(self._expected_val_fn)

    def _get_vals_and_sfs(self, dbn, var_suffix):
        """Gets all values for a set of DBN nodes alongside the SFs.

        The values are given as a list of lists. Each sub-list has all values for that
        state factor in the DBN.

        Args:
            dbn: The DBN we're getting values from
            var_suffix: The suffix on the variable name after the state factor

        Returns:
            A list of variable names, a list of lists of values, and a list
            of the corresponding state factors
        """
        vars_used = []
        sfs_used = []
        for sf in self._sf_list:
            var_name = sf.get_name() + var_suffix
            if var_name in dbn.names():
                vars_used.append(var_name)
                sfs_used.append(sf)
        vals = self._str_to_sf_vals(
            [list(dbn[v].labels()) for v in vars_used], sfs_used
        )

        return vars_used, vals, sfs_used

    def get_transition_prism_string(self):
        """Return a PRISM string which captures all transitions for this option.

        Returns:
            The transition PRISM string
        """
        prism_str = ""
        inf_eng = gum.LazyPropagation(self._transition_dbn)

        ev_vars, pre_iterator, ev_sfs_used = self._get_vals_and_sfs(
            self._transition_dbn, "0"
        )
        target, post_iterator, target_sfs_used = self._get_vals_and_sfs(
            self._transition_dbn, "t"
        )

        inf_eng.addJointTarget(set(target))
        instantiation = gum.Instantiation()
        instantiation.addVarsFromModel(self._transition_dbn, target)

        for pre_state_vals in itertools.product(*pre_iterator):
            # Build the predecessor state object
            pre_state_dict, evidence = {}, {}
            for i in range(len(pre_state_vals)):
                pre_state_dict[ev_sfs_used[i]] = pre_state_vals[i]
                evidence[ev_vars[i]] = str(pre_state_vals[i])
            pre_state = State(pre_state_dict)

            # Check action is enabled
            if not self._is_enabled(pre_state):
                continue

            inf_eng.setEvidence(evidence)  # Setting evidence for posterior
            posterior = inf_eng.jointPosterior(set(target))
            post_cond_str = ""

            for next_state_vals in itertools.product(*post_iterator):
                # Build the successor state object
                next_state_dict, inst_dict = {}, {}
                for i in range(len(next_state_vals)):
                    next_state_dict[target_sfs_used[i]] = next_state_vals[i]
                    inst_dict[target[i]] = str(next_state_vals[i])
                next_state = State(next_state_dict)

                instantiation.fromdict(inst_dict)
                prob = posterior.get(instantiation)
                zero_cost_self_loop = False
                # Remove zero cost self loops for Storm (correspond to non-enabled actions)
                if pre_state == next_state and np.isclose(prob, 1.0):
                    zero_cost_self_loop = np.isclose(self.get_reward(pre_state), 0.0)
                if not np.isclose(prob, 0.0) and not zero_cost_self_loop:
                    post_cond_str += "{}:{} + ".format(
                        posterior.get(instantiation),
                        next_state.to_and_cond().to_prism_string(is_post_cond=True),
                    )

            if post_cond_str != "":  # Only write out if there are valid transitions
                prism_str += "[{}] {} -> ".format(  # Write precondition to PRISM
                    self.get_name(), pre_state.to_and_cond().to_prism_string()
                )
                prism_str += post_cond_str
                prism_str = prism_str[:-3] + "; \n"  # Remove final " + "
            inf_eng.eraseAllEvidence()

        return prism_str

    def get_reward_prism_string(self):
        """Return a PRISM string which captures all rewards for this option.

        Returns:
            The reward PRISM string
        """

        # Setup DBN inference
        inf_eng = gum.LazyPropagation(self._reward_dbn)
        prism_str = ""

        # Get all states
        _, sf_vals, sfs_used = self._get_vals_and_sfs(self._reward_dbn, "")

        for state_vals in itertools.product(*sf_vals):

            # Build the current state object
            state_dict = {}
            for i in range(len(state_vals)):
                state_dict[sfs_used[i]] = state_vals[i]
            state = State(state_dict)

            # If not enabled in this state, move to next state
            if not self._is_enabled(state):
                continue

            # Get the reward
            evidence = {sf.get_name(): state_dict[sf] for sf in state_dict}
            inf_eng.setEvidence(evidence)
            r = inf_eng.jointPosterior(set("r")).expectedValue(self._expected_val_fn)
            inf_eng.eraseAllEvidence()

            # Add to the PRISM string
            if not np.isclose(r, 0.0):
                prism_str += "[{}] {}: {};\n".format(
                    self.get_name(), state.to_and_cond().to_prism_string(), r
                )

        return prism_str
