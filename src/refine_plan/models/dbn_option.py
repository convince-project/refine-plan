#!/usr/bin/env python3
"""Subclass of Option for options represented using dynamic Bayesian networks.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import BoolStateFactor, IntStateFactor
from pyeda.boolalg.expr import Complement, Variable, AndOp, OrOp
from refine_plan.models.option import Option
from refine_plan.models.state import State
from refine_plan.models.condition import (
    TrueCondition,
    NotCondition,
    AddCondition,
    AndCondition,
    OrCondition,
)
from pyeda.inter import expr
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
        _enabled_cond: A Condition which captures the states where this option is enabled
    """

    def __init__(
        self,
        name,
        transition_dbn_path,
        reward_dbn_path,
        sf_list,
        enabled_cond,
        prune_dists=True,
        transition_dbn=None,
        reward_dbn=None,
    ):
        """Initialise attributes.

        Args:
            name: The option's name
            transition_dbn_path: A path to a .bifxml file for the transition DBN.
            reward_dbn_path: A path to a .bifxml file for the reward function DBN.
            sf_list: The list of state factors that make up the state space
            enabled_cond: A Condition which captures the states where this option is enabled
            prune_dists: If True, remove small probs in transition DBN and renormalise
            transition_dbn: If set, use an existing DBN rather than that at transition_dbn_path
            reward_dbn: If set, use an existing BN rather than that at reward_dbn_path
        """
        super(DBNOption, self).__init__(name, [], [])
        if transition_dbn is None:
            self._transition_dbn = gum.loadBN(transition_dbn_path)
        else:
            self._transition_dbn = transition_dbn
            assert isinstance(self._transition_dbn, gum.pyAgrum.BayesNet)
        if reward_dbn is None:
            self._reward_dbn = gum.loadBN(reward_dbn_path)
        else:
            self._reward_dbn = reward_dbn
            assert isinstance(self._reward_dbn, gum.pyAgrum.BayesNet)
        self._sf_list = sf_list
        assert enabled_cond.is_pre_cond()
        self._enabled_cond = enabled_cond
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
            name_0, name_t = "{}0".format(name), "{}t".format(name)

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

    def _prune_dists(self, prune_threshold=1e-3):
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
        if not self._enabled_cond.is_satisfied(state):
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
        inf_eng.addJointTarget(target)

        posterior = inf_eng.jointPosterior(target)
        self._prune_posterior(posterior)

        # The successor state needs to be written as a pyAgrum Instantiation
        pyagrum_next_state = gum.Instantiation()
        pyagrum_next_state.addVarsFromModel(self._transition_dbn, list(target))
        next_state_dict = {var: str(next_state[var[:-1]]) for var in target}
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
        if not self._enabled_cond.is_satisfied(state):
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

    def _sub_state_info_into_enabled_cond(self, state, cond):
        """Substitute the truth values from a state into the enabled condition.

        Parts of the enabled condition which hold/don't hold will be set to True/False.

        Args:
            state: The (partial) state
            cond: The portion of self._enabled_cond

        Returns:
            A modified version of self._enabled_cond with truth values inserted
        """

        if isinstance(cond, TrueCondition):
            return TrueCondition()
        elif isinstance(cond, NotCondition):
            return NotCondition(
                self._sub_state_info_into_enabled_cond(state, cond._cond)
            )
        elif isinstance(cond, AddCondition):
            raise Exception("AddCondition cannot be used for self._enabled_cond")
        elif isinstance(cond, AndCondition):
            new_and = AndCondition()
            for c in cond._cond_list:
                new_and.add_cond(self._sub_state_info_into_enabled_cond(state, c))
            return new_and
        elif isinstance(cond, OrCondition):
            new_or = OrCondition()
            for c in cond._cond_list:
                new_or.add_cond(self._sub_state_info_into_enabled_cond(state, c))
            return new_or
        else:  # These are all single state factor conditions, e.g. ==, !=, <, <=, >, >=
            if cond._sf.get_name() not in state._state_dict:
                return cond

            if cond.is_satisfied(state):
                return TrueCondition()
            else:
                return NotCondition(TrueCondition())

    def _pyeda_to_cond(self, pyeda_expr, var_map):
        """Convert a Pyeda expression into a Condition object.

        Args:
            pyeda_expr: The Pyeda expression
            var_map: A map from Pyeda variables to leaf Condition objects

        Returns:
            The corresponding Condition object
        """

        if pyeda_expr == expr(True):
            return TrueCondition()
        elif pyeda_expr == expr(False):
            return NotCondition(TrueCondition())
        elif isinstance(pyeda_expr, Variable):
            return var_map[str(pyeda_expr)]
        elif isinstance(pyeda_expr, Complement):
            return NotCondition(self._pyeda_to_cond(pyeda_expr.inputs[0], var_map))
        elif isinstance(pyeda_expr, AndOp):
            and_cond = AndCondition()
            for sub_expr in pyeda_expr.xs:
                and_cond.add_cond(self._pyeda_to_cond(sub_expr, var_map))
            return and_cond
        elif isinstance(pyeda_expr, OrOp):
            or_cond = OrCondition()
            for sub_expr in pyeda_expr.xs:
                or_cond.add_cond(self._pyeda_to_cond(sub_expr, var_map))
            return or_cond

    def _get_prism_guard_for_state(self, state):
        """Get the PRISM guard for a state.

        This is the conjunction of the state with anything that still
        needs to be satisfied in self._enabled_cond.

        If state is incompatible with self._enabled_cond, this returns None

        Args:
            state: The (partial) state we want the guard for

        Returns:
            The guard for this state in the PRISM string, or None
        """

        # Step 1: Sub state info into enabled cond
        subbed_cond = self._sub_state_info_into_enabled_cond(state, self._enabled_cond)

        # Step 2: Convert to pyeda, which implicitly simplifies the expression
        pyeda_expr, var_map = subbed_cond.to_pyeda_expr()

        # Step 3: Bail now if the pyeda expr is simply True or False
        guard = state.to_and_cond()
        if pyeda_expr == expr(True):
            return guard
        elif pyeda_expr == expr(False):  # Can't satisfy the enabled condition
            return None

        # Step 3: Convert the pyeda expression back into a condition
        remaining_cond = self._pyeda_to_cond(pyeda_expr, var_map)

        # Step 4: Compute the guard
        if isinstance(remaining_cond, AndCondition):  # Avoid unnecessary nesting
            for cond in remaining_cond._cond_list:
                guard.add_cond(cond)
        else:
            guard.add_cond(remaining_cond)

        return guard

    def _prune_posterior(self, posterior, threshold=1e-2):
        """Prune small probabilities from a posterior and re-normalise.

        Args:
            posterior: The posterior as a PyAgrum Potential object
            threshold: The pruning threshold
        """
        for i in posterior.loopIn():
            if posterior.get(i) <= threshold:
                posterior.set(i, 0.0)

        posterior.normalize()

    def get_pre_post_cond_pairs(self):
        """Return a list of (pre, prob_post_cond) pairs from the DBN.

        Returns:
            A list of (pre, prob_post_cond) pairs
        """
        cond_pairs = []
        inf_eng = gum.LazyPropagation(self._transition_dbn)

        ev_vars, pre_iterator, ev_sfs_used = self._get_vals_and_sfs(
            self._transition_dbn, "0"
        )
        target, _, trg_sfs_used = self._get_vals_and_sfs(self._transition_dbn, "t")

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

            # Check action is enabled and get the enabled condition for this state
            # We need this as during learning, all options were executed only in enabled states
            # We need to capture this here by adding on the 'rest' of the enabled condition
            pre_cond = self._get_prism_guard_for_state(pre_state)
            if pre_cond is None:
                continue

            inf_eng.setEvidence(evidence)  # Setting evidence for posterior
            posterior = inf_eng.jointPosterior(set(target))
            self._prune_posterior(posterior)
            # Get all states with non-zero probabilities
            non_zeros = posterior.isNonZeroMap().findAll(1.0)
            prob_post_conds = {}

            for next_state_vals in non_zeros:
                # Build the successor state object
                next_state_dict = {}
                for sf in trg_sfs_used:
                    var = "{}t".format(sf.get_name())
                    label_as_str = self._transition_dbn[var].label(next_state_vals[var])
                    sf_val = self._str_to_sf_vals([[label_as_str]], [sf])[0][0]
                    next_state_dict[sf] = sf_val

                next_state = State(next_state_dict)

                instantiation.fromdict(next_state_vals)
                prob = posterior.get(instantiation)
                zero_cost_self_loop = False
                # Remove zero cost self loops for Storm (correspond to non-enabled actions)
                if pre_state == next_state and np.isclose(prob, 1.0):
                    zero_cost_self_loop = np.isclose(self.get_reward(pre_state), 0.0)
                if not zero_cost_self_loop:
                    post_cond = next_state.to_and_cond()
                    prob_post_conds[post_cond] = posterior.get(instantiation)

            if len(prob_post_conds) > 0:  # Only write if valid transitions present
                cond_pairs.append((pre_cond, prob_post_conds))
            inf_eng.eraseAllEvidence()

        return cond_pairs

    def get_scxml_transitions(self, sf_names, policy_name):
        """Return a list of SCXML transition elements for this option.

        Args:
            sf_names: The list of state factor names
            policy_name: The name of the policy in SCXML

        Returns:
            A list of SCXML transition elements
        """
        transitions = []
        cond_pairs = self.get_pre_post_cond_pairs()
        for pair in cond_pairs:
            pre_cond, prob_post_conds = pair
            transitions.append(
                self._build_single_scxml_transition(
                    pre_cond, prob_post_conds, sf_names, policy_name
                )
            )

        return transitions

    def get_transition_prism_string(self):
        """Return a PRISM string which captures all transitions for this option.

        Returns:
            The transition PRISM string
        """
        prism_str = ""
        cond_pairs = self.get_pre_post_cond_pairs()
        for pair in cond_pairs:
            pre_cond, prob_post_conds = pair
            prism_str += "[{}] {} -> ".format(  # Write precondition to PRISM
                self.get_name(), pre_cond.to_prism_string()
            )

            for post_cond in prob_post_conds:
                prism_str += "{}:{} + ".format(
                    prob_post_conds[post_cond],
                    post_cond.to_prism_string(is_post_cond=True),
                )

            prism_str = prism_str[:-3] + "; \n"  # Remove final " + "

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

            # Check action is enabled and get the enabled condition for this state
            # We need this as during learning, all options were executed only in enabled states
            # We need to capture this here by adding on the 'rest' of the enabled condition
            state_cond = self._get_prism_guard_for_state(state)
            if state_cond is None:
                continue

            # Get the reward
            evidence = {sf.get_name(): state_dict[sf] for sf in state_dict}
            inf_eng.setEvidence(evidence)
            r = inf_eng.jointPosterior(set("r")).expectedValue(self._expected_val_fn)
            inf_eng.eraseAllEvidence()

            # Add to the PRISM string
            if not np.isclose(r, 0.0):
                prism_str += "[{}] {}: {};\n".format(
                    self.get_name(), state_cond.to_prism_string(), r
                )

        return prism_str
