#!/usr/bin/env python3
""" Subclass of Option for options represented using dynamic Bayesian networks.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.option import Option
from refine_plan.models.state import State
import pyAgrum as gum
import itertools


class DBNOption(Option):
    """A subclass of options represented using dynamic Bayesian networks.

    DBNs are passed in as files using the .bifxml format used by pyAgrum.
    The DBNs assume all state factor values are strings.
    This is to stop lookup errors with integers.
    States can be passed in without considering this however.
    The transition function must use two variables for each state factor sf: sf0 and sft.
    The reward function should just use one variable per state factor, and one more called r
    which represents the reward.

    Attributes:
        Same as superclass, plus:
        _transition_dbn: The transition function DBN.
        _reward_dbn: The reward function DBN.
        _sf_list: A list of state factor objects that define the state space
    """

    def __init__(self, name, transition_dbn_path, reward_dbn_path, sf_list):
        """Initialise attributes.

        Args:
            name: The option's name
            transition_dbn_path: A path to a .bifxml file for the transition DBN.
            reward_dbn_path: A path to a .bifxml file for the reward function DBN.
            sf_list: The list of state factors that make up the state space
        """
        super(DBNOption, self).__init__(name, [], [])
        self._transition_dbn = gum.loadBN(transition_dbn_path)
        self._reward_dbn = gum.loadBN(reward_dbn_path)
        self._sf_list = sf_list
        self._check_valid_dbns()

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

        if set(zero_names) != set(sf_names) or set(t_names) != set(sf_names):
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

        if set(r_names) != set(sf_names):
            raise Exception("Reward variables do not match with state factors")

        # Step 3: Check the values for each state variable
        for i in range(len(sf_names)):
            name = sf_names[i]
            vals = set(self._sf_list[i].get_valid_values())
            trans_0_values = set(
                self._transition_dbn.variableFromName("{}0".format(name)).labels()
            )
            trans_t_values = set(
                self._transition_dbn.variableFromName("{}t".format(name)).labels()
            )
            r_values = set(self._reward_dbn.variableFromName(name).labels())

            if trans_0_values != vals or trans_t_values != vals or r_values != vals:
                raise Exception("State factor values don't match with those in DBNs")

    def _expected_val_fn(self, x):
        """The auxiliary function for computing the expected reward value.

        Args:
            x: A dictionary from state factor name to value index

        Returns:
            The corresponding state factor value for r (a numerical value)
        """
        return float(self._reward_dbn.variableFromName("r").labels()[x["r"]])

    def _get_independent_groups(self):
        """Compute the set of independent transition DBN variable groups.

        The variables considered are the successor variables suffixed with t.
        Variables between groups are independent.
        Variables within a group are dependent.

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

        Returns:
            A list of sets of variable names (the parents)
        """
        succ_vars = set([var for var in self._transition_dbn.names() if var[-1] == "t"])
        parents = []

        for group in groups:
            parent_set = set([])
            for var in group:
                parent_set.update(self._transition_dbn.parents(var))

            parent_set = set(
                [[self._transition_dbn.variable(p).name() for p in parent_set]]
            )

            # Remove any 't' variables
            parents.append(parent_set.difference(succ_vars))

        return parents

    def get_transition_prob(self, state, next_state):
        """Return the transition probability for an (s,s') pair.

        Args:
            state: The first state
            next_state: The next state

        Returns:
            The transition probability
        """
        # The state is the prior
        evidence = {"{}0".format(sf): str(state[sf]) for sf in state._state_dict}

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
        # The state is the prior in the reward DBN
        evidence = {sf: str(state[sf]) for sf in state._state_dict}

        # Create the inference engine and set the evidence
        inf_eng = gum.LazyPropagation(self._reward_dbn)
        inf_eng.setEvidence(evidence)

        return inf_eng.jointPosterior(set("r")).expectedValue(self._expected_val_fn)

    def get_transition_prism_string(self):
        """Return a PRISM string which captures all transitions for this option.

        Returns:
            The transition PRISM string
        """
        prism_str = ""
        inf_eng = gum.LazyPropagation(self._transition_dbn)
        groups = self._get_independent_groups()
        parents = self._get_parents_for_groups(groups)
        sf_dict = {}
        for sf in self._sf_list:
            sf_dict["{}0".format(sf.get_name())] = sf
            sf_dict["{}t".format(sf.get_name())] = sf

        for i in range(len(groups)):
            group = list(groups[i])
            parent_set = list(parents[i])

            parent_vals = [sf_dict[p].get_valid_values() for p in parent_set]
            group_vals = [sf_dict[var].get_valid_values() for var in group]
            inf_eng.addJointTarget(set(group))
            instantiation = gum.Instantiation()
            instantiation.addVarsFromModel(self._transition_dbn, group)

            for pre_state_vals in itertools.product(*parent_vals):
                # Build the predecessor state object
                pre_state_dict, evidence = {}, {}
                for i in range(len(pre_state_vals)):
                    pre_state_dict[sf_dict[parent_set[i]]] = pre_state_vals[i]
                    evidence[parent_set[i]] = str(pre_state_vals[i])
                pre_state = State(pre_state_dict)

                prism_str += "[{}] {} -> ".format(  # Write precondition to PRISM
                    self.get_name(), pre_state.to_and_cond().to_prism_string()
                )

                inf_eng.setEvidence(evidence)  # Setting evidence for posterior
                posterior = inf_eng.jointPosterior(set(group))

                for next_state_vals in itertools.product(*group_vals):
                    # Build the successor state object
                    next_state_dict, inst_dict = {}, {}
                    for i in range(len(next_state_vals)):
                        next_state_dict[sf_dict[group[i]]] = next_state_vals[i]
                        inst_dict[group[i]] = str(next_state_vals[i])
                    next_state = State(next_state_dict)

                    instantiation.fromdict(inst_dict)
                    prism_str += "{}:{} + ".format(
                        posterior.get(instantiation),
                        next_state.to_and_cond().to_prism_string(is_post_cond=True),
                    )

                prism_str = prism_str[:-3] + "; \n"  # Remove final " + "
                inf_eng.eraseAllEvidence()

            inf_eng.eraseAllJointTargets()  # Erase targets when we move to a different group

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
        sf_vals = [sf.get_valid_values() for sf in self._sf_list]
        for state_vals in itertools.product(*sf_vals):

            # Build the current state object
            state_dict = {}
            for i in range(len(state_vals)):
                state_dict[self._sf_list[i]] = state_vals[i]
            state = State(state_dict)

            # Get the reward
            evidence = {sf.get_name(): state_dict[sf] for sf in state_dict}
            inf_eng.setEvidence(evidence)
            r = inf_eng.jointPosterior(set("r")).expectedValue(self._expected_val_fn)
            inf_eng.eraseAllEvidence()

            # Add to the PRISM string
            prism_str += "[{}] {}: {};\n".format(
                self.get_name(), state.to_and_cond().to_prism_string(), r
            )
