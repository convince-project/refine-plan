#!/usr/bin/env python3
""" A converter from deterministic memoryless policies to behaviour trees.

This is an implementation of the work in:
Gugliermo, S., Schaffernicht, E., Koniaris, C. and Pecora, F., 2023. 
Learning behavior trees from planning experts using decision tree and logic 
factorization. IEEE Robotics and Automation Letters.

This code is reimplemented from: 
https://github.com/SimonaGug/BT-from-planning-experts

The above code is by the lead author of the above paper, and HAS NO LICENSE ATTACHED.
Though there is no license, the code is open source.

The reimplementation fixes a number of bugs in the linked implementation.
It also streamlines data conversions and makes other tidiness/quality of life
improvements.

Note that the linked repo could not be used directly for REFINE-PLAN; the 
reimplementation is necessary.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.condition import OrCondition
from pyeda.boolalg.expr import Complement, Variable
from refine_plan.models.policy import Policy
from pyeda.inter import espresso_exprs, Not
from sympy import Symbol, sympify


class PolicyBTConverter(object):
    """This class contains the functionality to convert a policy into a BT.

    Attributes:
        _vars_to_conds: A dictionary from variable names to Condition objects.
        _conds_to_vars: A dictionary from Condition objects to variable names.
        _vars_to_symbols: A dictionary form variable names to sympy symbols.
    """

    def __init__(self):
        """Reset all attributes."""
        self._reset()

    def _reset(self):
        """Reset all attributes to None."""
        self._vars_to_conds = None
        self._conds_to_vars = None
        self._vars_to_symbols = None

    def _extract_rules_from_policy(self, policy):
        """Turns a policy from S->A into a dictionary from A->logical rules.

        This involves grouping states by their policy action and combining them
        via disjunction.

        Args:
            policy: The policy to extract logical rules form

        Returns:
            act_to_rule: A dictionary from actions/options to a pyeda expression (the rule)
            act_to_var_map: A dictionary from actions/options to variable name to Condition

        Raises:
            not_a_policy: Raised if policy is not of type Policy
        """

        if not isinstance(policy, Policy):
            raise Exception("Invalid policy passed to _extract_rules_from_policy.")

        act_to_cond = {}

        for state in policy._state_action_dict:
            action = policy[state]
            if action not in act_to_cond:
                # This state or this state or this state etc.
                act_to_cond[action] = OrCondition(state.to_and_cond())
            else:
                act_to_cond[action].add_cond(state.to_and_cond())

        act_to_rule = {}
        act_to_var_map = {}

        for action in act_to_cond:
            pyeda_expr, var_map = act_to_cond[action].to_pyeda_expr()
            act_to_rule[action] = pyeda_expr
            act_to_var_map[action] = var_map

        return act_to_rule, act_to_var_map

    def _build_internal_mappings(self, act_to_var_map):
        """Construct the internal mappings used for BT conversion.

        Args:
            act_to_var_map: A mapping from action/option to var names to conditions
        """

        self._vars_to_conds = {}
        self._conds_to_vars = {}
        self._vars_to_symbols = {}

        for action in act_to_var_map:
            # Add all new variables to the global map
            var_map = act_to_var_map[action]
            for var in var_map:
                if var not in self._vars_to_conds:
                    self._vars_to_conds[var] = var_map[var]
                    self._vars_to_symbols[var] = Symbol(var)
                else:
                    # Given these come from the same policy, this should always pass
                    assert self._vars_to_conds[var] == var_map[var]
                    assert var in self._vars_to_symbols

        # self._conds_to_vars is just the reverse of self._vars_to_conds
        for var in self._vars_to_conds:
            self._conds_to_vars[self._vars_to_conds[var]] = var

    def _minimise_with_espresso(self, act_to_rule):
        """Minimise each rule using the espresso algorithm.

        Args:
            act_to_rule: A dictionary from action/option to pyeda expr

        Returns:
            act_to_min_rule: A dictionary from action/option to minimised pyeda expr
        """

        # The to_dnf() shouldn't make any difference to the original rule
        # As policy states are combined via disjunctions to get the rules
        # This is by definition in DNF (disjunction of conjunctions)
        # But I've kept it here as a sanity check as the original authors kept it
        # The [0] is as a singleton tuple is returned
        return {a: espresso_exprs(act_to_rule[a].to_dnf())[0] for a in act_to_rule}

    def _convert_to_horn_clauses(self, act_rule_dict):
        """Convert a set of rules into Horn clauses.

        let r be a rule and a be an action.
        If we have an implication r -> a, the Horn clause is of the form
        !r V a. For the BT conversion, we need to take the action a out as
        it needs special treatment in the BT construction.

        Therefore, here we just return a dictionary {a: !r} for all actions and
        corresponding rules.


        Args:
            act_rule_dict: A dictionary from action to (minimised) rules

        Returns:
            act_to_horn: A dictionary from action to Horn clause (still in pyeda)
        """
        # The two to_dnf() calls are to ensure equal behaviour with the repo
        # linked at the top of this file
        return {a: Not(act_rule_dict[a].to_dnf()).to_dnf() for a in act_rule_dict}

    def _get_variables_in_dnf_expr(self, rule):
        """Extract a list of variables used in a pyeda dnf expression/rule.

        Assumes rule is already in DNF.
        Unlike rule.inputs, duplicate variables are included in the list here.

        The paper suggests duplicates in a rule should be included. They don't
        do that in their code though, so I'm following the paper and counting
        duplicates.

        Args:
            rule: The pyeda expr to extract variables from

        Returns:
            var_list: A list of variables in the rule (duplicates included)

        Raises:
            not_dnf_exception: Raised if rule not in DNF
        """

        if not rule.is_dnf():
            raise Exception("Rule must be in dnf to extract variables")

        var_list = []

        for lit_set in rule._cover:
            for lit in lit_set:
                # There may be negated variables here,
                # I believe we need to care just about the literals occurring
                # and not what is done with them
                if isinstance(lit, Complement):
                    assert len(lit.inputs) == 1  # If in DNF, should always pass
                    var_list.append(lit.inputs[0])
                elif isinstance(lit, Variable):
                    var_list.append(lit)
                else:
                    raise Exception("Literal is not a negation or a variable.")

        return var_list

    def _score_and_sort_rules(self, act_to_horn):
        """Score and sort rules in descending score order.

        Here the score is the sum of occurrences of each literal in a rule
        amongst the rule set (duplicates in a single rule counted as per the paper).
        This is then divided by the number of unique literals in that set.

        Args:
            act_to_horn: A dictionary from action/option to horn clause

        Returns:
            ordered_ra_pairs: A list of (horn clause, action pairs), sorted
        """
        rule_act_pairs = []
        var_sets = {}
        frequencies = {}

        # First, compute the frequencies, literal sets, and rule-action pairs
        for action in act_to_horn:
            rule_act_pairs.append((act_to_horn[action], action))
            var_list = self._get_variables_in_dnf_expr(act_to_horn[action])

            for var in var_list:
                if var not in frequencies:
                    frequencies[var] = 0
                frequencies[var] += 1

            var_sets[action] = set(var_list)

        # Next, compute each rule's score (here dict keys are the corresponding action)
        scores = {}
        for action in act_to_horn:
            sum_of_frequencies = sum([frequencies[var] for var in var_sets[action]])
            scores[action] = sum_of_frequencies / len(var_sets[action])

        return sorted(rule_act_pairs, key=lambda ra: scores[ra[1]], reverse=True)

    def _logic_to_algebra(self, logic_ast, lit_map):
        """Converts a pyeda logical expr as an ast to a sympy algebraic expression.

        This is an auxiliary function which deals with AST recursion, hence the
        slightly odd input/output types.

        The function assumes logic_ast is in DNF, in particularly that only
        variables are negated.

        Args:
            logic_ast: A pyeda expression as an AST computed with to_ast()
            lit_map: A mapping from literals (e.g. "var1") to integers

        Returns:
            sympy_str: An algebraic expression (string) which can be sympified
        """

        if logic_ast[0] == "or" or logic_ast[0] == "and":
            sub_asts = logic_ast[1:]
            join_str = " + " if logic_ast[0] == "or" else "*"
            components = [self._logic_to_algebra(sub, lit_map) for sub in sub_asts]
            return components.join(join_str)
        elif logic_ast[0] == "lit":
            # Add to self._symbols
            lit = lit_map[logic_ast[1]]
            if isinstance(lit, Complement):
                # The original authors create auxiliary variables for factorisation
                # when a variable is negated.
                # I imagine this is because it's hard to translate the negation
                # cleanly into the algebraic form?
                assert len(lit.inputs) == 1
                var = str(lit.inputs[0])
                assert var[:3] != "NOT"  # Checking for bad variable names
                not_var = "NOT{}".format(var)
                if not_var not in self._symbols:
                    self._symbols[not_var] = Symbol(not_var)
                return not_var
            else:
                assert str(lit)[:3] != "NOT"  # Checking for bad variable names
                return str(lit)

    def _pyeda_rules_to_sympy_algebraic(self, ordered_ra_pairs):
        """Converts a set of pyeda rules into sympy algebraic expressions.

        This function assumes that all variable names are included in
        self._symbols.

        Args:
            ordered_ra_pairs: A list of (pyeda rule, action/option) pairs

        Returns:
            ordered_alg_act_pairs: ordered_ra_pairs where each rule
                                   is replaced with a sympy expression
        """

        ordered_alg_act_pairs = []

        for pair in ordered_ra_pairs:
            assert pair[0].is_dnf()
            ast = pair[0].to_ast()
            lit_map, _, _ = pair[0].encode_dnf()

            sympy_str = self._logic_to_algebra(ast, lit_map)
            sympy_expr = sympify(sympy_str, locals=self._symbols)

            ordered_alg_act_pairs.append((sympy_expr, pair[1]))

        return ordered_alg_act_pairs

    def convert_policy(self, policy, out_file):
        """Convert a Policy into a BehaviourTree and write the BT to file.

        Args:
            policy: A deterministic, memoryless policy
            out_file: The output file for the BT
        """

        # Step 1: Reset all internal data structures
        self._reset()

        # Step 2: Group states by their policy action
        act_to_rule, act_to_var_map = self._extract_rules_from_policy(policy)

        # Step 3: Build all internal bookkeeping structures
        self._build_internal_mappings(act_to_var_map)

        # Step 4: Minimise the rules for each action/option using Espresso
        act_to_min_rule = self._minimise_with_espresso(act_to_rule)

        # Step 5: Convert minimised rules into Horn clauses (sort of)
        # See self._convert_to_horn_clauses() for more information
        act_to_horn = self._convert_to_horn_clauses(act_to_min_rule)

        # Step 6: Score the rules and sort them in descending order
        ordered_ra_pairs = self._score_and_sort_rules(act_to_horn)

        # Step 7: Convert logical operators & and | into * and + for factorisation
        ordered_alg_act_pairs = self._pyeda_rules_to_sympy_algebraic(ordered_ra_pairs)

        # Step 8: Run GFactor to factorise each rule
        # TODO: Fill in

        # Step 9: Convert the rules into a BT
        bt = None  # TODO: Replace with function once written

        # Step 10: Write the BT to file as a BT.cpp XML file
        bt.to_BT_XML(out_file)
