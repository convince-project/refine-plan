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

from pyeda.boolalg.expr import Complement, Variable, AndOp, OrOp
from sympy import Symbol, sympify, Mul, Add, simplify
from refine_plan.algorithms.gfactor import gfactor
from refine_plan.models.behaviour_tree import (
    BehaviourTree,
    SequenceNode,
    FallbackNode,
    ActionNode,
    ConditionNode,
)
from refine_plan.models.condition import (
    OrCondition,
    EqCondition,
    NeqCondition,
    GtCondition,
    LeqCondition,
    GeqCondition,
    LtCondition,
)
from refine_plan.models.policy import Policy
from pyeda.inter import espresso_exprs, Not


class PolicyBTConverter(object):
    """This class contains the functionality to convert a policy into a BT.

    Attributes:
        _vars_to_conds: A dictionary from variable names to Condition objects.
        _vars_to_symbols: A dictionary form variable names to sympy symbols.
    """

    def __init__(self):
        """Reset all attributes."""
        self._reset()

    def _reset(self):
        """Reset all attributes to None."""
        self._vars_to_conds = None
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

    def _logic_to_algebra(self, logical_rule):
        """Converts a pyeda logical expr as an ast to a sympy algebraic expression.

        The function assumes logical_rule is in DNF, in particularly that only
        variables are negated.

        Args:
            logical_rule: A logical rule as a pyeda expression

        Returns:
            sympy_str: An algebraic expression (string) which can be sympified
        """

        if isinstance(logical_rule, OrOp) or isinstance(logical_rule, AndOp):
            join_str = " + " if isinstance(logical_rule, OrOp) else "*"
            components = [self._logic_to_algebra(sub) for sub in logical_rule.xs]
            return join_str.join(components)
        elif isinstance(logical_rule, Complement):
            # The original authors create auxiliary variables for factorisation
            # when a variable is negated.
            # I imagine this is because it's hard to translate the negation
            # cleanly into the algebraic form?
            assert len(logical_rule.inputs) == 1
            var = str(logical_rule.inputs[0])
            assert var[:3] != "NOT"  # Checking for bad variable names
            not_var = "NOT{}".format(var)
            if not_var not in self._vars_to_symbols:
                self._vars_to_symbols[not_var] = Symbol(not_var)
            return not_var
        elif isinstance(logical_rule, Variable):
            assert str(logical_rule)[:3] != "NOT"  # Checking for bad variable names
            return str(logical_rule)

    def _pyeda_rules_to_sympy_algebraic(self, ordered_ra_pairs):
        """Converts a set of pyeda rules into sympy algebraic expressions.

        This function assumes that all variable names are included in
        self._vars_to_symbols.

        Args:
            ordered_ra_pairs: A list of (pyeda rule, action/option) pairs

        Returns:
            ordered_alg_act_pairs: ordered_ra_pairs where each rule
                                   is replaced with a sympy expression
        """

        ordered_alg_act_pairs = []

        for pair in ordered_ra_pairs:
            assert pair[0].is_dnf()
            sympy_str = self._logic_to_algebra(pair[0])
            sympy_expr = sympify(sympy_str, locals=self._vars_to_symbols)

            ordered_alg_act_pairs.append((sympy_expr, pair[1]))

        return ordered_alg_act_pairs

    def _reduce_sympy_expressions(self, ordered_alg_act_pairs):
        """Minimises sympy expressions representing rules using GFactor.

        Args:
            ordered_alg_act_pairs: List of (sympy expression, action/option) pairs

        Returns:
            min_alg_act_pairs: List of (minimised sympy expr, action/option) pairs
        """
        min_alg_act_pairs = []

        for pair in ordered_alg_act_pairs:
            reduced_expr = gfactor(pair[0])
            min_alg_act_pairs.append((reduced_expr, pair[1]))

        return min_alg_act_pairs

    def _build_condition_node(self, var_name):
        """Build a condition node for a given variable.

        If the variable name is prefixed with NOT, we lookup the suffix, and then
        negate the condition. Note that the only condition types which will be negated
        are EqCondition, GtCondition, and GeqCondition, due to definitions in condition.py.

        Args:
            var_name: The variable name to build the condition node for

        Returns:
            condition_node: The corresponding condition node
        """
        if var_name[:3] == "NOT":  # Undoing the slightly hacky negation symbols
            cond = self._vars_to_conds[var_name[3:]]
            if isinstance(cond, EqCondition):
                return ConditionNode(var_name, NeqCondition(cond._sf, cond._value))
            elif isinstance(cond, GeqCondition):
                return ConditionNode(var_name, LtCondition(cond._sf, cond._value))
            elif isinstance(cond, GtCondition):
                return ConditionNode(var_name, LeqCondition(cond._sf, cond._value))
        else:
            return ConditionNode(var_name, self._vars_to_conds[var_name])

    def _sub_bt_for_rule(self, rule):
        """Convert a single rule into a sub-behaviour tree.

        Args:
            rule: A logical rule represented as a sympy algebraic expression

        Returns:
            bt: The sub behaviour tree

        Raises:
            bad_operator_exception: Raised if non +/*/symbol expression found
        """

        # Add = OR; Mul = AND
        if isinstance(rule, Add) or isinstance(rule, Mul):
            bt = FallbackNode() if isinstance(rule, Add) else SequenceNode()
            for child in rule.args:
                bt.add_child(self._sub_bt_for_rule(child))
            return bt
        elif isinstance(rule, Symbol):  # A variable representing a condition
            return self._build_condition_node(str(rule))
        else:
            raise Exception("Invalid operator found in rule.")

    def _convert_rules_to_bt(self, min_alg_act_pairs):
        """Converts the ordered rules into the final behaviour tree (BT).

        Note that here the sympy expressions are still in algebraic form, i.e.
        with + and * instead of | and &, respectively.

        Args:
            min_alg_act_pairs: List of (sympy expr, action/option pairs)

        Returns:
            bt: The final behaviour tree
        """

        # Root node is always a sequence
        root_node = SequenceNode()

        for pair in min_alg_act_pairs:

            sub_bt = self._compute_sub_bt_for_rule(simplify(pair[0]))
            # The sub_bt for a rule should always have an OR at the outer level
            # This corresponds to a fallback node
            assert isinstance(sub_bt, FallbackNode)

            sub_bt.add_child(ActionNode(pair[1]))  # Add the action at the end
            root_node.add_child(sub_bt)

        return BehaviourTree(root_node)

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
        min_alg_act_pairs = self._reduce_sympy_expressions(ordered_alg_act_pairs)

        # Step 9: Convert the rules into a BT
        bt = self._convert_rules_to_bt(min_alg_act_pairs)

        # Step 10: Write the BT to file as a BT.cpp XML file
        bt.to_BT_XML(out_file)
