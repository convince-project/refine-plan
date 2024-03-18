#!/usr/bin/env python3
""" Functions which solve a semi-MDP into storm and retrieve the policy.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.policy import Policy
from refine_plan.models.state import State
import tempfile
import stormpy


def _build_prism_program(semi_mdp):
    """Build PRISM program from semi-MDP.

    Storm has no way of reading PRISM files directly from string, so
    it has to be written to a file first.

    Args:
        semi_mdp: The SemiMDP.

    Returns:
        prism_program: A stormpy PRISMProgram object
    """

    tmp_file = tempfile.NamedTemporaryFile()

    with open(tmp_file, "w") as prism_in:
        prism_in.write(semi_mdp.to_prism_string())

    return stormpy.parse_prism_program(tmp_file)


def _build_storm_model(prism_program, formula):
    """Builds a storm model of the semi-MDP.

    Args:
        prism_program: The PRISM program
        formula: The formula to model check

    Returns:
        storm_model: The storm model

    Raises:
        no_choice_labelling: Raised if the model has no choice labels
        no_state_valuations: Raised if the model doesn't include state values
    """
    options = stormpy.BuilderOptions([formula.raw_formula])
    options.set_build_all_labels()
    options.set_build_choice_labels()
    options.set_build_state_valuations()
    storm_model = stormpy.build_sparse_model_with_options(prism_program, options)

    if not storm_model.has_choice_labeling():
        raise Exception("Storm model has no choice labelling")

    if not storm_model.has_state_valuations():
        raise Exception("Storm model has no state valuations")

    return storm_model


def _check_result(result):
    """Runs a few sanity checks on the model checking result.

    Args:
        result: The model checking result

    Raises:
        bad_result_exception: Raised if something is wrong with the result
    """

    all_states = result.result_for_all_states
    has_scheduler = result.has_scheduler
    memoryless = has_scheduler and result.scheduler.memoryless
    deterministic = has_scheduler and result.scheduler.deterministic

    if (
        (not all_states)
        or (not has_scheduler)
        or (not memoryless)
        or (not deterministic)
    ):
        raise Exception(
            "Error with model checking:\nResult for all states: "
            + str(all_states)
            + "\nScheduler extracted: "
            + str(has_scheduler)
            + "\nScheduler is memoryless: "
            + str(memoryless)
            + "\nScheduler is deterministic: "
            + str(deterministic)
        )


def _extract_policy(result, storm_model, semi_mdp):
    """Extract the policy and value function from the storm result.

    Args:
        result: The storm model checking result
        storm_model: The storm model object
        semi_mdp: The semi-MDP (needed for state factors)

    Returns:
        policy: A Policy object
    """
    state_action_dict = {}
    value_dict = {}

    choice_labeling = storm_model.choice_labeling  # For action names
    state_vals = storm_model.state_valuations  # For state names
    scheduler = result.scheduler
    state_factors = semi_mdp.get_state_factors()

    for state_id in storm_model.states:

        state_json = state_vals.get_json(state_id)  # sf_name -> value of sf
        state = State({state_factors[sf]: state_json[sf] for sf in state_factors})

        # Need to offset action by the row group for the state according to Github
        action_id = (
            storm_model.transition_matrix.get_row_group_start(state_id)
            + scheduler.get_choice(state).get_deterministic_choice()
        )
        # Action labels for action (should just be one)
        choice_set = choice_labeling.get_labels_of_choice(action_id)
        action_label = None if len(choice_set) > 0 else list(choice_set)[0]

        state_action_dict[state] = action_label
        value_dict[state] = result.at(state_id)

    return Policy(state_action_dict, value_dict=value_dict)


def synthesise_policy(semi_mdp, prism_prop='Rmax=?[F "goal"]'):
    """Solve a semi-MDP and return the policy.

    Note that stormpy can only handle reachability properties.

    Args:
        semi_mdp: The semiMDP
        prism_prop: The property to solve for (must be reachability-based)

    Returns:
        policy: A policy with the value function included

    Raises:
        solution_exception: Raised if issue arises with model checking result
    """

    prism_program = _build_prism_program(semi_mdp)
    formula = stormpy.parse_properties(prism_prop, prism_program)[0]

    storm_model = _build_storm_model(prism_program, formula)

    result = stormpy.model_checking(storm_model, formula, extract_scheduler=True)

    _check_result(result)

    return _extract_policy(result, semi_mdp)
