#!/usr/bin/env python3
"""A script to run REFINE-PLAN for the object pushing domain.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.condition import (
    Label,
    EqCondition,
    AndCondition,
    OrCondition,
    NeqCondition,
)
from refine_plan.learning.option_learning import mongodb_to_yaml, learn_dbns
from refine_plan.models.state_factor import StateFactor, IntStateFactor
from refine_plan.algorithms.semi_mdp_solver import synthesise_policy
from refine_plan.models.dbn_option import DBNOption
from refine_plan.models.semi_mdp import SemiMDP
from pymongo import MongoClient
from datetime import datetime
from pathlib import Path
import yaml
import os

MONGO_STR = "mongodb://localhost:27017/"
SUCCESS = 1
FAIL = -1

# Hardcoded reward values
PUSH_REWARD = 1
PUSH_FAILURE_COST = 100
BLOCK_COST = 10


def get_sf_values():
    """Get the state factor values for planning.

    Reads from data, so only run this once.
    """
    objects = set()
    clutter_levels = set()
    data_dir = Path("../data/object_pushing/sim_data")
    for path in os.listdir(data_dir):
        with open(data_dir / path, "r") as yaml_in:
            data = yaml.safe_load(yaml_in)
            for item in data:
                objects.add(item[0])
                clutter_levels.add(item[1])

    print(f"Objects: {objects}")
    print(f"Clutter levels: {clutter_levels}")


def build_sfs():
    """Build the state factors for planning.

    Returns:
        The state factors for planning
    """
    # NOTE: A bit of a hack that pushed and blocked are in the SF here, but it just makes things
    # A lot easier!
    obj_vals = ["stool", "lamp", "chair", "bottle", "table", "pushed", "blocked"]
    return [StateFactor("object", obj_vals), IntStateFactor("clutter_level", 0, 5)]


def _get_successor_state(action, success):
    """Get the successor object state factor value.

    Args:
        action: push or block
        success: The success or failure flag

    Returns:
        pushed or blocked
    """
    if action == "push":
        if success == SUCCESS:
            return "pushed"
        else:
            return "blocked"
    else:
        if success == SUCCESS:
            return "blocked"
        else:
            raise Exception("Failed block action - unexpected")


def _get_reward(clutter_level, action, success):
    """Get the action reward given clutter level and success state.

    If pushed and successful: clutter_level * push_success_reward
    If blocked: -1 * clutter_level * block_cost
    If pushed and failed: <VALUE FOR BLOCKED> - failure_penalty

    Args:
        clutter_level: How cluttered is the area around the robot?
        action: push or block
        success: SUCCESS or FAIL

    Returns:
        The reward
    """
    # NOTE: Use clutter_level + 1 to ensure 0 still gets something!
    clutter_multiplier = clutter_level + 1
    if action == "push":
        if success == SUCCESS:
            return clutter_multiplier * PUSH_REWARD
        else:
            return -1 * (clutter_multiplier * BLOCK_COST + PUSH_FAILURE_COST)
    else:
        return -1 * clutter_multiplier * BLOCK_COST


def write_bosch_data_to_db():
    """Write the Bosch sim data to a DB so we can use it for planning."""
    client = MongoClient(MONGO_STR)["refine-plan"]["object-pushing-data"]
    docs = []
    data_dir = Path("../data/object_pushing/sim_data")
    for path in os.listdir(data_dir):
        with open(data_dir / path, "r") as yaml_in:
            data = yaml.safe_load(yaml_in)
            for item in data:
                if item[2] == "block":
                    if item[3] == FAIL:
                        print("BLOCK FAIL")
                        continue
                doc = {}
                doc["run_id"] = int(path[3:5])
                doc["object0"] = item[0]
                doc["objectt"] = _get_successor_state(item[2], item[3])
                doc["clutter_level0"] = item[1]
                doc["clutter_levelt"] = item[1]
                doc["option"] = item[2]
                doc["duration"] = _get_reward(item[1], item[2], item[3])
                doc["_meta"] = {"inserted_at": datetime.now()}
                docs.append(doc)
    client.insert_many(docs)


def _get_enabled_cond(object_sf):
    """Get the enabled condition for block and push actions.

    Both actions have the same enabled condition.

    Args:
        object_sf: The state factor representing object states
        option: The option we want the condition for

    Returns:
        The enabled condition for the option
    """

    return AndCondition(
        NeqCondition(object_sf, "blocked"), NeqCondition(object_sf, "pushed")
    )


def write_mongodb_to_yaml():
    """Get the dataset from the database."""

    print("Writing Mongo Database to Yaml File")
    mongodb_to_yaml(
        MONGO_STR,
        "refine-plan",
        "object-pushing-data",
        build_sfs(),
        "../data/object_pushing/dataset.yaml",
    )
    print("YAML Dataset Created")


def learn_options():
    """Learn the options from the YAML file."""
    dataset_path = "../data/object_pushing/dataset.yaml"
    output_dir = "../data/object_pushing/"

    learn_dbns(dataset_path, output_dir, build_sfs())


def run_planner():
    """Run REFINE-PLAN and synthesise a policy.

    Returns:
        The refined BT
    """

    sf_list = build_sfs()

    goal_cond = OrCondition(
        EqCondition(sf_list[0], "pushed"), EqCondition(sf_list[0], "blocked")
    )
    labels = [Label("goal", goal_cond)]

    option_names = set(["push", "block"])

    option_list = []
    for option in option_names:
        print("Reading in option: {}".format(option))
        t_path = "../data/object_pushing/{}_transition.bifxml".format(option)
        r_path = "../data/object_pushing/{}_reward.bifxml".format(option)
        option_list.append(
            DBNOption(option, t_path, r_path, sf_list, _get_enabled_cond(sf_list[0]))
        )

    print("Creating MDP...")
    semi_mdp = SemiMDP(sf_list, option_list, labels)
    print("Synthesising Policy...")
    policy = synthesise_policy(semi_mdp, prism_prop='Rmax=?[F "goal"]')
    policy.write_policy("../data/object_pushing/pushing_policy.yaml")


if __name__ == "__main__":
    # get_sf_values()
    write_bosch_data_to_db()
    write_mongodb_to_yaml()
    learn_options()
    run_planner()
