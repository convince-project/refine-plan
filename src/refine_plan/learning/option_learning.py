#!/usr/bin/env python
""" Functions for processing datsets and learning DBNs.

Author: Charlie Street
Owner: Charlie Street
"""

from pymongo import MongoClient
import pyAgrum as gum
import pandas as pd
import numpy as np
import yaml
import os


def _initialise_dict_for_option(yaml_dict, option, sf_list):
    """Initialise a yaml dataset dict for a new option.

    Args:
        yaml_dict: The dictionary with the dataset to be written to yaml
        option: The option we're initialising for
        sf_list: The list state factors we expect to see in the Mongo DB
    """
    yaml_dict[option] = {"transition": {}, "reward": {}}
    for sf in sf_list:
        sf_name = sf.get_name()
        sf_0_name = "{}0".format(sf_name)
        sf_t_name = "{}t".format(sf_name)
        yaml_dict[option]["transition"][sf_0_name] = []
        yaml_dict[option]["transition"][sf_t_name] = []
        yaml_dict[option]["reward"][sf_name] = []

    yaml_dict[option]["reward"]["r"] = []


def _is_zero_cost_loop(doc, sf_list):
    """Test whether a MongoDB document captures a zero-cost self loop.

    These represent unenabled actions and should be filtered out.
    If left in, DBN learning doesn't perform well.

    Args:
        doc: The MongoDB document
        sf_list: The list of state factors to expect in the MongoDB

    Returns:
        Whether the doc captures a zero-cost self loop
    """

    # If duration not 0, not zero cost self loop
    if not np.isclose(doc["duration"], 0.0):
        return False

    # If a state factor has changed, not a self loop
    for sf in sf_list:
        if doc["{}0".format(sf.get_name())] != doc["{}t".format(sf.get_name())]:
            return False

    return True


def mongodb_to_yaml(connection_str, db_name, collection_name, sf_list, out_file):
    """Processes a mongodb collection into a yaml dataset for DBN learning.

    Args:
        connection_str: The mongodb connection string
        db_name: The Mongo database name
        collection_name: The collection within the database
        sf_list: The list of state factors to expect in the MongoDB
        out_file: The path for the yaml file
    """

    client = MongoClient(connection_str)

    yaml_dict = {}

    collection = client[db_name][collection_name]

    # Search through all documents
    for doc in collection.find({}):
        if _is_zero_cost_loop(doc, sf_list):
            continue
        option = doc["option"]
        if option not in yaml_dict:
            _initialise_dict_for_option(yaml_dict, option, sf_list)

        for sf in sf_list:
            sf_name = sf.get_name()
            sf_0_name = "{}0".format(sf_name)
            sf_t_name = "{}t".format(sf_name)
            yaml_dict[option]["transition"][sf_0_name].append(doc[sf_0_name])
            yaml_dict[option]["transition"][sf_t_name].append(doc[sf_t_name])
            yaml_dict[option]["reward"][sf_name].append(doc[sf_0_name])

        # Round durations to integers for the sake of feeding into the DBN
        yaml_dict[option]["reward"]["r"].append(round(doc["duration"]))

    # Write dataset to yaml
    with open(out_file, "w") as yaml_out:
        yaml.dump(yaml_dict, yaml_out)


def _check_dataset(dataset, sf_list):
    """Test that the dataset follows the expected format.

    Args:
        dataset: The dataset dictionary
        sf_list: The list of state factors we expect to see in the dataet

    Raises:
        bad_dataset: Raised if the dataset is invalid
    """

    assert isinstance(dataset, dict)

    for option in dataset:
        assert isinstance(dataset[option], dict)
        assert len(dataset[option]) == 2
        assert "transition" in dataset[option]
        assert "reward" in dataset[option]

        assert isinstance(dataset[option]["transition"], dict)
        assert isinstance(dataset[option]["reward"], dict)

        # Check state factor names
        num_entries = None
        for sf in sf_list:
            sf_name = sf.get_name()
            sf_0_name = "{}0".format(sf_name)
            sf_t_name = "{}t".format(sf_name)

            assert sf_0_name in dataset[option]["transition"]
            assert isinstance(dataset[option]["transition"][sf_0_name], list)
            assert sf_t_name in dataset[option]["transition"]
            assert isinstance(dataset[option]["transition"][sf_t_name], list)
            assert sf_name in dataset[option]["reward"]
            assert isinstance(dataset[option]["reward"][sf_name], list)

            if num_entries is None:
                num_entries = len(dataset[option]["transition"][sf_0_name])

            # Check for consistent number of data entries
            assert len(dataset[option]["transition"][sf_0_name]) == num_entries
            assert len(dataset[option]["transition"][sf_t_name]) == num_entries
            assert len(dataset[option]["reward"][sf_name]) == num_entries

        # Check reward values
        assert "r" in dataset[option]["reward"]
        assert isinstance(dataset[option]["reward"]["r"], list)
        assert len(dataset[option]["reward"]["r"]) == num_entries


def _dataset_vals_to_str(dataset):
    """Converts all data items in the dataset to strings.

    This is to ensure that pyagrum will work the way we want it to.

    Args:
        dataset: The dataset dictionary.

    Returns:
        The modified dataset dictionary
    """

    str_dataset = {}

    for option in dataset:
        str_dataset[option] = {"transition": {}, "reward": {}}

        for key in dataset[option]["transition"]:
            str_dataset[option]["transition"][key] = list(
                map(lambda x: str(x), dataset[option]["transition"][key])
            )

        for key in dataset[option]["reward"]:
            str_dataset[option]["reward"][key] = list(
                map(lambda x: str(x), dataset[option]["reward"][key])
            )

    return str_dataset


def _setup_learners(option_dataset, sf_list):
    """Setup the BNLearner objects for the transition and reward function for an option.

    Args:
        option_dataset: The dataset for the option
        sf_list: The list of state factors we expect to see in the dataset

    Returns:
        The transition function learner and the reward function learner
    """

    trans_learner = gum.BNLearner(pd.DataFrame(data=option_dataset["transition"]))
    reward_learner = gum.BNLearner(pd.DataFrame(data=option_dataset["reward"]))

    # Restrict the edges allowed in the BN
    # 1: Transition DBN - 0 vars can't go to other 0 vars
    # 2: Reward DBN - sf vars can't go between each other
    # 3: Transition DBN - t vars can't go back to 0 vars
    # 4: Reward DBN - r can't go back to other vars
    for sf in sf_list:
        sf_name = "{}0".format(sf.get_name())
        sf_name_t = "{}t".format(sf.get_name())
        reward_learner.addForbiddenArc("r", sf.get_name())
        for sf_2 in sf_list:
            sf_2_name = "{}0".format(sf_2.get_name())
            trans_learner.addForbiddenArc(sf_name, sf_2_name)
            reward_learner.addForbiddenArc(sf.get_name(), sf_2.get_name())
            trans_learner.addForbiddenArc(sf_name_t, sf_2_name)

    return trans_learner, reward_learner


def learn_dbns(dataset_path, output_dir, sf_list):
    """Learn a set of DBNs representing options.

    The dataset should be a dictionary from options to a dictionary with two keys:
    'transition' and 'reward'. In 'transition' there should be a dictionary with keys
    sf0 and sft for each state factor sf. At each of these keys is a list of data.
    In 'reward' there should be a dictionary with keys sf for each state factor sf, and
    'r' to represent the reward. At each of these keys is a list of data

    Args:
        dataset_path: A yaml file containing the dataset.
        output_dir: The output directory for the DBNs.
        sf_list: The list of state factors we expect to see in the dataset
    """

    print("READING IN DATASET")
    with open(dataset_path, "r") as yaml_in:
        dataset = yaml.load(yaml_in, yaml.FullLoader)

    # Test the dataset has the correct format
    _check_dataset(dataset, sf_list)
    dataset = _dataset_vals_to_str(dataset)

    for option in dataset:
        trans_learner, reward_learner = _setup_learners(dataset[option], sf_list)

        print("LEARNING TRANSITION DBN FOR OPTION: {}".format(option))
        trans_bn = trans_learner.learnBN()
        trans_out = os.path.join(output_dir, "{}_transition.bifxml".format(option))
        trans_bn.saveBIFXML(trans_out)

        print("LEARNING REWARD DBN FOR OPTION: {}".format(option))
        reward_bn = reward_learner.learnBN()
        reward_out = os.path.join(output_dir, "{}_reward.bifxml".format(option))
        reward_bn.saveBIFXML(reward_out)
