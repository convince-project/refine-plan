#!/usr/bin/env python
"""Functions for processing datsets and learning DBNs.

Author: Charlie Street
Owner: Charlie Street
"""

from pymongo import MongoClient
import pyAgrum as gum
import pandas as pd
import numpy as np
import yaml
import os


def _initialise_dict_for_option(dataset_dict, option, sf_list):
    """Initialise a yaml dataset dict for a new option.

    Args:
        dataset_dict: The dictionary with the dataset
        option: The option we're initialising for
        sf_list: The list state factors we expect to see in the Mongo DB
    """
    dataset_dict[option] = {"transition": {}, "reward": {}}
    for sf in sf_list:
        sf_name = sf.get_name()
        sf_0_name = "{}0".format(sf_name)
        sf_t_name = "{}t".format(sf_name)
        dataset_dict[option]["transition"][sf_0_name] = []
        dataset_dict[option]["transition"][sf_t_name] = []
        dataset_dict[option]["reward"][sf_name] = []

    dataset_dict[option]["reward"]["r"] = []


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


def mongodb_to_dict(
    connection_str,
    db_name,
    collection_name,
    sf_list,
    query={},
    sort_by=None,
    limit=None,
    split_by_motion=False,
):
    """Process a mongodb collection into a dictionary for learning.

    Args:
        connection_str: The mongodb connection string
        db_name: The Mongo database name
        collection_name: The collection within the database
        sf_list: The list of state factors to expect in the MongoDB
        out_file: The path for the yaml file
        query: A query to filter the documents that get collected
        sort_by: A field to sort the documents by
        limit: A limit on the number of documents returned
        split_by_motion: Should the option datasets be split by motion parameter?

    Returns:
        dataset_dict: The dataset from mongodb into a pyAgrum format dictionary
    """
    client = MongoClient(connection_str)
    collection = client[db_name][collection_name]

    dataset_dict = {}

    # Search through all documents
    docs = collection.find(query)
    if sort_by is not None:
        docs = docs.sort(sort_by)

    if limit is not None:
        docs = docs.limit(limit)

    for doc in docs:
        if _is_zero_cost_loop(doc, sf_list):
            continue
        option = doc["option"]
        if split_by_motion:
            option += ".{}".format(doc["motion"])
        if option not in dataset_dict:
            _initialise_dict_for_option(dataset_dict, option, sf_list)

        for sf in sf_list:
            sf_name = sf.get_name()
            sf_0_name = "{}0".format(sf_name)
            sf_t_name = "{}t".format(sf_name)
            dataset_dict[option]["transition"][sf_0_name].append(doc[sf_0_name])
            dataset_dict[option]["transition"][sf_t_name].append(doc[sf_t_name])
            dataset_dict[option]["reward"][sf_name].append(doc[sf_0_name])

        # Round durations to integers for the sake of feeding into the DBN
        dataset_dict[option]["reward"]["r"].append(round(doc["duration"]))

    return dataset_dict


def mongodb_to_yaml(
    connection_str,
    db_name,
    collection_name,
    sf_list,
    out_file,
    query={},
    sort_by=None,
    limit=None,
    split_by_motion=False,
):
    """Processes a mongodb collection into a yaml dataset for DBN learning.

    Args:
        connection_str: The mongodb connection string
        db_name: The Mongo database name
        collection_name: The collection within the database
        sf_list: The list of state factors to expect in the MongoDB
        out_file: The path for the yaml file
        query: A query to filter the documents that get collected
        sort_by: A field to sort the documents by
        limit: A limit on the number of documents returned
        split_by_motion: Should the option datasets be split by motion parameter?
    """

    yaml_dict = mongodb_to_dict(
        connection_str,
        db_name,
        collection_name,
        sf_list,
        query=query,
        sort_by=sort_by,
        limit=limit,
        split_by_motion=split_by_motion,
    )

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


def _remove_unchanging_vars(dataset, sf_list):
    """Removes {}t variables from transition datasets if they are unchanged.

    These variables don't need to be in the DBN as they never change under an option.

    If included, they have a tendency to mess up structure learning.

    Args:
        dataset: A dictionary from option to transition and reward information
        sf_list: The list of state factors which appear in the dataset

    Return:
        The dataset with the unchanging variables removed
    """

    filtered_dataset = {}

    for option in dataset:

        filtered_dataset[option] = {
            "transition": {},
            "reward": dataset[option]["reward"],
        }
        for sf in sf_list:
            name_0 = "{}0".format(sf.get_name())
            name_t = "{}t".format(sf.get_name())
            vals_at_0 = dataset[option]["transition"][name_0]
            vals_at_t = dataset[option]["transition"][name_t]
            filtered_dataset[option]["transition"][name_0] = vals_at_0
            if vals_at_0 != vals_at_t:
                filtered_dataset[option]["transition"][name_t] = vals_at_t

    return filtered_dataset


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

    # I have found empirically that hill climbing works nicely
    trans_learner.useGreedyHillClimbing()
    reward_learner.useGreedyHillClimbing()

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
            # Some {}t variables might have been removed by this point
            if sf_name_t in option_dataset["transition"]:
                trans_learner.addForbiddenArc(sf_name_t, sf_2_name)

    return trans_learner, reward_learner


def _remove_edgeless_vars(bn, sf_list, is_transition_dbn):
    """Remove state factor variables from Bayesian networks with no in/out edges.

    These variables are useless to the network and can be removed.

    If bn is a transition DBN, we look at all {}0 variables.
    If bn is a reward DBN, we look at all variables except r.

    Args:
        bn: The Bayesian network
        sf_list: The list of state factors captured in the network
        is_transition_dbn: If True, bn is a transition DBN, else it is a reward DBN
    """
    for sf in sf_list:
        if is_transition_dbn:
            var_name = "{}0".format(sf.get_name())
        else:
            var_name = sf.get_name()

        # If no parents and no descendants, the node is isolated and can be removed
        if len(bn.parents(var_name)) == 0 and len(bn.descendants(var_name)) == 0:
            bn.erase(var_name)


def learn_bns_for_one_option(option_dataset, sf_list):
    """Learn a transition DBN and a reward BN for a single option.

    This function is slightly different to learn_dbns.
    learn_dbns is the general recommended function, but this function allows
    for one option model to be learned without reading or writing anything.
    It's intended use is for updating models online for exploration.

    The dataset should be a dictionary with two keys: 'transition' and 'reward'.
    In 'transition' there should be a dictionary with keys sf0 and sft for each
    state factor sf. At each of these keys is a list of data. In 'reward' there
    should be a dictionary with keys sf for each state factor sf, and
    'r' to represent the reward. At each of these keys is a list of data.

    Args:
        option_dataset: A dictionary containing the dataset for this option
        sf_list: The list of state factors we expect to see in the dataset

    Returns:
        The DBN for the transition function and the BN for the reward function
    """

    wrapper = {"option": option_dataset}
    _check_dataset(wrapper, sf_list)
    wrapper = _remove_unchanging_vars(wrapper, sf_list)
    wrapper = _dataset_vals_to_str(wrapper)

    trans_learner, reward_learner = _setup_learners(wrapper["option"], sf_list)
    trans_bn = trans_learner.learnBN()
    _remove_edgeless_vars(trans_bn, sf_list, True)
    reward_bn = reward_learner.learnBN()
    _remove_edgeless_vars(reward_bn, sf_list, False)

    return trans_bn, reward_bn


def learn_dbns(dataset_path, output_dir, sf_list):
    """Learn a set of DBNs representing options.

    The dataset should be a dictionary from options to a dictionary with two keys:
    'transition' and 'reward'. In 'transition' there should be a dictionary with keys
    sf0 and sft for each state factor sf. At each of these keys is a list of data.
    In 'reward' there should be a dictionary with keys sf for each state factor sf, and
    'r' to represent the reward. At each of these keys is a list of data.

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

    # Remove any unchanging variables
    dataset = _remove_unchanging_vars(dataset, sf_list)

    dataset = _dataset_vals_to_str(dataset)

    for option in dataset:
        trans_learner, reward_learner = _setup_learners(dataset[option], sf_list)

        print("LEARNING TRANSITION DBN FOR OPTION: {}".format(option))
        trans_bn = trans_learner.learnBN()
        _remove_edgeless_vars(trans_bn, sf_list, True)
        trans_out = os.path.join(output_dir, "{}_transition.bifxml".format(option))
        trans_bn.saveBIFXML(trans_out)

        print("LEARNING REWARD DBN FOR OPTION: {}".format(option))
        reward_bn = reward_learner.learnBN()
        _remove_edgeless_vars(reward_bn, sf_list, False)
        reward_out = os.path.join(output_dir, "{}_reward.bifxml".format(option))
        reward_bn.saveBIFXML(reward_out)
