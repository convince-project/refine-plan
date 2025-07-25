#!/usr/bin/env python3
"""Unit tests for option learning functions.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import StateFactor, BoolStateFactor
from refine_plan.learning.option_learning import (
    _initialise_dict_for_option,
    mongodb_to_dict,
    mongodb_to_yaml,
    _check_dataset,
    _dataset_vals_to_str,
    _setup_learners,
    learn_bns_for_one_option,
    learn_dbns,
    _is_zero_cost_loop,
    _remove_unchanging_vars,
    _remove_edgeless_vars,
)
from pymongo import MongoClient
import pyAgrum as gum
import unittest
import yaml
import os

try:
    client = MongoClient("localhost:27017", timeoutMS=1000)
    client.server_info()
    MONGO_RUNNING = True
except:
    MONGO_RUNNING = False


class InitialiseDictForOptionTest(unittest.TestCase):

    def test_function(self):
        dataset = {}

        sf_list = [StateFactor("x", ["a", "b", "c"]), StateFactor("y", ["d", "e", "f"])]

        _initialise_dict_for_option(dataset, "test", sf_list)

        expected = {
            "test": {
                "transition": {"x0": [], "xt": [], "y0": [], "yt": []},
                "reward": {"x": [], "y": [], "r": []},
            }
        }

        self.assertEqual(dataset, expected)


class IsZeroCostLoopTest(unittest.TestCase):

    def test_function(self):
        doc_1 = {
            "option": "test",
            "x0": 1,
            "xt": 1,
            "y0": False,
            "yt": False,
            "duration": 0,
        }
        doc_2 = {
            "option": "test",
            "x0": 1,
            "xt": 1,
            "y0": False,
            "yt": False,
            "duration": 5,
        }
        doc_3 = {
            "option": "test",
            "x0": 3,
            "xt": 1,
            "y0": False,
            "yt": True,
            "duration": 0,
        }
        doc_4 = {
            "option": "test",
            "x0": 1,
            "xt": 2,
            "y0": False,
            "yt": False,
            "duration": 5,
        }

        sf_list = [StateFactor("x", [1, 2, 3]), BoolStateFactor("y")]

        self.assertTrue(_is_zero_cost_loop(doc_1, sf_list))
        self.assertFalse(_is_zero_cost_loop(doc_2, sf_list))
        self.assertFalse(_is_zero_cost_loop(doc_3, sf_list))
        self.assertFalse(_is_zero_cost_loop(doc_4, sf_list))


@unittest.skipIf(not MONGO_RUNNING, "MongoDB server not running.")
class MongodbToDictTest(unittest.TestCase):

    def test_function(self):
        # This test requires a local mongo server to be setup on port 27017
        connection_str = "mongodb://localhost:27017"
        db_name = "test_db"
        collection_name = "test_collection"

        client = MongoClient(connection_str)
        collection = client[db_name][collection_name]
        doc_1 = {
            "option": "test",
            "x0": 1,
            "xt": 2,
            "y0": False,
            "yt": False,
            "duration": 5,
        }
        doc_2 = {
            "option": "test",
            "x0": 2,
            "xt": 3,
            "y0": True,
            "yt": False,
            "duration": 5,
        }
        doc_3 = {
            "option": "test",
            "x0": 3,
            "xt": 1,
            "y0": False,
            "yt": True,
            "duration": 7,
        }
        doc_4 = {  # Zero cost self loop, should be removed
            "option": "test",
            "x0": 1,
            "xt": 1,
            "y0": False,
            "yt": False,
            "duration": 0,
        }
        collection.insert_many([doc_1, doc_2, doc_3, doc_4])

        sf_list = [StateFactor("x", [1, 2, 3]), BoolStateFactor("y")]
        dataset = mongodb_to_dict(connection_str, db_name, collection_name, sf_list)
        client.drop_database("test_db")

        expected = {
            "test": {
                "transition": {
                    "x0": [1, 2, 3],
                    "xt": [2, 3, 1],
                    "y0": [False, True, False],
                    "yt": [False, False, True],
                },
                "reward": {"x": [1, 2, 3], "y": [False, True, False], "r": [5, 5, 7]},
            }
        }

        self.assertEqual(dataset, expected)

    def test_with_query(self):
        # This test requires a local mongo server to be setup on port 27017
        connection_str = "mongodb://localhost:27017"
        db_name = "test_db"
        collection_name = "test_collection"

        client = MongoClient(connection_str)
        collection = client[db_name][collection_name]
        doc_1 = {
            "option": "test",
            "x0": 1,
            "xt": 2,
            "y0": False,
            "yt": False,
            "duration": 5,
        }
        doc_2 = {
            "option": "test",
            "x0": 2,
            "xt": 3,
            "y0": True,
            "yt": False,
            "duration": 5,
        }
        doc_3 = {
            "option": "test",
            "x0": 3,
            "xt": 1,
            "y0": False,
            "yt": True,
            "duration": 7,
        }
        doc_4 = {  # Zero cost self loop, should be removed
            "option": "test",
            "x0": 1,
            "xt": 1,
            "y0": False,
            "yt": False,
            "duration": 0,
        }
        collection.insert_many([doc_1, doc_2, doc_3, doc_4])

        sf_list = [StateFactor("x", [1, 2, 3]), BoolStateFactor("y")]
        dataset = mongodb_to_dict(
            connection_str, db_name, collection_name, sf_list, query={"duration": 7}
        )
        client.drop_database("test_db")

        expected = {
            "test": {
                "transition": {
                    "x0": [3],
                    "xt": [1],
                    "y0": [False],
                    "yt": [True],
                },
                "reward": {"x": [3], "y": [False], "r": [7]},
            }
        }

        self.assertEqual(dataset, expected)


@unittest.skipIf(not MONGO_RUNNING, "MongoDB server not running.")
class MongodbToYamlTest(unittest.TestCase):

    def test_function(self):
        # This test requires a local mongo server to be setup on port 27017
        connection_str = "mongodb://localhost:27017"
        db_name = "test_db"
        collection_name = "test_collection"

        client = MongoClient(connection_str)
        collection = client[db_name][collection_name]
        doc_1 = {
            "option": "test",
            "x0": 1,
            "xt": 2,
            "y0": False,
            "yt": False,
            "duration": 5,
        }
        doc_2 = {
            "option": "test",
            "x0": 2,
            "xt": 3,
            "y0": True,
            "yt": False,
            "duration": 5,
        }
        doc_3 = {
            "option": "test",
            "x0": 3,
            "xt": 1,
            "y0": False,
            "yt": True,
            "duration": 7,
        }
        doc_4 = {  # Zero cost self loop, should be removed
            "option": "test",
            "x0": 1,
            "xt": 1,
            "y0": False,
            "yt": False,
            "duration": 0,
        }
        collection.insert_many([doc_1, doc_2, doc_3, doc_4])

        yaml_file = "dataset_test.yaml"

        sf_list = [StateFactor("x", [1, 2, 3]), BoolStateFactor("y")]
        mongodb_to_yaml(connection_str, db_name, collection_name, sf_list, yaml_file)
        client.drop_database("test_db")

        expected = {
            "test": {
                "transition": {
                    "x0": [1, 2, 3],
                    "xt": [2, 3, 1],
                    "y0": [False, True, False],
                    "yt": [False, False, True],
                },
                "reward": {"x": [1, 2, 3], "y": [False, True, False], "r": [5, 5, 7]},
            }
        }

        assert os.path.exists(yaml_file)
        with open(yaml_file, "r") as yaml_in:
            dataset = yaml.load(yaml_in, Loader=yaml.FullLoader)

        self.assertEqual(dataset, expected)

        os.remove(yaml_file)


class CheckDatasetTest(unittest.TestCase):

    def test_function(self):
        dataset = 4
        with self.assertRaises(Exception):
            _check_dataset(dataset, [])

        dataset = {"transition": {}}
        with self.assertRaises(Exception):
            _check_dataset(dataset, [])

        dataset = {"reward": {}, "rand": {}}
        with self.assertRaises(Exception):
            _check_dataset(dataset, [])

        dataset = {"transition": {}, "rand": {}}
        with self.assertRaises(Exception):
            _check_dataset(dataset, [])

        dataset = {"transition": [], "reward": {}}
        with self.assertRaises(Exception):
            _check_dataset(dataset, [])

        dataset = {"transition": {}, "reward": []}
        with self.assertRaises(Exception):
            _check_dataset(dataset, [])

        sf_list = [StateFactor("x", [1, 2, 3]), BoolStateFactor("y")]
        dataset = {
            "test": {
                "transition": {
                    "xt": [2, 3, 1],
                    "y0": [False, True, False],
                    "yt": [False, False, True],
                },
                "reward": {"x": [1, 2, 3], "y": [False, False, True], "r": [5, 5, 7]},
            }
        }
        with self.assertRaises(Exception):
            _check_dataset(dataset, sf_list)

        dataset = {
            "test": {
                "transition": {
                    "x0": "woops",
                    "xt": [2, 3, 1],
                    "y0": [False, True, False],
                    "yt": [False, False, True],
                },
                "reward": {"x": [1, 2, 3], "y": [False, False, True], "r": [5, 5, 7]},
            }
        }
        with self.assertRaises(Exception):
            _check_dataset(dataset, sf_list)

        dataset = {
            "test": {
                "transition": {
                    "x0": [1, 2, 3],
                    "y0": [False, True, False],
                    "yt": [False, False, True],
                },
                "reward": {"x": [1, 2, 3], "y": [False, False, True], "r": [5, 5, 7]},
            }
        }
        with self.assertRaises(Exception):
            _check_dataset(dataset, sf_list)

        dataset = {
            "test": {
                "transition": {
                    "x0": [1, 2, 3],
                    "xt": "woops",
                    "y0": [False, True, False],
                    "yt": [False, False, True],
                },
                "reward": {"x": [1, 2, 3], "y": [False, False, True], "r": [5, 5, 7]},
            }
        }
        with self.assertRaises(Exception):
            _check_dataset(dataset, sf_list)

        dataset = {
            "test": {
                "transition": {
                    "x0": [1, 2, 3],
                    "xt": [2, 3, 1],
                    "y0": [False, True, False],
                    "yt": [False, False, True],
                },
                "reward": {"y": [False, False, True], "r": [5, 5, 7]},
            }
        }
        with self.assertRaises(Exception):
            _check_dataset(dataset, sf_list)

        dataset = {
            "test": {
                "transition": {
                    "x0": [1, 2, 3],
                    "xt": [2, 3, 1],
                    "y0": [False, True, False],
                    "yt": [False, False, True],
                },
                "reward": {"x": "woops", "y": [False, False, True], "r": [5, 5, 7]},
            }
        }
        with self.assertRaises(Exception):
            _check_dataset(dataset, sf_list)

        dataset = {
            "test": {
                "transition": {
                    "x0": [1, 3],
                    "xt": [2, 3, 1],
                    "y0": [False, True, False],
                    "yt": [False, False, True],
                },
                "reward": {"x": [1, 2, 3], "y": [False, False, True], "r": [5, 5, 7]},
            }
        }
        with self.assertRaises(Exception):
            _check_dataset(dataset, sf_list)

        dataset = {
            "test": {
                "transition": {
                    "x0": [1, 2, 3],
                    "xt": [2, 3, 1],
                    "y0": [False, True, False],
                    "yt": [False, False],
                },
                "reward": {"x": [1, 2, 3], "y": [False, False, True], "r": [5, 5, 7]},
            }
        }
        with self.assertRaises(Exception):
            _check_dataset(dataset, sf_list)

        dataset = {
            "test": {
                "transition": {
                    "x0": [1, 2, 3],
                    "xt": [2, 3, 1],
                    "y0": [False, True, False],
                    "yt": [False, False, True],
                },
                "reward": {"x": [1, 2, 3], "y": [False, True], "r": [5, 5, 7]},
            }
        }
        with self.assertRaises(Exception):
            _check_dataset(dataset, sf_list)

        dataset = {
            "test": {
                "transition": {
                    "x0": [1, 2, 3],
                    "xt": [2, 3, 1],
                    "y0": [False, True, False],
                    "yt": [False, False, True],
                },
                "reward": {"x": [1, 2, 3], "y": [False, False, True]},
            }
        }
        with self.assertRaises(Exception):
            _check_dataset(dataset, sf_list)

        dataset = {
            "test": {
                "transition": {
                    "x0": [1, 2, 3],
                    "xt": [2, 3, 1],
                    "y0": [False, True, False],
                    "yt": [False, False, True],
                },
                "reward": {"x": [1, 2, 3], "y": [False, False, True], "r": "woops"},
            }
        }
        with self.assertRaises(Exception):
            _check_dataset(dataset, sf_list)

        dataset = {
            "test": {
                "transition": {
                    "x0": [1, 2, 3],
                    "xt": [2, 3, 1],
                    "y0": [False, True, False],
                    "yt": [False, False, True],
                },
                "reward": {"x": [1, 2, 3], "y": [False, False, True], "r": [5, 7]},
            }
        }
        with self.assertRaises(Exception):
            _check_dataset(dataset, sf_list)

        dataset = {
            "test": {
                "transition": {
                    "x0": [1, 2, 3],
                    "xt": [2, 3, 1],
                    "y0": [False, True, False],
                    "yt": [False, False, True],
                },
                "reward": {"x": [1, 2, 3], "y": [False, False, True], "r": [5, 5, 7]},
            }
        }
        _check_dataset(dataset, sf_list)


class DatasetValsToStrTest(unittest.TestCase):

    def test_function(self):
        dataset = {
            "test": {
                "transition": {
                    "x0": [1, 2, 3],
                    "xt": [2, 3, 1],
                    "y0": [False, True, False],
                    "yt": [False, False, True],
                },
                "reward": {"x": [1, 2, 3], "y": [False, False, True], "r": [5, 5, 7]},
            }
        }

        str_dataset = _dataset_vals_to_str(dataset)

        expected = {
            "test": {
                "transition": {
                    "x0": ["1", "2", "3"],
                    "xt": ["2", "3", "1"],
                    "y0": ["False", "True", "False"],
                    "yt": ["False", "False", "True"],
                },
                "reward": {
                    "x": ["1", "2", "3"],
                    "y": ["False", "False", "True"],
                    "r": ["5", "5", "7"],
                },
            }
        }

        self.assertEqual(str_dataset, expected)


class RemoveUnchangingVarsTest(unittest.TestCase):

    def test_function(self):
        option_1_dataset = {
            "transition": {
                "x0": [1, 2, 3],
                "xt": [2, 3, 1],
                "y0": [False, True, False],
                "yt": [False, False, True],
            },
            "reward": {"x": [1, 2, 3], "y": [False, False, True], "r": [5, 5, 7]},
        }

        option_2_dataset = {
            "transition": {
                "x0": [1, 2, 3],
                "xt": [2, 3, 1],
                "y0": [False, False, True],
                "yt": [False, False, True],
            },
            "reward": {"x": [1, 2, 3], "y": [False, False, True], "r": [5, 5, 7]},
        }

        dataset = {"opt1": option_1_dataset, "opt2": option_2_dataset}

        sf_list = [StateFactor("x", [1, 2, 3]), BoolStateFactor("y")]

        new_dataset = _remove_unchanging_vars(dataset, sf_list)

        self.assertEqual(len(new_dataset), 2)
        self.assertEqual(new_dataset["opt1"], option_1_dataset)

        ex_opt2_filtered = {
            "transition": {
                "x0": [1, 2, 3],
                "xt": [2, 3, 1],
                "y0": [False, False, True],
            },
            "reward": {"x": [1, 2, 3], "y": [False, False, True], "r": [5, 5, 7]},
        }

        self.assertEqual(new_dataset["opt2"], ex_opt2_filtered)


class SetupLearnersTest(unittest.TestCase):

    def test_function(self):
        dataset = {
            "test": {
                "transition": {
                    "x0": ["1", "2", "3"],
                    "xt": ["2", "3", "1"],
                    "y0": ["False", "True", "False"],
                    "yt": ["False", "False", "True"],
                },
                "reward": {
                    "x": ["1", "2", "3"],
                    "y": ["False", "False", "True"],
                    "r": ["5", "5", "7"],
                },
            }
        }

        sf_list = [StateFactor("x", [1, 2, 3]), BoolStateFactor("y")]

        trans_learner, reward_learner = _setup_learners(dataset["test"], sf_list)

        trans_forbidden_arcs = trans_learner.state()["Constraint Forbidden Arcs"][0]
        self.assertTrue("x0->x0" in trans_forbidden_arcs)
        self.assertTrue("y0->y0" in trans_forbidden_arcs)
        self.assertTrue("x0->y0" in trans_forbidden_arcs)
        self.assertTrue("y0->x0" in trans_forbidden_arcs)
        self.assertTrue("xt->x0" in trans_forbidden_arcs)
        self.assertTrue("xt->y0" in trans_forbidden_arcs)
        self.assertTrue("yt->x0" in trans_forbidden_arcs)
        self.assertTrue("yt->y0" in trans_forbidden_arcs)
        self.assertEqual(len(trans_forbidden_arcs), 64)

        reward_forbidden_arcs = reward_learner.state()["Constraint Forbidden Arcs"][0]
        self.assertTrue("x->x" in reward_forbidden_arcs)
        self.assertTrue("y->y" in reward_forbidden_arcs)
        self.assertTrue("x->y" in reward_forbidden_arcs)
        self.assertTrue("y->x" in reward_forbidden_arcs)
        self.assertTrue("r->y" in reward_forbidden_arcs)
        self.assertTrue("r->x" in reward_forbidden_arcs)
        self.assertEqual(len(reward_forbidden_arcs), 36)

    def test_with_removed_var(self):
        dataset = {
            "test": {
                "transition": {
                    "x0": ["1", "2", "3"],
                    "xt": ["2", "3", "1"],
                    "y0": ["False", "True", "False"],
                },
                "reward": {
                    "x": ["1", "2", "3"],
                    "y": ["False", "False", "True"],
                    "r": ["5", "5", "7"],
                },
            }
        }

        sf_list = [StateFactor("x", [1, 2, 3]), BoolStateFactor("y")]

        trans_learner, reward_learner = _setup_learners(dataset["test"], sf_list)

        trans_forbidden_arcs = trans_learner.state()["Constraint Forbidden Arcs"][0]
        self.assertTrue("x0->x0" in trans_forbidden_arcs)
        self.assertTrue("y0->y0" in trans_forbidden_arcs)
        self.assertTrue("x0->y0" in trans_forbidden_arcs)
        self.assertTrue("y0->x0" in trans_forbidden_arcs)
        self.assertTrue("xt->x0" in trans_forbidden_arcs)
        self.assertTrue("xt->y0" in trans_forbidden_arcs)
        self.assertEqual(len(trans_forbidden_arcs), 48)

        reward_forbidden_arcs = reward_learner.state()["Constraint Forbidden Arcs"][0]
        self.assertTrue("x->x" in reward_forbidden_arcs)
        self.assertTrue("y->y" in reward_forbidden_arcs)
        self.assertTrue("x->y" in reward_forbidden_arcs)
        self.assertTrue("y->x" in reward_forbidden_arcs)
        self.assertTrue("r->y" in reward_forbidden_arcs)
        self.assertTrue("r->x" in reward_forbidden_arcs)
        self.assertEqual(len(reward_forbidden_arcs), 36)


class RemoveEdgelessVarsTest(unittest.TestCase):

    def test_function(self):
        # Create the transition function DBN
        t_bn = gum.BayesNet()
        _ = t_bn.add(gum.LabelizedVariable("x0", "x0?", ["False", "True"]))
        _ = t_bn.add(gum.LabelizedVariable("xt", "xt?", ["False", "True"]))
        _ = t_bn.add(gum.LabelizedVariable("y0", "y0?", ["False", "True"]))
        _ = t_bn.add(gum.LabelizedVariable("yt", "yt?", ["False", "True"]))
        t_bn.addArc("x0", "xt")
        t_bn.addArc("x0", "yt")
        t_bn.addArc("y0", "xt")
        t_bn.addArc("y0", "yt")

        t_bn.cpt("xt")[{"x0": "False", "y0": "False"}] = [0.4, 0.6]
        t_bn.cpt("xt")[{"x0": "False", "y0": "True"}] = [0.5, 0.5]
        t_bn.cpt("xt")[{"x0": "True", "y0": "False"}] = [0.6, 0.4]
        t_bn.cpt("xt")[{"x0": "True", "y0": "True"}] = [0.3, 0.7]

        t_bn.cpt("yt")[{"x0": "False", "y0": "False"}] = [0.8, 0.2]
        t_bn.cpt("yt")[{"x0": "False", "y0": "True"}] = [0.1, 0.9]
        t_bn.cpt("yt")[{"x0": "True", "y0": "False"}] = [0.7, 0.3]
        t_bn.cpt("yt")[{"x0": "True", "y0": "True"}] = [0.2, 0.8]

        # Create the reward DBN
        r_bn = gum.BayesNet()
        _ = r_bn.add(gum.LabelizedVariable("x", "x?", ["False", "True"]))
        _ = r_bn.add(gum.LabelizedVariable("y", "y?", ["False", "True"]))
        _ = r_bn.add(gum.LabelizedVariable("r", "r?", ["0", "1", "2", "3"]))
        r_bn.addArc("x", "r")
        r_bn.addArc("y", "r")

        r_bn.cpt("x").fillWith([0.5, 0.5])
        r_bn.cpt("y").fillWith([0.5, 0.5])
        r_bn.cpt("r")[{"x": "False", "y": "False"}] = [0.0, 0.2, 0.3, 0.5]
        r_bn.cpt("r")[{"x": "False", "y": "True"}] = [0.0, 0.6, 0.1, 0.3]
        r_bn.cpt("r")[{"x": "True", "y": "False"}] = [0.0, 0.4, 0.4, 0.2]
        r_bn.cpt("r")[{"x": "True", "y": "True"}] = [0.0, 0.3, 0.3, 0.4]

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]

        _remove_edgeless_vars(t_bn, sf_list, True)
        self.assertEqual(sorted(list(t_bn.names())), sorted(["x0", "xt", "y0", "yt"]))
        t_bn.eraseArc("y0", "xt")
        t_bn.eraseArc("y0", "yt")
        _remove_edgeless_vars(t_bn, sf_list, True)
        self.assertEqual(sorted(list(t_bn.names())), sorted(["x0", "xt", "yt"]))

        _remove_edgeless_vars(r_bn, sf_list, False)
        self.assertEqual(sorted(list(r_bn.names())), sorted(["x", "y", "r"]))
        r_bn.eraseArc("x", "r")
        _remove_edgeless_vars(r_bn, sf_list, False)
        self.assertEqual(sorted(list(r_bn.names())), sorted(["y", "r"]))


class LearnBNsForOneOptionTest(unittest.TestCase):

    def test_function(self):
        dataset = {
            "transition": {
                "x0": [1, 2, 3],
                "xt": [2, 3, 1],
                "y0": [False, True, False],
                "yt": [False, False, True],
            },
            "reward": {"x": [1, 2, 3], "y": [False, False, True], "r": [5, 5, 7]},
        }

        sf_list = [StateFactor("x", [1, 2, 3]), BoolStateFactor("y")]

        transition_bn, reward_bn = learn_bns_for_one_option(dataset, sf_list)

        self.assertEqual(sorted(list(reward_bn.names())), sorted(["y", "r"]))
        self.assertEqual(
            sorted(list(transition_bn.names())), sorted(["x0", "y0", "xt", "yt"])
        )

    def test_with_useless_vars(self):
        dataset = {
            "transition": {
                "x0": [1, 2, 3],
                "xt": [2, 3, 1],
                "y0": [False, True, False],
                "yt": [False, True, False],
            },
            "reward": {"x": [1, 2, 3], "y": [False, False, True], "r": [5, 5, 7]},
        }

        sf_list = [StateFactor("x", [1, 2, 3]), BoolStateFactor("y")]

        transition_bn, reward_bn = learn_bns_for_one_option(dataset, sf_list)

        self.assertEqual(sorted(list(reward_bn.names())), sorted(["y", "r"]))
        self.assertEqual(
            sorted(list(transition_bn.names())), sorted(["x0", "y0", "xt"])
        )


class LearnDBNsTest(unittest.TestCase):

    def test_function(self):
        dataset = {
            "test": {
                "transition": {
                    "x0": [1, 2, 3],
                    "xt": [2, 3, 1],
                    "y0": [False, True, False],
                    "yt": [False, False, True],
                },
                "reward": {"x": [1, 2, 3], "y": [False, False, True], "r": [5, 5, 7]},
            }
        }

        with open("test_dataset.yaml", "w") as yaml_in:
            yaml.dump(dataset, yaml_in)

        sf_list = [StateFactor("x", [1, 2, 3]), BoolStateFactor("y")]

        learn_dbns("test_dataset.yaml", "", sf_list)

        self.assertTrue(os.path.exists("test_reward.bifxml"))
        self.assertTrue(os.path.exists("test_transition.bifxml"))

        reward_bn = gum.loadBN("test_reward.bifxml")
        transition_bn = gum.loadBN("test_transition.bifxml")

        self.assertEqual(sorted(list(reward_bn.names())), sorted(["y", "r"]))
        self.assertEqual(
            sorted(list(transition_bn.names())), sorted(["x0", "y0", "xt", "yt"])
        )

        os.remove("test_dataset.yaml")
        os.remove("test_reward.bifxml")
        os.remove("test_transition.bifxml")

    def test_with_useless_vars(self):
        dataset = {
            "test": {
                "transition": {
                    "x0": [1, 2, 3],
                    "xt": [2, 3, 1],
                    "y0": [False, True, False],
                    "yt": [False, True, False],
                },
                "reward": {"x": [1, 2, 3], "y": [False, False, True], "r": [5, 5, 7]},
            }
        }

        with open("test_dataset.yaml", "w") as yaml_in:
            yaml.dump(dataset, yaml_in)

        output_dir = ""
        sf_list = [StateFactor("x", [1, 2, 3]), BoolStateFactor("y")]

        learn_dbns("test_dataset.yaml", output_dir, sf_list)

        self.assertTrue(os.path.exists("test_reward.bifxml"))
        self.assertTrue(os.path.exists("test_transition.bifxml"))

        reward_bn = gum.loadBN("test_reward.bifxml")
        transition_bn = gum.loadBN("test_transition.bifxml")

        self.assertEqual(sorted(list(reward_bn.names())), sorted(["y", "r"]))
        self.assertEqual(
            sorted(list(transition_bn.names())), sorted(["x0", "y0", "xt"])
        )

        os.remove("test_dataset.yaml")
        os.remove("test_reward.bifxml")
        os.remove("test_transition.bifxml")


if __name__ == "__main__":
    unittest.main()
