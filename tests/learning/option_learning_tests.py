#!/usr/bin/env python3
""" Unit tests for option learning functions.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import StateFactor, BoolStateFactor
from refine_plan.learning.option_learning import (
    _initialise_dict_for_option,
    mongodb_to_yaml,
    _check_dataset,
    _dataset_vals_to_str,
    _setup_learners,
    learn_dbns,
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
        collection.insert_many([doc_1, doc_2, doc_3])

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

        output_dir = "/home/charlie/work"
        sf_list = [StateFactor("x", [1, 2, 3]), BoolStateFactor("y")]

        learn_dbns("test_dataset.yaml", output_dir, sf_list)

        self.assertTrue(os.path.exists("/home/charlie/work/test_reward.bifxml"))
        self.assertTrue(os.path.exists("/home/charlie/work/test_transition.bifxml"))

        reward_bn = gum.loadBN("/home/charlie/work/test_reward.bifxml")
        transition_bn = gum.loadBN("/home/charlie/work/test_transition.bifxml")

        self.assertEqual(sorted(list(reward_bn.names())), sorted(["x", "y", "r"]))
        self.assertEqual(
            sorted(list(transition_bn.names())), sorted(["x0", "y0", "xt", "yt"])
        )

        os.remove("test_dataset.yaml")
        os.remove("/home/charlie/work/test_reward.bifxml")
        os.remove("/home/charlie/work/test_transition.bifxml")


if __name__ == "__main__":
    unittest.main()
