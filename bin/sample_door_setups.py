#!/usr/bin/env python3
""" Script to generate a set of door setups for bookstore/fake museum experiments.

Author: Charlie Street
Owner: Charlie Street
"""

import random
import yaml
import sys


def sample_door_setups(door_file, out_file, num_samples=100):
    """Sample a set of door setups.

    Args:
        door_file: A yaml file with all door information.
        out_file: The file to output the samples to
        num_samples: The number of setups to sample
    """

    with open(door_file, "r") as yaml_in:
        door_list = yaml.load(yaml_in, Loader=yaml.FullLoader)

    samples = []

    for _ in range(num_samples):
        door_setup = []
        for i in range(len(door_list)):
            door_setup.append(
                "closed" if random.random() <= door_list[i]["p_closed"] else "open"
            )
        samples.append(door_setup)

    with open(out_file, "w") as yaml_out:
        yaml.dump(samples, yaml_out)


if __name__ == "__main__":
    sample_door_setups(sys.argv[1], sys.argv[2])
