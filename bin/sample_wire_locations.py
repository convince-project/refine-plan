#!/usr/bin/env python3
""" Script to generate a set of wire locations for the house experiments.

Author: Charlie Street
Owner: Charlie Street
"""

import numpy as np
import yaml
import sys


def sample_wire_locations(wire_prob_map, out_file, num_samples=100):
    """Sample a set of wire locations.

    Args:
        wire_prob_map: A yaml file with the wire's probabilistic map
        out_file: The file to output the samples to
        num_samples: The number of setups to sample
    """

    with open(wire_prob_map, "r") as yaml_in:
        prob_map = yaml.load(yaml_in, Loader=yaml.FullLoader)

    locs = [entry["node"] for entry in prob_map]
    probs = [entry["p_wire"] for entry in prob_map]

    samples = []

    for _ in range(num_samples):
        samples.append(str(np.random.choice(locs, p=probs)))

    with open(out_file, "w") as yaml_out:
        yaml.dump(samples, yaml_out)


if __name__ == "__main__":
    sample_wire_locations(sys.argv[1], sys.argv[2])
