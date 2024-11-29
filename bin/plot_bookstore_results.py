#!/usr/bin/env python
""" Script for plotting the bookstoreREFINE-PLAN results.

Author: Charlie Street
"""

from scipy.stats import mannwhitneyu
from pymongo import MongoClient
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sys

plt.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams.update({"font.size": 40})


def which_to_rerun(collection):
    """Check the results for a method and check which didn't reach the end.

    This is because of some annoying localisation issue I believe

    Args:
        collection: The MongoDB collection
        sf_names: A list of state factor names
    """

    # Group docs together by run_id
    docs_per_run = {}
    for doc in collection.find({}):
        if doc["run_id"] not in docs_per_run:
            docs_per_run[doc["run_id"]] = []
        docs_per_run[doc["run_id"]].append(doc)

    runs_sorted = []
    for run_id in docs_per_run:
        runs_sorted.append(
            sorted(docs_per_run[run_id], key=lambda d: d["date_started"])
        )

    print(type(runs_sorted[0][0]["_meta"]["inserted_at"]))
    runs_sorted.sort(key=lambda x: x[0]["_meta"]["inserted_at"])

    for i in range(len(runs_sorted)):
        if runs_sorted[i][-1]["locationt"] != "v8":
            print("RERUN {}".format(i))


def read_results_for_method(collection, sf_names):
    """Read the mongo results for a single method (i.e. a collection).

    Args:
        collection: The MongoDB collection
        sf_names: A list of state factor names

    Returns:
        results: The list of run durations
    """

    # Group docs together by run_id
    docs_per_run = {}
    for doc in collection.find({}):
        if doc["run_id"] not in docs_per_run:
            docs_per_run[doc["run_id"]] = []
        docs_per_run[doc["run_id"]].append(doc)

    # Sanity check each run
    results = []
    for run_id in docs_per_run:
        total_duration = 0.0
        in_order = sorted(docs_per_run[run_id], key=lambda d: d["date_started"])
        if in_order[0]["location0"] != "v1":
            continue
        if in_order[-1]["locationt"] != "v8":
            continue

        for i in range(len(in_order) - 1):
            total_duration += in_order[i]["duration"]
            for sf in sf_names:
                if in_order[i]["{}t".format(sf)] != in_order[i + 1]["{}0".format(sf)]:
                    continue

        total_duration += in_order[-1]["duration"]
        results.append(total_duration)

    assert len(results) == 100
    return results


def print_stats(init_results, refined_results):
    """Print the statistics for the initial and refined results.

    Args:
        init_results: The durations for the initial behaviour
        refined_results: The durations for the refined behaviour
    """
    print(
        "INITIAL BEHAVIOUR: AVG COST: {}; VARIANCE: {}".format(
            np.mean(init_results), np.var(init_results)
        )
    )
    print(
        "REFINED BEHAVIOUR: AVG COST: {}; VARIANCE: {}".format(
            np.mean(refined_results), np.var(refined_results)
        )
    )
    p = mannwhitneyu(
        refined_results,
        init_results,
        alternative="less",
    )[1]
    print(
        "REFINED BEHAVIOUR BETTER THAN INITIAL BT: p = {}, stat sig better = {}".format(
            p, p < 0.05
        )
    )


def set_box_colors(bp):
    plt.setp(bp["boxes"][0], color="tab:blue", linewidth=8.0)
    plt.setp(bp["caps"][0], color="tab:blue", linewidth=8.0)
    plt.setp(bp["caps"][1], color="tab:blue", linewidth=8.0)
    plt.setp(bp["whiskers"][0], color="tab:blue", linewidth=8.0)
    plt.setp(bp["whiskers"][1], color="tab:blue", linewidth=8.0)
    plt.setp(bp["fliers"][0], color="tab:blue")
    plt.setp(bp["medians"][0], color="tab:blue", linewidth=8.0)

    plt.setp(bp["boxes"][1], color="tab:red", linewidth=8.0)
    plt.setp(bp["caps"][2], color="tab:red", linewidth=8.0)
    plt.setp(bp["caps"][3], color="tab:red", linewidth=8.0)
    plt.setp(bp["whiskers"][2], color="tab:red", linewidth=8.0)
    plt.setp(bp["whiskers"][3], color="tab:red", linewidth=8.0)
    plt.setp(bp["medians"][1], color="tab:red", linewidth=8.0)


def plot_box_plot(init_results, refined_results):
    """Plot a box plot showing the initial and refined results.

    Args:
        init_results: The durations for the initial behaviour
        refined_results: The durations for the refined behaviour
    """

    box = plt.boxplot(
        [init_results, refined_results],
        whis=[0, 100],
        positions=[1, 2],
        widths=0.6,
    )
    set_box_colors(box)

    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=True,  # labels along the bottom edge are offcd
        labelsize=40,
    )
    plt.ylabel("Time to Reach Goal (s)")

    plt.xticks([1, 2], ["Initial BT", "Refined Behaviour"])

    plt.show()


if __name__ == "__main__":

    sf_names = ["v{}_door".format(v) for v in range(2, 7)]
    sf_names = ["location"] + sf_names
    client = MongoClient(sys.argv[1])
    db = client["refine-plan"]
    init_results = read_results_for_method(db["bookstore-initial"], sf_names)
    print("Initial Results Complete")
    # refined_results = read_results_for_method(db["bookstore-refined"], sf_names)
    # print_stats(init_results, refined_results)
    # plot_box_plot(init_results, refined_results)
