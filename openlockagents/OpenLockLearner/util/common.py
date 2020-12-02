import argparse
import heapq
import multiprocessing
import os
from shutil import copytree, ignore_patterns

import numpy as np
import openlockagents.common.common as common
from openlockagents.OpenLockLearner.causal_classes.CausalRelation import (
    CausalRelationType,
)

# typedef for values to use during chain generation
GRAPH_INT_TYPE = np.uint8

PARALLEL_MAX_NBYTES = "50000M"

SANITY_CHECK_ELEMENT_LIMIT = 1000000

ALL_CAUSAL_CHAINS = 1


class AblationParams:
    def __init__(self):
        self.TOP_DOWN = False
        self.BOTTOM_UP = False
        self.PRUNING = False
        self.TOP_DOWN_FIRST_TRIAL = False
        self.INDEXED_DISTRIBUTIONS = False
        self.ACTION_DISTRIBUTION = False

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self)


def parse_arguments():
    parser = argparse.ArgumentParser("OpenLockLearner")
    parser.add_argument(
        "--savedir",
        metavar="DIR",
        type=str,
        help="directory to save the output of the OpenLockLearner",
    )
    parser.add_argument(
        "--scenario",
        metavar="XX4",
        type=str,
        help="training-testing scenarios. E.g. CE3-CC4. For baselines, use CE4 or CC4",
    )
    parser.add_argument(
        "--ablations",
        type=str,
        nargs="+",
        help="ablations, options are: 'top_down', 'bottom_up', 'pruning'",
    )
    parser.add_argument("--bypass_confirmation", action="store_true")
    parser.add_argument("--verbosity", type=str, help="Logging verbosity")
    args = parser.parse_args()
    return args


def decode_bad_jsonpickle_str(bad_bytes):
    # encode the bad byes as a unicode string
    bad_str = str(bad_bytes, "utf-8")
    # every 4th character is a legitimate character
    chars = []
    for i in range(0, len(bad_str), 4):
        chars.append(bad_str[i])
    return "".join(chars)


def write_source_code(project_src_path, destination_path):
    copytree(
        project_src_path,
        destination_path,
        ignore=ignore_patterns("*.mp4", "*.pyc", ".git", ".gitignore", ".gitmodules"),
    )


def setup_actions(states):
    actions = ["push_{}".format(x) for x in states if x is not "door_lock"] + [
        "pull_{}".format(x) for x in states if x is not "door_lock" and x is not "door"
    ]
    return actions


def renormalize(input_arr):
    return input_arr / sum(input_arr)


def verify_valid_probability_distribution(dist):
    return abs(sum(dist) - 1.0) < 0.000001 and min(dist) >= 0


def get_highest_N_values_and_idxs(N, arr, min_value=None):
    return (
        get_highest_N_values(N, arr, min_value),
        get_highest_N_idxs(N, arr, min_value),
    )


def get_highest_N_values(N, arr, min_value=None):
    return arr[get_highest_N_idxs(N, arr, min_value)]


def get_highest_N_idxs(N, arr, min_value=None):
    result_idxs = []
    if isinstance(arr, np.ndarray):
        result_idxs = heapq.nlargest(N, range(len(arr)), arr.take)
    elif isinstance(arr, list):
        result_idxs = heapq.nlargest(N, range(len(arr)), arr.__getitem__)
    else:
        ValueError("Unexpected array type")
    if min_value is not None:
        result_idxs = [x for x in result_idxs if arr[x] > min_value]
    return result_idxs


def get_lowest_N_values_and_idxs(N, arr, max_value=None):
    return get_lowest_N_values(N, arr, max_value), get_lowest_N_idxs(N, arr, max_value)


def get_lowest_N_values(N, arr, max_value=None):
    return arr[get_lowest_N_idxs(N, arr, max_value)]


def get_lowest_N_idxs(N, arr, max_value=None):
    result_idxs = []
    if isinstance(arr, np.ndarray):
        result_idxs = heapq.nsmallest(N, range(len(arr)), arr.take)
    elif isinstance(arr, list):
        result_idxs = heapq.nsmallest(N, range(len(arr)), arr.__getitem__)
    else:
        ValueError("Unexpected array type")
    if max_value is not None:
        result_idxs = [x for x in result_idxs if arr[x] < max_value]
    return result_idxs


def load_openlock_learner_config_json(
    path="openlockagents/OpenLockLearner/openlock_learner_config.json",
):
    return common.load_json_config(path)


def setup_structure_space_paths(config_data=None):
    if not config_data:
        config_data = load_openlock_learner_config_json()

    causal_chain_structure_space_path = os.path.expanduser(
        config_data["DATA_BASE_PATH"] + config_data["CAUSAL_CHAIN_PICKLE"]
    )
    two_solution_schemas_structure_space_path = os.path.expanduser(
        config_data["DATA_BASE_PATH"] + config_data["TWO_SOLUTION_SCHEMA_PICKLE"]
    )
    three_solution_schemas_structure_space_path = os.path.expanduser(
        config_data["DATA_BASE_PATH"] + config_data["THREE_SOLUTION_SCHEMA_PICKLE"]
    )

    return (
        causal_chain_structure_space_path,
        two_solution_schemas_structure_space_path,
        three_solution_schemas_structure_space_path,
    )


ACTION_REGEX_STR = "action([0-9]+)"
STATE_REGEX_STR = "state([0-9]+)"

GRAPH_BATCH_SIZE = 1000000

# Attributes
DUMMY_ATTRIBUTES = ["attr1", "attr2"]


# define causal chain over both state_space x actions

FLUENTS = [CausalRelationType.one_to_zero, CausalRelationType.zero_to_one]
FLUENT_STATES = [0, 1]
ACTIONS = ["push", "pull"]

STATES_ROLE = [
    "l0",
    "l1",
    "l2",
    "inactive0",
    "inactive1",
    "inactive2",
    "inactive3",
    "door_lock",
    "door",
]

ACTIONS_ROLE = setup_actions(STATES_ROLE)

DOOR_STATES = ["door_lock"]


CAUSAL_CHAIN_EDGES = (
    ("action0", "state0"),
    ("state0", "state1"),
    ("action1", "state1"),
    ("state1", "state2"),
    ("action2", "state2"),
)

THREE_LEVER_TRIALS = ["trial1", "trial2", "trial3", "trial4", "trial5", "trial6"]
FOUR_LEVER_TRIALS = ["trial7", "trial8", "trial9", "trial10", "trial11"]

TRUE_GRAPH_CPT_CHOICES = (1, 1, 0)
TRUE_GRAPH_CPT_CHOICES = tuple([GRAPH_INT_TYPE(x) for x in TRUE_GRAPH_CPT_CHOICES])
PLAUSIBLE_CPT_CHOICES = [
    TRUE_GRAPH_CPT_CHOICES,
    tuple([GRAPH_INT_TYPE(x) for x in (1, 1, 1)]),
    tuple([GRAPH_INT_TYPE(x) for x in (1, 0, 1)]),
]


def print_message(trial_count, attempt_count, message, print_message=True):
    if print_message:
        print("T{}.A{}: ".format(trial_count, attempt_count) + message)


def merge_perceptually_causal_relations_from_dict_of_trials(
    perceptually_causal_relations,
):
    merged_perceptually_causal_relations = []
    for key in perceptually_causal_relations.keys():
        merged_perceptually_causal_relations.extend(
            list(perceptually_causal_relations[key].values())
        )

    return merged_perceptually_causal_relations


def merge_solutions_from_dict_of_trials(true_chains):
    merged_true_chains = []
    for key in true_chains.keys():
        merged_true_chains.extend(true_chains[key])

    merged_true_chains = list(set(merged_true_chains))
    return merged_true_chains


def homogenous_list(seq):
    iseq = iter(seq)
    first_type = type(next(iseq))
    return first_type if all((type(x) is first_type) for x in iseq) else False


def create_birdirectional_dict(values, birdir_type):
    bidir_dict = {values[i]: birdir_type(i) for i in range(len(values))}
    bidir_dict.update(dict([reversed(i) for i in bidir_dict.items()]))
    return bidir_dict


def generate_slicing_indices(l, batch_size=None):
    # if no fixed batch size is specified, compute batch size based on cpu_count and list length
    if batch_size is None:
        # if the list is less than l, we can use len(l) cpus
        if len(l) < multiprocessing.cpu_count():
            batch_size = 1
        else:
            batch_size = max(2, len(l) // (multiprocessing.cpu_count()))
    # must start at 0 index
    slicing_indices = [0] + list(range(batch_size, len(l), batch_size))

    # assign last portion to end
    slicing_indices.append(len(l))
    return slicing_indices


def check_for_duplicates(l):
    seen = set()
    for i in range(len(l)):
        if l[i] in seen:
            return True, i
        seen.add(l[i])
    return False, None


if __name__ == "__main__":
    print("inside of causal_classes.py")
