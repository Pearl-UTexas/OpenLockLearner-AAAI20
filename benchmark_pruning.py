import pickle
import time

import fire
import numpy as np

from openlockagents.OpenLockLearner.causal_classes.CausalChainStructureSpace import (
    CausalChainStructureSpace,
)


def make_change_observed(length: int):
    out = []
    for i in range(1, length + 1):
        out.append([True] * i + [False] * (length - i))
    return out


def main(n_trials: int = 10000):
    chain_space: CausalChainStructureSpace = pickle.load(
        open("data/openlocklearner_causal_chain_space.pickle", "rb")
    )
    events = list(chain_space.trie.keys())

    # Old
    start = time.perf_counter()
    for n_actions in range(1, 3):
        for _ in range(n_trials):
            observed_events = [
                events[i]
                for i in np.random.randint(low=0, high=len(events), size=(n_actions,))
            ]
            actions = [event[0] for event in observed_events]
            for change_observed in make_change_observed(len(actions)):
                chain_space.find_causal_chain_idxs_with_actions(
                    actions, change_observed
                )
    end = time.perf_counter()
    print(f"Old method average speed={(end - start) / n_trials}")

    # New
    start = time.perf_counter()
    for n_actions in range(1, 3):
        for _ in range(n_trials):
            observed_events = [
                events[i]
                for i in np.random.randint(low=0, high=len(events), size=(n_actions,))
            ]
            actions = [event[0] for event in observed_events]
            for change_observed in make_change_observed(len(actions)):
                chain_space.get_chains_from_actions(actions, change_observed)
    end = time.perf_counter()
    print(f"New method average speed={(end - start) / n_trials}")


if __name__ == "__main__":
    fire.Fire(main)
