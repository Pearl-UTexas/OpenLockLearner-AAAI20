import pickle as pickle
from typing import List, Set

from openlockagents.OpenLockLearner.causal_classes.CausalChainStructureSpace import (
    CausalChainStructureSpace,
)


def make_change_observed(length: int):
    out = []
    for i in range(1, length + 1):
        out.append([True] * i + [False] * (length - i))
    return out


def test_trie():
    chain_space: CausalChainStructureSpace = pickle.load(
        open("data/openlocklearner_causal_chain_space.pickle", "rb")
    )
    possible_events = list(chain_space.trie.keys())
    ids = [1, 13, 2]
    actions = [possible_events[i][0] for i in ids]
    for change_observed in make_change_observed(3):
        consistent_idxs = chain_space.get_chain_idxs_from_actions(
            actions, change_observed
        )
        events = chain_space._make_causal_events(actions, change_observed)
        consistent_chains = [chain_space.causal_chains[i] for i in consistent_idxs]
        for chain in chain_space.causal_chains:
            consistent = True
            for subchain, event in zip(chain, events):
                if str(subchain.action) != str(event[0]) or subchain.delay > event[1]:
                    consistent = False
                    break
            if consistent:
                assert chain in consistent_chains

        # delays: List[Set[int]] = [set()] * 3
        # for chain in consistent_chains:
        #     for i, (chain_event, (action, _)), in enumerate(zip(chain, events)):
        #         if str(chain_event.action) != str(action):
        #             print(chain)
        #             print(actions)
        #             assert False
        #         delays[i].add(chain_event.delay)

        # for i, event in enumerate(events):
        #     for j in range(event[1] + 1):
        #         assert j in delays[i]

