import pickle as pickle


def test_trie():
    chain_space = pickle.load(
        open("data/openlocklearner_causal_chain_space.pickle", "rb")
    )
    events = list(chain_space.trie.keys())
    ids = [1, 13]
    actions = [events[i][0] for i in ids]
    for change_observed in [[True, True], [True, False], [False, False]]:
        consistent_idxs = chain_space.get_chains_from_actions(actions, change_observed)
        events = chain_space._make_causal_events(actions, change_observed)
        consistent_chains = [chain_space.causal_chains[i] for i in consistent_idxs]
        for chain in consistent_chains:
            for chain_event, (action, _), in zip(chain, events):
                if str(chain_event.action) != str(action):
                    print(chain)
                    print(actions)
                    assert False
