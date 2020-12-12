import copy
import logging
import random
import time
from collections import defaultdict
from operator import itemgetter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jsonpickle  # type: ignore
import numpy as np
import texttable  # type: ignore
from numpy.lib.index_tricks import fill_diagonal
from openlock.common import Action
from openlock.logger_env import ActionLog
from openlockagents.common.io.log_io import pretty_write
from openlockagents.OpenLockLearner.causal_classes.CausalRelation import \
    CausalRelation
from openlockagents.OpenLockLearner.util.common import (ALL_CAUSAL_CHAINS,
                                                        check_for_duplicates)


class CausalChainStructureSpace:
    def __init__(
        self,
        causal_relation_space,
        chain_length,
        attributes,
        structure,
        attribute_order,
        lever_index_mode,
        print_messages=True,
        trie: bool = False,
        max_delay: int = 1,
    ):
        self.causal_relation_space = copy.deepcopy(causal_relation_space)
        self.chain_length = chain_length
        self.attributes = attributes
        self.structure = structure
        self.attribute_order = copy.copy(attribute_order)
        self.lever_index_mode = lever_index_mode
        self.state_index_in_attributes = self.attribute_order.index(
            self.lever_index_mode
        )
        self.print_messages = print_messages

        assert len(self.attribute_order) == len(
            attributes
        ), "Attribute order and attributes have different lengths"

        self.true_chains = []
        self.true_chain_idxs = []

        self.using_ids = False

        self.subchain_indexed_domains = [
            self.causal_relation_space.causal_relations_no_parent
        ]
        for _ in range(1, self.chain_length):
            self.subchain_indexed_domains.append(
                self.causal_relation_space.causal_relations_parent
            )

        self.causal_chains = self.generate_chains()

        self.subchain_indexed_causal_relation_to_chain_index_map = self.construct_chain_indices_by_subchain_index(
            self.causal_chains, self.chain_length
        )

        if trie:
            self.max_delay = max_delay
            self.trie = self._make_trie()

    def _make_trie(self) -> Dict:
        # Dict key consists of an action and how many unused actions after it
        Key = Tuple[Action, int]
        # Dict values consists of the next layer of the Trie and the indexes of the causal chains
        # consistent up to this point
        Value = Tuple[Dict[Key, Any], Sequence[int]]
        trie: Dict[Key, Value] = {}
        for idx, chain in enumerate(self.causal_chains):
            parents = [trie]
            children = []
            for subchain in chain:
                # We don't have the true delay, we just have the number of actions that don't work
                # So we want a mapping that uses the observed delay to include chains which have
                # shorter delays. This means that we add each chain along all paths through the
                # trie with delays at least as large as the one in this chain.
                for delay in range(subchain.delay, self.max_delay + 1):
                    key: Key = (subchain.action, delay)
                    for parent in parents:
                        child, indexes = parent.get(key, ({}, []))
                        indexes.append(idx)
                        children.append(child)
                        parent[key] = (child, indexes)
                parents = children
                children = []
        return trie

    def construct_chain_indices_by_subchain_index(self, chains, num_subchains):
        chain_indices_by_subchain_index = [
            defaultdict(set) for i in range(num_subchains)
        ]
        for chain_index in range(len(chains)):
            chain = chains[chain_index]
            for subchain_index in range(len(chain)):
                causal_relation = chain[subchain_index]
                chain_indices_by_subchain_index[subchain_index][causal_relation].add(
                    chain_index
                )
        return chain_indices_by_subchain_index

    def generate_chains(self):
        logging.debug("Generating chains")
        subchain_indexed_domains = [list(x) for x in self.subchain_indexed_domains]
        chains = []
        rejected_chains = []
        counter = 0
        total_num_chains = int(
            np.prod([len(chain_domain) for chain_domain in subchain_indexed_domains])
        )
        # generate all possible chains
        for i in range(len(subchain_indexed_domains[0])):
            root_subchain = subchain_indexed_domains[0][i]
            counter = self.recursive_chain_generation(
                root_subchain,
                [i],
                subchain_indexed_domains,
                depth=1,
                chains=chains,
                rejected_chains=rejected_chains,
                counter=counter,
            )
            logging.info(
                "{}/{} chains generated. {} valid chains".format(
                    counter, total_num_chains, len(chains)
                )
            )
        logging.info(
            "{}/{} chains generated. {} valid chains".format(
                counter, total_num_chains, len(chains)
            )
        )
        return chains

    def recursive_chain_generation(
        self,
        parent_subchain,
        parent_indices,
        chain_domains,
        depth,
        chains,
        rejected_chains,
        counter,
    ):
        terminal_depth = depth == len(chain_domains) - 1
        for i in range(len(chain_domains[depth])):
            child_subchain = chain_domains[depth][i]
            local_parent_indices = copy.copy(parent_indices)
            # verify postcondition of parent matches precondition of child
            if (
                parent_subchain.attributes,
                parent_subchain.causal_relation_type[1],
            ) == child_subchain.precondition:
                # if we aren't at the last chain domain, continue recursing
                if not terminal_depth:
                    local_parent_indices.append(i)
                    counter = self.recursive_chain_generation(
                        child_subchain,
                        local_parent_indices,
                        chain_domains,
                        depth + 1,
                        chains,
                        rejected_chains,
                        counter,
                    )
                # if we are at the terminal depth, this is the final check, add the chain to chains
                else:
                    # collect all parents
                    chain = [
                        chain_domains[j][local_parent_indices[j]]
                        for j in range(len(local_parent_indices))
                    ]
                    chain.append(child_subchain)
                    chains.append(tuple(chain))
                counter += 1
            else:
                if not terminal_depth:
                    counter += len(chain_domains[depth + 1])
                else:
                    counter += 1
        return counter

    def __getitem__(self, item):
        # handle slices
        if isinstance(item, slice):
            slice_result = CausalChainStructureSpace(
                self.causal_relation_space,
                self.chain_length,
                self.attributes,
                self.structure,
                self.attribute_order,
                self.lever_index_mode,
            )
            slice_result.causal_chains = self.causal_chains[item]
            return slice_result
        # handle slicing by an arbitrary list of indices
        elif isinstance(item, list) or isinstance(item, np.ndarray):
            slice_result = CausalChainStructureSpace(
                self.causal_relation_space,
                self.chain_length,
                self.attributes,
                self.structure,
                self.attribute_order,
                self.lever_index_mode,
            )
            slice_result.causal_chains = itemgetter(*self.causal_chains)(item)
            return slice_result
        # handle integer access
        elif isinstance(item, int) or np.issubdtype(item, np.integer):
            return self.causal_chains[item]
        else:
            raise TypeError("Invalid argument type")

    def __len__(self):
        return len(self.causal_chains)

    def _size(self):
        return len(self.causal_chains)

    def index(self, item, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = len(self.causal_chains)
        for i in range(start, end):
            if self.causal_chains[i] == item:
                return i

    @property
    def num_subchains_in_chain(self):
        if len(self.causal_chains) == 0:
            return 0
        else:
            return len(self.causal_chains[0])

    def append(self, causal_chain):
        self.causal_chains.append(causal_chain)

    def extend(self, causal_chain_manager):
        self.causal_chains.extend(causal_chain_manager.causal_chains)

    def find_causal_chain_idxs(self, target_chain):
        # matching_chains = []
        matching_chain_idxs = []
        for i in range(len(self.causal_chains)):
            if self.causal_chains[i] == target_chain:
                # matching_chains.append(self.causal_chains[i])
                matching_chain_idxs.append(i)

        return matching_chain_idxs

    def legacy_chain_search(
        self, actions: Sequence[Action], change_observed: Sequence[bool],
    ) -> List[int]:
        """ Finds all chains consistent with the given action sequence and action success vector
        by brute force searching over all chains, which is slow, but this has been debugged."""
        # TODO(joschnei): Probably delete this, I never got it to work in the first place.
        matching_chain_idxs = list()
        for idx, chain in enumerate(self.causal_chains):
            timestep = 0
            skip_chain = False
            for causal_relation in chain:
                if skip_chain:
                    break

                if timestep >= len(actions):
                    break

                if not change_observed[timestep]:
                    timestep += 1
                    continue
                elif causal_relation.action != actions[timestep]:
                    skip_chain = True
                    break

                for dud_timestep in range(1, causal_relation.delay + 1):
                    if timestep + dud_timestep >= len(actions):
                        break
                    elif change_observed[timestep + dud_timestep]:
                        skip_chain = True
                        break

                timestep += 1 + causal_relation.delay
            if not skip_chain:
                matching_chain_idxs.append(idx)
        return matching_chain_idxs

    def find_causal_chain_idxs_with_actions(
        self,
        actions: Sequence[Action],
        change_observed: Sequence[bool],
        legacy: bool = False,
    ) -> List[int]:
        assert len(actions) == len(
            change_observed
        ), f"action and change lengths don't match {len(actions)} vs {len(change_observed)}"

        logging.debug(f"actions={actions}, change_observed={change_observed}")

        if legacy:
            return self.legacy_chain_search(actions, change_observed)

        if getattr(self, "trie", None) is not None:
            return self.get_chains_from_actions(actions, change_observed)

        inclusion_constraints: List[Dict[str, Union[Any, List[Any]]]] = []
        max_delay = 0
        last_good_action = None
        for action, change in zip(actions, change_observed):
            if not change:
                max_delay += 1
                continue
            if last_good_action is not None:
                new_constraint = {
                    "action": last_good_action,
                    "delay": list(range(max_delay + 1)),
                }
                logging.debug(f"Adding constraint {new_constraint}")
                inclusion_constraints.append(new_constraint)

            last_good_action = action
            max_delay = 0

        if last_good_action is not None:
            # The last good action has no constraint on delay, as any number of future actions might fail.
            inclusion_constraints.append({"action": last_good_action})

        return self.find_all_causal_chains_satisfying_constraints(inclusion_constraints)

    def _make_causal_events(
        self, actions: Sequence[Action], change_observed: Sequence[bool]
    ) -> Sequence[Tuple[Action, int]]:
        assert len(actions) == len(change_observed)
        events: Sequence[Tuple[Action, int]] = []
        noops = 0
        last_good_action = None
        for action, change in zip(actions, change_observed):
            if not change:
                noops += 1
            else:
                if last_good_action is not None:
                    events.append((last_good_action, noops))
                    noops = 0
                last_good_action = action
        if last_good_action is not None:
            events.append((last_good_action, self.max_delay))
        return events

    def get_chains_from_actions(
        self, actions: Sequence[Action], change_observed: Sequence[bool]
    ) -> List[int]:
        assert len(actions) == len(change_observed)
        causal_events = self._make_causal_events(actions, change_observed)
        if len(causal_events) > 0:
            node = self.trie
            for event in causal_events:
                node, indexes = node[event]
        else:
            indexes = list(range(len(self.causal_chains)))
        return indexes

    def find_all_causal_chains_satisfying_constraints(
        self,
        inclusion_constraints: Sequence[Dict[str, Union[Any, List[Any]]]],
        exclusion_constraints: Sequence[Sequence[CausalRelation]] = None,
    ):
        if exclusion_constraints is None:
            exclusion_constraints = set()

        causal_chain_indices_satisfying_constraints_at_subchain_indices = [
            [] for i in range(len(self.subchain_indexed_domains))
        ]
        # collection relations that satisfy constraints at each index
        for subchain_index in range(len(inclusion_constraints)):
            # if we have a constraint, find causal relations that adhere to the constraints
            if len(inclusion_constraints[subchain_index]) > 0:
                # TODO(mjedmonds): refactor these to not directly access member classes
                causal_relations_satisfying_constraints = self.causal_relation_space.find_relations_satisfying_constraints(
                    self.subchain_indexed_domains[subchain_index],
                    inclusion_constraints=inclusion_constraints[subchain_index],
                    exclusion_constraints=exclusion_constraints,
                )
                # constraints constitute all causal chains
                if len(causal_relations_satisfying_constraints) == len(
                    self.subchain_indexed_domains[subchain_index]
                ):
                    causal_chain_indices_satisfying_constraints_at_subchain_indices[
                        subchain_index
                    ] = ALL_CAUSAL_CHAINS
                    continue
            else:
                # remove all exclusion constraints from possible relations at this index
                causal_relations_satisfying_constraints = (
                    self.subchain_indexed_domains[subchain_index]
                    - exclusion_constraints
                )

            # now we have a list of all relations that satisfy the constraint at this subchain index
            # find the causal chain indices that have these relations at this subchain index
            causal_chain_indices_satisfying_constraints_at_subchain_indices[
                subchain_index
            ] = self.find_causal_chain_indices_satisfying_constraints_at_subchain_index(
                subchain_index, causal_relations_satisfying_constraints
            )
        # if inclusion_constraints did not specify constraints for every subchain, find the remaining
        for subchain_index in range(
            len(inclusion_constraints), len(self.subchain_indexed_domains)
        ):
            causal_relations_satisfying_constraints = self.causal_relation_space.find_relations_satisfying_constraints(
                self.subchain_indexed_domains[subchain_index],
                inclusion_constraints=None,
                exclusion_constraints=exclusion_constraints,
            )
            causal_chain_indices_satisfying_constraints_at_subchain_indices[
                subchain_index
            ] = self.find_causal_chain_indices_satisfying_constraints_at_subchain_index(
                subchain_index, causal_relations_satisfying_constraints
            )

        # the final list of causal chain indices is the intersection of all indices satisfy constraints at each subchain index
        final_set_of_causal_chain_indices = set()
        all_subchain_indices_free = True
        optimal_subchain_order = np.argsort(
            [
                len(x)
                for x in causal_chain_indices_satisfying_constraints_at_subchain_indices
            ]
        )
        for subchain_index in optimal_subchain_order:
            # this subchain index has a constraint
            if (
                causal_chain_indices_satisfying_constraints_at_subchain_indices[
                    subchain_index
                ]
                != ALL_CAUSAL_CHAINS
            ):
                all_subchain_indices_free = False
                # if we already have values in the final set, take intersection
                if bool(final_set_of_causal_chain_indices):
                    final_set_of_causal_chain_indices = final_set_of_causal_chain_indices.intersection(
                        causal_chain_indices_satisfying_constraints_at_subchain_indices[
                            subchain_index
                        ]
                    )
                # otherwise start final set with instantiated
                else:
                    final_set_of_causal_chain_indices = causal_chain_indices_satisfying_constraints_at_subchain_indices[
                        subchain_index
                    ]

        if all_subchain_indices_free:
            # if every subchain index was free, return full range of causal chains
            return set(range(len(self.causal_chains)))
        else:
            return set(final_set_of_causal_chain_indices)

    def find_chain_indices_using_causal_relation_at_subchain_index(
        self, subchain_index, causal_relation
    ):
        return self.subchain_indexed_causal_relation_to_chain_index_map[subchain_index][
            causal_relation
        ]

    def find_causal_chain_indices_satisfying_constraints_at_subchain_index(
        self, subchain_index, causal_relations_satisfying_constraints
    ):
        causal_chain_indices_satisfying_constraints_at_subchain_index = set()
        for causal_relation in causal_relations_satisfying_constraints:
            causal_chain_indices_satisfying_constraints_at_subchain_index.update(
                self.find_chain_indices_using_causal_relation_at_subchain_index(
                    subchain_index, causal_relation
                )
            )
        return causal_chain_indices_satisfying_constraints_at_subchain_index

    def set_chains(self, causal_chains):
        self.causal_chains = copy.copy(causal_chains)

    def pop(self, pos=-1):
        return self.causal_chains.pop(pos)

    def clear(self):
        self.causal_chains.clear()

    def reset(self):
        self.true_chains.clear()
        self.true_chain_idxs.clear()

    def equals(self, item1, item2):
        return item1 == item2

    def chain_equals(self, idx, other):
        return self.equals(self.causal_chains[idx], other)

    def get_chain_idx(self, causal_chain):
        return self.causal_chains.index(causal_chain)

    def get_outcome(self, index):
        return tuple([x.causal_relation_type[1] for x in self.causal_chains[index]])

    def get_actions(
        self, index: int, fill_delays: bool = False
    ) -> Sequence[Optional[Action]]:
        # TODO(mjedmonds): 0 is hacked in here, need a way to index by the attribute we are indexing state on
        actions = list()
        for causal_relation in self.causal_chains[index]:
            actions.append(causal_relation.action)
            if fill_delays:
                # Add None for actions which don't matter.
                actions.extend([None] * causal_relation.delay)
        return actions

    def get_attributes(self, index):
        return tuple([x.attributes for x in self.causal_chains[index]])

    def remove_chains(self, chain_idxs_to_remove):
        for index in chain_idxs_to_remove:
            self.causal_chains.pop(index)

    def shuffle(self):
        random.shuffle(self.causal_chains)
        self.subchain_indexed_causal_relation_to_chain_index_map = self.construct_chain_indices_by_subchain_index(
            self.causal_chains, self.chain_length
        )

    @property
    def num_attributes(self):
        return len(self.attribute_order)

    def set_true_causal_chains(self, true_chains, belief_space):
        """
        sets the true chains for the chain space based on true_chains
        :param true_chains: list of CompactCausalChains representing the true solutions/causally plausible chains
        :return: nothing
        """
        t = time.time()
        logging.info("Setting true causal chains...")
        self.true_chains = true_chains
        self.true_chain_idxs = []
        for true_chain in self.true_chains:
            # if true_chain not in self.causal_chains:
            #     logging.warning(
            #         f"Causal chain {true_chain} not in {self.causal_chains}"
            #     )
            chain_idx = self.causal_chains.index(true_chain)
            self.true_chain_idxs.append(chain_idx)

        assert (
            len(self.true_chains) == len(self.true_chain_idxs)
            and None not in self.true_chain_idxs
        ), "Could not find all true chain indices in causal chain space"
        logging.info(
            "Setting true causal chains took {}s. True causal chains: ".format(
                time.time() - t
            )
        )
        self.pretty_print_causal_chain_idxs(
            self.true_chain_idxs, belief_space, print_messages=self.print_messages
        )

    def check_for_duplicate_chains(self):
        check_for_duplicates(self.causal_chains)

    def delete_causal_chains(self):
        if hasattr(self, "causal_chains"):
            del self.causal_chains

    def sample_chains(
        self, causal_chain_idxs, sample_size=None, action_sequences_executed=None
    ):

        # if we have no sample_size, sample all possible
        chain_sample_size = (
            len(causal_chain_idxs)
            if sample_size is None
            else min(len(causal_chain_idxs), sample_size)
        )

        assert (
            chain_sample_size != 0
        ), "Chain sample size is 0! No chains would be sampled"

        all_chains_executed = False

        # sample chains
        sampled_causal_chain_idxs = set()

        # randomly pick chains from selected list
        new_idxs = random.sample(range(len(causal_chain_idxs)), chain_sample_size)
        new_idxs = itemgetter(*new_idxs)(causal_chain_idxs)
        # prevent picking the same intervention twice
        if action_sequences_executed is not None and len(action_sequences_executed) > 0:
            new_idxs = [
                new_idxs[i]
                for i in range(len(new_idxs))
                if self.get_actions(new_idxs[i]) not in action_sequences_executed
            ]
            if len(new_idxs) == 0:
                all_chains_executed = True
        if isinstance(new_idxs, int):
            sampled_causal_chain_idxs.add(new_idxs)
        else:
            sampled_causal_chain_idxs.update(new_idxs)

        return (list(sampled_causal_chain_idxs), all_chains_executed)

    def get_all_chains_with_actions(self, actions):
        chains = []
        for causal_chain_idx in range(len(self.causal_chains)):
            chain_actions = self.get_actions(causal_chain_idx)
            if self.equals(chain_actions, actions):
                chains.append(causal_chain_idx)
        return chains

    def get_all_chains_with_attributes(self, attributes):
        chains = []
        for causal_chain_idx in range(len(self.causal_chains)):
            chain_attributes = self.get_attributes(causal_chain_idx)
            if self.equals(chain_attributes, attributes):
                chains.append(causal_chain_idx)
        return chains

    def pretty_print_causal_chain_idxs(
        self,
        causal_chain_idxs,
        beliefs=None,
        energies=None,
        q_values=None,
        belief_label="belief",
        print_messages=True,
    ):
        # suppress printing
        if not print_messages:
            return

        table = texttable.Texttable()

        chain_content = []
        if len(causal_chain_idxs) > 0:
            for i in range(len(causal_chain_idxs)):
                new_chain_content = self.pretty_list_compact_chain(causal_chain_idxs[i])
                if beliefs is not None:
                    new_chain_content.append(beliefs[causal_chain_idxs[i]])
                if q_values is not None:
                    new_chain_content.append(q_values[causal_chain_idxs[i]])
                if energies is not None:
                    new_chain_content.append(energies[causal_chain_idxs[i]])
                chain_content.append(new_chain_content)
        else:
            return

        num_subchains = self.num_subchains_in_chain
        headers = ["idx"]
        headers.extend(["subchain{}".format(i) for i in range(num_subchains)])

        alignment = ["l"]
        alignment.extend(["l" for i in range(num_subchains)])

        widths = [5]
        widths.extend([60 for i in range(num_subchains)])

        if beliefs is not None:
            headers.append(belief_label)
            alignment.append("l")
            widths.append(15)
        if q_values is not None:
            headers.append("q-value")
            alignment.append("l")
            widths.append(15)
        if energies is not None:
            headers.append("energy")
            alignment.append("l")
            widths.append(15)

        content = [headers]
        content.extend(chain_content)
        table.add_rows(content)
        table.set_cols_align(alignment)
        table.set_cols_width(widths)
        logging.info(table.draw())

    def pretty_list_compact_chain(self, causal_chain):
        # argument causal_chain is an index into self.causal_chains
        if isinstance(causal_chain, int) or np.issubdtype(causal_chain, np.integer):
            chain_chain = self.causal_chains[causal_chain]
        else:
            raise ValueError(
                "Expected causal chain index as causal_chain argument to pretty_list_compact_chain()"
            )
        l = [causal_chain]
        l.extend(
            [
                self.pretty_str_causal_relation(causal_relation)
                for causal_relation in chain_chain
            ]
        )
        return l

    @staticmethod
    def pretty_str_causal_relation(causal_relation):
        return "pre={},action={},attr={},fluent_change={}".format(
            causal_relation.precondition,
            causal_relation.action,
            causal_relation.attributes,
            causal_relation.causal_relation_type,
        )

    def print_random_set_of_causal_chains(self, causal_chain_idxs, num_chains=100):
        idxs = np.random.randint(0, len(causal_chain_idxs), size=num_chains)
        causal_chains_to_print = []
        for idx in idxs:
            causal_chains_to_print.append(causal_chain_idxs[idx])
        self.pretty_print_causal_chain_idxs(
            causal_chains_to_print, print_messages=self.print_messages
        )

    def pretty_print_causal_observations(
        self, causal_observations, print_messages=True
    ):
        # suppress printing
        if not print_messages:
            return

        table = texttable.Texttable()
        table.set_cols_align(["l", "l"])
        content = [["step", "causal relation"]]
        for i in range(len(causal_observations)):
            causal_observation = causal_observations[i]
            content.append(
                [
                    str(i),
                    self.pretty_str_causal_relation(causal_observation.causal_relation),
                ]
            )
        table.add_rows(content)
        table.set_cols_width([7, 130])
        logging.info(table.draw())

    def print_chains_above_threshold(self, belief_space, threshold):
        chains = []
        for causal_chain_idx in range(len(self.causal_chains)):
            chain_belief = belief_space.beliefs[causal_chain_idx]
            if chain_belief > threshold:
                chains.append(causal_chain_idx)
        if len(chains) > 0:
            self.pretty_print_causal_chain_idxs(
                chains, belief_space, print_messages=self.print_messages
            )

    @staticmethod
    def check_for_equality_or_containment(item, target):
        return item == target or item in target

    @staticmethod
    def extract_states_from_actions(actions):
        """
        extracts the states used in an action
        :param actions: a list of actions
        :return: states, the states used in actions
        """
        states = []
        # convert ActionLogs to str if needed
        if isinstance(actions[0], ActionLog):
            actions = [action.name for action in actions]
        for action in actions:
            state = action.split("_", 1)[1]
            assert state not in states, "Solutions should only use each lever/door once"
            states.append(state)
        return tuple(states)

    def write_to_json(self, filename):
        pretty_write(jsonpickle, filename)
