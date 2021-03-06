import time
from typing import Sequence, Tuple

from openlock.common import Action
from openlock.envs.openlock_env import OpenLockEnv
from openlockagents.common.common import DEBUGGING
from openlockagents.OpenLockLearner.causal_classes.CausalChainStructureSpace import (
    CausalChainStructureSpace,
)
from openlockagents.OpenLockLearner.causal_classes.CausalRelation import (
    CausalObservation,
    CausalRelation,
    CausalRelationType,
)
from openlockagents.OpenLockLearner.learner.ChainPruner import ChainPruner
from openlockagents.OpenLockLearner.util.common import print_message


class CausalLearner:
    def __init__(self, print_messages=True):
        self.chain_pruner = ChainPruner(print_messages)
        self.print_messages = print_messages

    # updates the learner's model based on the results
    def update_bottom_up_causal_model(
        self,
        env: OpenLockEnv,
        causal_chain_space: CausalChainStructureSpace,
        sequences_to_prune: Sequence[Tuple[Sequence[Action], Sequence[bool]]],
        trial_name: str,
        trial_count: int,
        attempt_count: int,
        prune_inconsitent_chains=True,
        multiproc=False,
    ):
        chain_idxs_consistent = (
            causal_chain_space.bottom_up_belief_space.get_idxs_with_belief_above_threshold()
        )
        chain_idxs_removed = []

        prev_num_chains_with_belief_above_threshold = (
            causal_chain_space.bottom_up_belief_space.num_idxs_with_belief_above_threshold
        )

        if prune_inconsitent_chains:
            # TODO(mjedmonds): this only processes one causal observation; cannot handle multiple fluent changes in one time step
            (
                chain_idxs_consistent,
                chain_idxs_removed,
            ) = self.chain_pruner.prune_inconsistent_chains_v2(
                causal_chain_space=causal_chain_space,
                causal_chain_idxs=chain_idxs_consistent,
                sequences_to_prune=sequences_to_prune,
            )

        map_chains = causal_chain_space.update_bottom_up_beliefs(
            env.attribute_order, trial_name, chain_idxs=chain_idxs_consistent,
        )

        # the remainder of this function is bookkeeping
        num_chains_with_belief_above_threshold = (
            causal_chain_space.bottom_up_belief_space.num_idxs_with_belief_above_threshold
        )
        num_chains_pruned = (
            prev_num_chains_with_belief_above_threshold
            - num_chains_with_belief_above_threshold
        )

        assert num_chains_pruned == len(
            chain_idxs_removed
        ), "Number of chains removed is incorrect"

        return map_chains, num_chains_pruned, chain_idxs_consistent, chain_idxs_removed

    @staticmethod
    def create_causal_observations(
        env, action_sequence, intervention_outcomes, trial_count, attempt_count
    ):
        causal_observations = []
        causal_change_idx = 0

        for i in range(len(action_sequence)):
            action = action_sequence[i]
            prev_state, cur_state = intervention_outcomes[i]
            causal_observations = CausalLearner.create_causal_observation(
                env,
                action,
                cur_state,
                prev_state,
                causal_observations,
                trial_count,
                attempt_count,
            )
        return causal_observations

    @staticmethod
    def create_causal_observation(
        env,
        action,
        cur_state,
        prev_state,
        causal_observations,
        trial_count,
        attempt_count,
    ):
        state_diff = cur_state - prev_state
        state_change_occurred = len(state_diff) > 0
        # TODO(mjedmonds): generalize to more than 1 state change
        if len(state_diff) > 2:
            print_message(
                trial_count,
                attempt_count,
                "More than one state change this iteration, chain assumes only one variable changes at a time: {}".format(
                    state_diff
                ),
            )

        precondition = None
        # need to check for previous effective precondition.
        # We could take an action with an effect, take an action with no effect, then take an action with an effect.
        # We want the precondition to carry over from the first action, so we need to find the preconditon of the last action with an effect
        for i in reversed(range(0, len(causal_observations))):
            if causal_observations[i].causal_relation.causal_relation_type is not None:
                precondition = (
                    causal_observations[i].causal_relation.attributes,
                    causal_observations[i].causal_relation.causal_relation_type[1],
                )
                # want the first precondition we find, so break
                break

        if state_change_occurred:
            # TODO(mjedmonds): refactor to include door_lock
            state_diff = [x for x in state_diff if x[0] != "door_lock"]
            # TODO(mjedmonds): this only handles a single state_diff per timestep
            assert (
                len(state_diff) < 2
            ), "Multiple fluents changing at each time step not yet implemented"
            state_diff = state_diff[0]
            causal_relation_type = CausalRelationType(state_diff[1])
            attributes = env.get_obj_attributes(state_diff[0])
            attributes = tuple(attributes[key] for key in env.attribute_order)
            causal_observations.append(
                CausalObservation(
                    CausalRelation(
                        action=action,
                        attributes=attributes,
                        causal_relation_type=causal_relation_type,
                        precondition=precondition,
                    ),
                    info_gain=None,
                )
            )
        return causal_observations
