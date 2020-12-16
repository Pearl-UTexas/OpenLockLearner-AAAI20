import logging
from itertools import product
from multiprocessing import Value
from typing import List, Optional, Sequence

from openlock.common import ENTITY_STATES, Action
from openlock.settings_scenario import select_scenario
from openlock.settings_trial import LEVER_CONFIGS
from openlockagents.OpenLockLearner.causal_classes.CausalChain import CausalChainCompact
from openlockagents.OpenLockLearner.causal_classes.CausalRelation import (
    CausalRelation,
    CausalRelationType,
)
from openlockagents.OpenLockLearner.causal_classes.CausalRelationSpace import (
    ACTIONS,
    POSITIONS,
)
from openlockagents.OpenLockLearner.util.common import GRAPH_INT_TYPE


def generate_solutions_by_trial(scenario_name, trial_name):
    solution_chains = []
    scenario = select_scenario(scenario_name, use_physics=False)

    # TODO(mjedmonds): extract these from the environment/scenario somehow. these are hard-coded
    lever_cpt_choice = GRAPH_INT_TYPE(1)
    door_cpt_choice = GRAPH_INT_TYPE(0)
    lever_ending_state = GRAPH_INT_TYPE(ENTITY_STATES["LEVER_PUSHED"])
    door_ending_state = GRAPH_INT_TYPE(ENTITY_STATES["DOOR_OPENED"])

    scenario_solutions = scenario.solutions
    trial_levers = LEVER_CONFIGS[trial_name]
    for scenario_solution in scenario_solutions:
        solution_actions = []
        solution_states = []
        solution_cpt_choices = []
        solution_attributes = []
        solution_outcomes = []
        for action_log in scenario_solution:
            action_name = action_log.name
            state_name = action_name.split("_")[1]
            if state_name == "door":
                ending_state = door_ending_state
                cpt_choice = door_cpt_choice
            else:
                # determine position of lever based on role
                for trial_lever in trial_levers:
                    if (
                        get_one_of(trial_lever, ["LeverRole", "LeverRoleEnum"])
                        == state_name
                    ):
                        state_name = trial_lever.LeverPosition.name
                ending_state = lever_ending_state
                cpt_choice = lever_cpt_choice

            action_name = "push_" + state_name
            attributes = (state_name, "GREY")
            solution_actions.append(action_name)
            solution_states.append(state_name)
            solution_attributes.append(attributes)
            solution_cpt_choices.append(cpt_choice)
            solution_outcomes.append(ending_state)
        solution_chains.append(
            CausalChainCompact(
                states=tuple(solution_states),
                actions=tuple(solution_actions),
                conditional_probability_table_choices=tuple(solution_cpt_choices),
                outcomes=tuple(solution_outcomes),
                attributes=tuple(solution_attributes),
            )
        )
    return solution_chains


def get_one_of(object, attrs):
    for attr in attrs:
        out = getattr(object, attr, None)
        if out is not None:
            return out
    raise ValueError("None of the attrs in object")


def expand_wildcards(traj: Sequence[Optional[Action]]) -> Sequence[Sequence[Action]]:
    actions = [
        Action(name=action, obj=obj, params={})
        for action, obj in product(ACTIONS, POSITIONS)
    ]
    actions.append(Action(name="push", obj="door", params={}))
    trajs: List[List[Action]] = [[]]
    for given_action in traj:
        if given_action is None:
            trajs = [traj + [action] for traj in trajs for action in actions]
        else:
            trajs = [traj + [given_action] for traj in trajs]
    return trajs


def generate_solutions_by_trial_causal_relation(scenario_name, trial_name):
    solution_chains = []
    scenario = select_scenario(scenario_name, use_physics=False)

    # TODO(mjedmonds): extract these from the environment/scenario somehow. these are hard-coded
    lever_causal_relation_type = CausalRelationType.one_to_zero
    door_causal_relation_type = CausalRelationType.zero_to_one

    scenario_solutions = get_one_of(scenario, ["SOLUTIONS", "solutions"])
    trial_levers = LEVER_CONFIGS[trial_name]
    for scenario_solution in scenario_solutions:
        solution_chain = []
        precondition = None

        attributes = None
        causal_relation = None
        delay = 0

        first = True

        # We can't know what the delay is until we see the next action. So we don't create the
        # causal relation for an action until we see the next non-wildcard action.
        for action_log in scenario_solution:
            action_name = action_log.name
            if action_name == "*":
                if first:
                    raise ValueError("Solutions cannot start with a wildcard action.")
                delay += 1
                continue
            elif not first:
                solution_chain.append(
                    CausalRelation(
                        action=Action(name="push", obj=attributes[0], params=None),
                        attributes=attributes,
                        causal_relation_type=causal_relation,
                        precondition=precondition,
                        delay=delay,
                    )
                )
                delay = 0
                precondition = (attributes, causal_relation[1])

            first = False

            state_name = action_name.split("_")[1]
            if state_name == "door":
                causal_relation = door_causal_relation_type
            else:
                # determine position of lever based on role
                for trial_lever in trial_levers:
                    if (
                        get_one_of(trial_lever, ["LeverRole", "LeverRoleEnum"])
                        == state_name
                    ):
                        state_name = trial_lever.LeverPosition.name
                causal_relation = lever_causal_relation_type

            attributes = (state_name, "GREY")

        # Append the last action
        solution_chain.append(
            CausalRelation(
                action=Action("push", attributes[0], None),
                attributes=attributes,
                causal_relation_type=causal_relation,
                precondition=precondition,
                delay=delay,
            )
        )

        solution_chains.append(tuple(solution_chain))
    return solution_chains
