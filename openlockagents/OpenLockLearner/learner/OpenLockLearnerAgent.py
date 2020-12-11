import copy
import logging
import math
import random
import time
from typing import List, Sequence, Tuple, Union

import numpy as np
import texttable  # type: ignore
from openlock.common import ENTITY_STATES, Action
from openlockagents.common.agent import Agent
from openlockagents.common.common import DEBUGGING
from openlockagents.OpenLockLearner.causal_classes.BeliefSpace import (
    AbstractSchemaBeliefSpace,
    AtomicSchemaBeliefSpace,
    BottomUpChainBeliefSpace,
    InstantiatedSchemaBeliefSpace,
    TopDownChainBeliefSpace,
)
from openlockagents.OpenLockLearner.causal_classes.OutcomeSpace import (
    Outcome,
    OutcomeSpace,
)
from openlockagents.OpenLockLearner.causal_classes.SchemaStructureSpace import (
    AtomicSchemaStructureSpace,
    InstantiatedSchemaStructureSpace,
)
from openlockagents.OpenLockLearner.causal_classes.StructureAndBeliefSpaceWrapper import (
    AbstractSchemaStructureAndBeliefWrapper,
    AtomicSchemaStructureAndBeliefWrapper,
    InstantiatedSchemaStructureAndBeliefWrapper,
    TopDownBottomUpStructureAndBeliefSpaceWrapper,
)
from openlockagents.OpenLockLearner.learner.CausalLearner import CausalLearner
from openlockagents.OpenLockLearner.learner.InterventionSelector import (
    InterventionSelector,
)
from openlockagents.OpenLockLearner.learner.ModelBasedRL import ModelBasedRLAgent
from openlockagents.OpenLockLearner.util.common import print_message
from openlockagents.OpenLockLearner.util.util import (
    generate_solutions_by_trial_causal_relation,
)


class OpenLockLearnerAgent(Agent):
    def __init__(self, env, causal_chain_structure_space, params, **kwargs):
        super(OpenLockLearnerAgent, self).__init__("OpenLockLearner", params, env)
        super(OpenLockLearnerAgent, self).setup_subject(
            human=False, project_src=params["src_dir"]
        )
        self.trial_order = []
        # dicts to keep track of what happened each trial
        self.rewards = dict()
        self.num_chains_with_belief_above_threshold_per_attempt = dict()
        self.attempt_count_per_trial = dict()
        self.num_attempts_between_solutions = dict()
        self.information_gains_per_attempt = dict()
        self.belief_thresholds_per_attempt = dict()
        self.intervention_chain_idxs_per_attempt = dict()
        self.interventions_per_attempt = dict()

        if "print_messages" in params.keys():
            self.print_messages = params["print_messages"]
        else:
            self.print_messages = True
        causal_chain_structure_space.print_messages = self.print_messages

        self.multiproc = params["multiproc"]
        self.deterministic = params["deterministic"]
        self.chain_sample_size = params["chain_sample_size"]
        self.lambda_multiplier = params["lambda_multiplier"]
        self.local_alpha_update = params["local_alpha_update"]
        self.global_alpha_update = params["global_alpha_update"]
        self.ablation = params["ablation_params"]
        self.epsilon = params["epsilon"]
        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon_active = params["epsilon_active"]
        self.epsilon_ratios = (
            params["epsilon_ratios"] if "epsilon_ratios" in params else None
        )

        self.causal_learner = CausalLearner(self.print_messages)
        self.intervention_selector = InterventionSelector(
            params["intervention_sample_size"], self.print_messages
        )
        self.model_based_agent = None

        # schema managers (structural)
        three_solution_abstract_schema_structure_space = kwargs[
            "three_solution_schemas"
        ]
        two_solution_abstract_schema_structure_space = kwargs["two_solution_schemas"]
        # shuffle the order of the abstract schemas (this help evenly distribute instantiated schemas among cpu cores
        random.shuffle(three_solution_abstract_schema_structure_space.schemas)
        random.shuffle(two_solution_abstract_schema_structure_space.schemas)

        atomic_schema_structure_space = AtomicSchemaStructureSpace()
        instantiated_schema_structure_space = InstantiatedSchemaStructureSpace()

        # belief managers
        bottom_up_components = copy.copy(env.attribute_labels)
        # TODO(mjedmonds): hacky way of adding action
        bottom_up_components["action"] = ["push", "pull"]
        bottom_up_chain_belief_space = BottomUpChainBeliefSpace(
            len(causal_chain_structure_space),
            bottom_up_components,
            use_indexed_distributions=not self.ablation.INDEXED_DISTRIBUTIONS,
            use_action_distribution=not self.ablation.ACTION_DISTRIBUTION,
        )
        top_down_chain_belief_space = TopDownChainBeliefSpace(
            len(causal_chain_structure_space), init_to_zero=False
        )
        three_solution_abstract_schema_belief_space = AbstractSchemaBeliefSpace(
            len(three_solution_abstract_schema_structure_space)
        )
        two_solution_abstract_schema_belief_space = AbstractSchemaBeliefSpace(
            len(two_solution_abstract_schema_structure_space)
        )
        atomic_schema_belief_space = AtomicSchemaBeliefSpace(
            len(atomic_schema_structure_space)
        )
        instantiated_schema_belief_space = InstantiatedSchemaBeliefSpace(0)

        # pair structures and beliefs
        self.causal_chain_space = TopDownBottomUpStructureAndBeliefSpaceWrapper(
            causal_chain_structure_space,
            bottom_up_chain_belief_space,
            top_down_chain_belief_space,
        )
        self.three_solution_abstract_schema_space = AbstractSchemaStructureAndBeliefWrapper(
            three_solution_abstract_schema_structure_space,
            three_solution_abstract_schema_belief_space,
        )
        self.two_solution_abstract_schema_space = AbstractSchemaStructureAndBeliefWrapper(
            two_solution_abstract_schema_structure_space,
            two_solution_abstract_schema_belief_space,
        )
        self.instantiated_schema_space = InstantiatedSchemaStructureAndBeliefWrapper(
            instantiated_schema_structure_space, instantiated_schema_belief_space
        )
        self.atomic_schema_space = AtomicSchemaStructureAndBeliefWrapper(
            atomic_schema_structure_space, atomic_schema_belief_space
        )
        self.current_abstract_schema_space = None

        self.outcome_space = [
            self.causal_chain_space.structure_space.get_outcome(x)
            for x in range(len(self.causal_chain_space.structure_space))
        ]

        self.observed_causal_observations = []

        self.attempt_count = 1
        self.trial_count = 1

    def setup_outcome_space(self, states, convert_to_ids=True):
        num_states_in_chain = (
            self.causal_chain_space.structure_space.num_subchains_in_chain
        )

        if convert_to_ids:
            states = self.causal_chain_space.structure_space.unique_id_manager.convert_states_to_target_type(
                states, target_type="int"
            )
        # convert outcome and intervention spaces to IDs if causal chain space is defined on IDs
        return OutcomeSpace(states, num_states_in_chain, using_ids=convert_to_ids)

    def setup_trial(
        self, scenario_name, action_limit, attempt_limit, specified_trial=None, **kwargs
    ):
        trial_selected = super(OpenLockLearnerAgent, self).setup_trial(
            scenario_name, action_limit, attempt_limit, specified_trial, self.multiproc
        )
        # comment to allow agent to execute the same trial multiple times
        assert (
            trial_selected not in self.trial_order
        ), "Trial selected has already been explored"
        self.rewards[trial_selected] = []
        self.num_chains_with_belief_above_threshold_per_attempt[trial_selected] = []
        self.attempt_count_per_trial[trial_selected] = 0
        self.num_attempts_between_solutions[trial_selected] = []
        self.information_gains_per_attempt[trial_selected] = []
        self.belief_thresholds_per_attempt[trial_selected] = []
        self.intervention_chain_idxs_per_attempt[trial_selected] = []
        self.interventions_per_attempt[trial_selected] = []

        # reset chain beliefs to be uniform, regardless of what occurred last trial
        self.causal_chain_space.bottom_up_belief_space.set_uniform_belief()
        self.causal_chain_space.top_down_belief_space.set_uniform_belief()

        # initialize abstract schema space
        n_solutions = self.env.get_num_solutions()
        self.initialize_abstract_schema_space(self.atomic_schema_space, n_solutions)

        # shuffle causal chains to enforce randomness
        if not self.deterministic:
            self.causal_chain_space.structure_space.shuffle()

        true_chains = generate_solutions_by_trial_causal_relation(
            scenario_name, trial_selected
        )
        self.causal_chain_space.structure_space.set_true_causal_chains(
            true_chains, self.causal_chain_space.bottom_up_belief_space
        )

        # define goal and setup model-based agent
        goal = [("door", ENTITY_STATES["DOOR_OPENED"])]
        self.model_based_agent = ModelBasedRLAgent(
            len(true_chains), goal, lambda_multiplier=self.lambda_multiplier
        )

        self.causal_chain_space.bottom_up_belief_space.attribute_space.initialize_local_attributes(
            trial_selected, use_scaled_prior=True
        )

        self.env.reset()
        # prune chains based on initial observation
        # TODO(mjedmonds): refactor where this is located/where this happens. ugly to pass around this set
        chain_idxs_pruned_from_initial_observation = self.causal_learner.chain_pruner.prune_chains_from_initial_observation(
            self.causal_chain_space,
            self.env,
            self.trial_count,
            self.attempt_count,
            multiproc=self.multiproc,
            using_ids=self.causal_chain_space.structure_space.using_ids,
        )
        # set uniform belief for chains that survived initial pruning
        self.causal_chain_space.bottom_up_belief_space.set_uniform_belief_for_ele_with_belief_above_threshold(
            self.causal_chain_space.bottom_up_belief_space.belief_threshold,
            multiproc=self.multiproc,
        )
        assert (
            self.verify_true_causal_idxs_have_belief_above_threshold()
        ), "True graphs do not have belief above threshold"

        self.num_chains_with_belief_above_threshold_per_attempt[trial_selected].append(
            self.causal_chain_space.bottom_up_belief_space.num_idxs_with_belief_above_threshold
        )

        # TODO(mjedmonds): initial chain beliefs based on attribute beliefs
        # initialize beliefs based on the attribute beliefs, p(G|T)
        self.causal_chain_space.update_bottom_up_beliefs(
            self.env.attribute_order, trial_selected, multiproc=self.multiproc
        )

        return trial_selected, chain_idxs_pruned_from_initial_observation

    def run_trial_openlock_learner(
        self,
        trial_selected,
        max_steps_with_no_pruning,
        interventions_predefined=None,
        use_random_intervention=False,
        chain_idxs_pruned_from_initial_observation=None,
        intervention_mode=None,
    ):
        if intervention_mode == "attempt":
            self.run_trial_openlock_learner_attempt_intervention(
                trial_selected,
                max_steps_with_no_pruning,
                interventions_predefined=interventions_predefined,
                chain_idxs_pruned_from_initial_observation=chain_idxs_pruned_from_initial_observation,
            )
        elif intervention_mode == "action":
            self.run_trial_openlock_learner_action_intervention(
                trial_selected,
                max_steps_with_no_pruning,
                chain_idxs_pruned_from_initial_observation=chain_idxs_pruned_from_initial_observation,
            )
        else:
            raise ValueError(
                "Unexpected intervention mode. Valid modes are 'action' or 'attempt'"
            )

    def run_trial_openlock_learner_action_intervention(
        self,
        trial_selected,
        max_steps_with_no_pruning,
        chain_idxs_pruned_from_initial_observation=None,
    ):
        self.trial_order.append(trial_selected)

        num_steps_since_last_pruning = 0
        trial_finished = False
        completed_solution_idxs = []
        chain_idxs_pruned_this_trial = (
            chain_idxs_pruned_from_initial_observation
            if chain_idxs_pruned_from_initial_observation is not None
            else set()
        )

        while not trial_finished:
            start_time = time.time()
            self.env.reset()

            timestep = 0
            attempt_reward = 0
            intervention_info_gain = 0
            num_chains_pruned_this_attempt = 0

            causal_observations = []
            action_sequence: List[int] = list()
            change_observed: List[bool] = list()
            action_beliefs_this_attempt = []

            sequences_to_prune: List[Tuple[List[Action], List[bool]]] = list()

            while not self.env.determine_attempt_finished():
                (
                    chain_idxs_with_positive_belief,
                    bottom_up_chain_idxs_with_positive_belief,
                    _,
                ) = self.get_causal_chain_idxs_with_positive_belief()

                e = np.random.sample()
                # if we provide a list of epsilons from human data, put it here
                if self.epsilon_ratios is not None:
                    epsilon = self.epsilon_ratios[self.trial_count - 1]
                else:
                    epsilon = self.epsilon
                # random policy
                if self.epsilon_active and e < epsilon:
                    action = self.model_based_agent.random_action_policy()
                # greedy policy
                else:
                    (
                        action,
                        action_beliefs,
                    ) = self.model_based_agent.greedy_action_policy(
                        causal_chain_space=self.causal_chain_space,
                        causal_chain_idxs=chain_idxs_with_positive_belief,
                        timestep=timestep,
                        action_sequence=action_sequence,
                        change_observed=change_observed,
                        interventions_executed=self.interventions_per_attempt[
                            self.current_trial_name
                        ],
                        first_agent_trial=self.determine_first_trial(),
                        ablation=self.ablation,
                    )
                    # if action_beliefs is an empty dict, we picked a random action
                    action_beliefs_this_attempt.append(action_beliefs)

                action_sequence.append(action)

                print_message(
                    self.trial_count,
                    self.attempt_count,
                    "Intervention selection took {:0.6f}s and selected intervention: {}".format(
                        time.time() - start_time, action
                    ),
                    self.print_messages,
                )

                # execute action
                reward, state_prev, state_cur = self.execute_action_intervention(action)
                # TODO(joschnei): This is bugged, seems to be essentially random when its true or false.
                # It definitely can be both.
                logging.debug(f"state_cur={state_cur}, state_prev={state_prev}")
                change_observed.append(state_cur != state_prev)
                logging.debug(f"change_observed={change_observed[-1]}")
                if not change_observed[-1]:
                    sequences_to_prune.append(
                        (list(action_sequence), list(change_observed))
                    )

                self.update_bottom_up_attribute_beliefs(
                    action, trial_selected, timestep
                )
                timestep += 1
                attempt_reward += reward

                # update causal models
                (
                    map_chains,
                    num_chains_pruned_this_action,
                    _,
                    chain_idxs_pruned,
                ) = self.causal_learner.update_bottom_up_causal_model(
                    env=self.env,
                    causal_chain_space=self.causal_chain_space,
                    causal_chain_idxs=bottom_up_chain_idxs_with_positive_belief,
                    sequences_to_prune=sequences_to_prune,
                    trial_name=trial_selected,
                    trial_count=self.trial_count,
                    attempt_count=self.attempt_count,
                    prune_inconsitent_chains=not self.ablation.PRUNING,
                    multiproc=self.multiproc,
                )
                num_chains_pruned_this_attempt += num_chains_pruned_this_action

                chain_idxs_pruned_this_trial.update(chain_idxs_pruned)
                assert (
                    self.verify_true_causal_idxs_have_belief_above_threshold()
                ), "True causal chain idx had belief drop below 0!"

                # update the top-down model based on which chains were pruned
                if len(self.instantiated_schema_space.structure_space) > 0:
                    self.instantiated_schema_space.update_instantiated_schema_beliefs(
                        chain_idxs_pruned, multiproc=self.multiproc
                    )
                    self.causal_chain_space.update_top_down_beliefs(
                        self.instantiated_schema_space, multiproc=self.multiproc
                    )
                    assert self.instantiated_schema_space.structure_space.verify_chain_assignment_in_schemas(
                        self.causal_chain_space.structure_space.true_chain_idxs
                    ), "True chains not in instantiated schemas!"

            # convert to tuple for hashability
            action_sequence = tuple(action_sequence)

            # decay epsilon
            if self.epsilon_active:
                self.epsilon = self.epsilon * self.epsilon_decay

            self.interventions_per_attempt[self.current_trial_name].append(
                action_sequence
            )
            self.information_gains_per_attempt[self.current_trial_name].append(
                float(intervention_info_gain)
            )

            prev_num_solutions_remaining = self.env.get_num_solutions_remaining()

            # finish the attempt in the environment
            self.finish_attempt()

            num_solutions_remaining = self.env.get_num_solutions_remaining()
            # solution found, instantiate schemas
            if prev_num_solutions_remaining != num_solutions_remaining:
                self.instantiate_schemas(
                    num_solutions_in_trial=self.env.get_num_solutions(),
                    solution_action_sequence=action_sequence,
                    solution_change_observed=change_observed,
                    completed_solution_idxs=completed_solution_idxs,
                    excluded_chain_idxs=chain_idxs_pruned_this_trial,
                    num_solutions_remaining=num_solutions_remaining,
                    multiproc=self.multiproc,
                )

            self.print_attempt_update(
                action_sequence,
                attempt_reward,
                num_chains_pruned_this_attempt,
                model_based_solution_chain_idxs=[],
                map_chains=map_chains,
            )

            self.finish_attempt_openlock_agent(
                trial_selected, prev_num_solutions_remaining, attempt_reward,
            )

            # TODO(mjedmonds): refactor to find better way to get trial success (and when this value will be available/set in the trial)
            trial_finished = self.env.get_trial_success()

            if num_chains_pruned_this_attempt == 0:
                num_steps_since_last_pruning += 1
            else:
                num_steps_since_last_pruning = 0

            if (
                num_steps_since_last_pruning > max_steps_with_no_pruning
                and not self.ablation.PRUNING
            ):
                print_message(
                    self.trial_count,
                    self.attempt_count,
                    "Exiting trial due to no chains pruned in {} steps".format(
                        num_steps_since_last_pruning
                    ),
                    self.print_messages,
                )
                trial_finished = True

        self.finish_trial(trial_selected, test_trial=False)

    def run_trial_openlock_learner_attempt_intervention(
        self,
        trial_selected,
        max_steps_with_no_pruning,
        interventions_predefined=None,
        use_random_intervention=False,
        chain_idxs_pruned_from_initial_observation=None,
    ):
        self.trial_order.append(trial_selected)

        num_steps_since_last_pruning = 0
        trial_finished = False
        completed_solution_idxs = []
        chain_idxs_pruned_across_all_attempts = (
            chain_idxs_pruned_from_initial_observation
            if chain_idxs_pruned_from_initial_observation is not None
            else set()
        )
        intervention_idxs_executed_this_trial = set()
        # loop for attempts in this trial
        while not trial_finished:
            start_time = time.time()
            self.env.reset()
            causal_change_idx = 0

            causal_chain_idxs_with_positive_belief = (
                self.get_causal_chain_idxs_with_positive_belief()
            )

            e = np.random.sample()
            # if we provide a list of epsilons from human data, put it here
            if self.epsilon_ratios is not None:
                epsilon = self.epsilon_ratios[self.trial_count - 1]
            else:
                epsilon = self.epsilon
            # random policy
            if self.epsilon_active and e < epsilon:
                intervention_chain_idx = self.model_based_agent.random_chain_policy(
                    causal_chain_idxs_with_positive_belief
                )
            # greedy policy
            else:
                intervention_chain_idx = self.model_based_agent.greedy_chain_policy(
                    causal_chain_space=self.causal_chain_space,
                    causal_chain_idxs=causal_chain_idxs_with_positive_belief,
                    intervention_idxs_executed=self.intervention_chain_idxs_per_attempt[
                        self.current_trial_name
                    ],
                    interventions_executed=self.interventions_per_attempt[
                        self.current_trial_name
                    ],
                    first_agent_trial=self.determine_first_trial(),
                    ablation=self.ablation,
                )
                assert (
                    intervention_chain_idx not in intervention_idxs_executed_this_trial
                ), "Intervention index already selected, should never select the same index twice"

            # decay epsilon
            if self.epsilon_active:
                self.epsilon = self.epsilon * self.epsilon_decay

            intervention_idxs_executed_this_trial.add(intervention_chain_idx)
            intervention = self.causal_chain_space.structure_space.get_actions(
                intervention_chain_idx
            )
            intervention_info_gain = 0

            # terminating codition; we have exhaustively explored remaining causal chain space
            if intervention is None:
                self.print_complete_exploration_message(
                    causal_chain_idxs_with_positive_belief
                )
                break

            self.intervention_chain_idxs_per_attempt[self.current_trial_name].append(
                intervention_chain_idx
            )
            self.interventions_per_attempt[self.current_trial_name].append(intervention)
            self.information_gains_per_attempt[self.current_trial_name].append(
                float(intervention_info_gain)
            )

            if DEBUGGING:
                print_message(
                    self.trial_count,
                    self.attempt_count,
                    "Optimal intervention is {} with info gain {:0.4f}. Took {:0.6f} seconds".format(
                        intervention, intervention_info_gain, time.time() - start_time
                    ),
                    self.print_messages,
                )

            print_message(
                self.trial_count,
                self.attempt_count,
                "Intervention selection took {:0.6f}s and selected intervention: {}".format(
                    time.time() - start_time, intervention
                ),
                self.print_messages,
            )

            prev_num_solutions_remaining = self.env.get_num_solutions_remaining()

            (
                intervention_outcomes,
                intervention_reward,
                causal_change_idx,
            ) = self.execute_attempt_intervention(
                intervention, trial_selected, causal_change_idx
            )
            causal_observations = self.causal_learner.create_causal_observations(
                env=self.env,
                action_sequence=intervention,
                intervention_outcomes=intervention_outcomes,
                trial_count=self.trial_count,
                attempt_count=self.attempt_count,
            )
            attempt_reward = intervention_reward
            # finish the attempt in the environment
            self.finish_attempt()

            if self.print_messages:
                logging.info("CAUSAL OBSERVATION:")
                self.causal_chain_space.structure_space.pretty_print_causal_observations(
                    causal_observations, print_messages=self.print_messages
                )

            num_solutions_remaining = self.env.get_num_solutions_remaining()
            # solution found, instantiate schemas
            if prev_num_solutions_remaining != num_solutions_remaining:
                completed_solution_idxs = self.process_solution(
                    causal_observations=causal_observations,
                    completed_solution_idxs=completed_solution_idxs,
                    solution_action_sequence=intervention,
                )
                self.instantiate_schemas(
                    num_solutions_in_trial=self.env.get_num_solutions(),
                    completed_solutions=completed_solution_idxs,
                    exclude_chain_idxs=chain_idxs_pruned_across_all_attempts,
                    num_solutions_remaining=num_solutions_remaining,
                    multiproc=self.multiproc,
                )

            # update beliefs, uses self's log to get executed interventions/outcomes among ALL chains with with positive belief (even those below threshold)
            (
                map_chains,
                num_chains_pruned_this_attempt,
                chain_idxs_consistent,
                chain_idxs_pruned,
            ) = self.causal_learner.update_bottom_up_causal_model(
                env=self.env,
                causal_chain_space=self.causal_chain_space,
                causal_chain_idxs=causal_chain_idxs_with_positive_belief,
                causal_observations=causal_observations,
                trial_name=trial_selected,
                trial_count=self.trial_count,
                attempt_count=self.attempt_count,
                prune_inconsitent_chains=not self.ablation.PRUNING,
                multiproc=self.multiproc,
            )
            chain_idxs_pruned_across_all_attempts.update(chain_idxs_pruned)

            assert (
                self.verify_true_causal_idxs_have_belief_above_threshold()
            ), "True causal chain idx had belief drop below 0!"

            # update the top-down model based on which chains were pruned
            if len(self.instantiated_schema_space.structure_space) > 0:
                self.instantiated_schema_space.update_instantiated_schema_beliefs(
                    chain_idxs_pruned, multiproc=self.multiproc
                )
                self.causal_chain_space.update_top_down_beliefs(
                    self.instantiated_schema_space, multiproc=self.multiproc
                )
                assert self.instantiated_schema_space.structure_space.verify_chain_assignment_in_schemas(
                    self.causal_chain_space.structure_space.true_chain_idxs
                ), "True chains not in instantiated schemas!"

            self.print_attempt_update(
                intervention,
                attempt_reward,
                num_chains_pruned_this_attempt,
                model_based_solution_chain_idxs=[],
                map_chains=map_chains,
            )

            self.finish_attempt_openlock_agent(
                trial_selected, prev_num_solutions_remaining, attempt_reward,
            )

            # TODO(mjedmonds): refactor to find better way to get trial success (and when this value will be available/set in the trial)
            trial_finished = self.env.get_trial_success()

            if num_chains_pruned_this_attempt == 0:
                num_steps_since_last_pruning += 1
            else:
                num_steps_since_last_pruning = 0

            if (
                num_steps_since_last_pruning > max_steps_with_no_pruning
                and not self.ablation.PRUNING
            ):
                print_message(
                    self.trial_count,
                    self.attempt_count,
                    "Exiting trial due to no chains pruned in {} steps".format(
                        num_steps_since_last_pruning
                    ),
                    self.print_messages,
                )
                trial_finished = True

        self.finish_trial(trial_selected, test_trial=False)

    def execute_attempt_intervention(
        self, intervention, trial_selected, causal_change_index
    ):
        intervention_reward = 0
        intervention_outcomes = []
        # execute the intervention in the simulator
        for action in intervention:
            if isinstance(action, str):
                action = Action(action)

            reward, state_prev, state_cur = self.execute_action_intervention(action)
            causal_change_index = self.update_bottom_up_attribute_beliefs(
                action, trial_selected, causal_change_index
            )

            intervention_reward += reward
            intervention_outcomes.append((state_prev, state_cur))

        return intervention_outcomes, intervention_reward, causal_change_index

    def execute_action_intervention(self, action: Union[str, Action]):
        if isinstance(action, str):
            action_env = self.env.action_map[action]
        elif isinstance(action, Action):
            action_env = self.env.action_map[action.name + "_" + action.obj]
        else:
            raise TypeError(f"Unexpected action type: {type(action)}, {action}")

        print_message(
            self.trial_count,
            self.attempt_count,
            "Executing action: {}".format(action),
            self.print_messages,
        )
        # TODO(joschnei): We should be able to use next_state instead of the outcome stuff below,
        # except I have no idea what the outcome stuff is doing.
        next_state, reward, _, _ = self.env.step(action_env)

        attempt_results = self.get_last_results()

        state_prev = Outcome.parse_results_into_outcome(attempt_results, idx=-2)
        state_cur = Outcome.parse_results_into_outcome(attempt_results, idx=-1)

        return reward, state_prev, state_cur

    def update_bottom_up_attribute_beliefs(
        self, action, trial_selected, timestep: int
    ) -> None:
        # if env changed state, accept this observation into attribute model and update attribute model
        if self.env.determine_fluent_change():
            obj = action.obj
            attributes = self.env.get_obj_attributes(obj)
            if self.causal_chain_space.bottom_up_belief_space.attribute_space.using_ids:
                for attribute in attributes:
                    attributes[
                        attribute
                    ] = self.causal_chain_space.structure_space.unique_id_manager.convert_attribute_to_target_type(
                        attribute, attributes[attribute], target_type="int"
                    )
            attributes_to_add = [(name, value) for name, value in attributes.items()]
            # TODO(mjedmonds): hacky way to add action
            attributes_to_add.append(("action", action.name))
            self.causal_chain_space.bottom_up_belief_space.attribute_space.add_frequencies(
                attributes_to_add,
                trial_selected,
                timestep,
                global_alpha_increase=self.global_alpha_update,
                local_alpha_increase=self.local_alpha_update,
            )

    def update_instantiated_schema_beliefs(self, chain_idxs_pruned):
        self.instantiated_schema_space.update_instantiated_schema_beliefs(
            chain_idxs_pruned, multiproc=self.multiproc
        )
        self.causal_chain_space.update_top_down_beliefs(
            self.instantiated_schema_space, multiproc=self.multiproc
        )
        assert self.instantiated_schema_space.structure_space.verify_chain_assignment_in_schemas(
            self.causal_chain_space.structure_space.true_chain_idxs
        ), "True chains not in instantiated schemas!"

    def initialize_abstract_schema_beliefs(self, n_solutions):
        if n_solutions == 2:
            self.two_solution_abstract_schema_space.initialize_abstract_schema_beliefs(
                self.atomic_schema_space
            )
        elif n_solutions == 3:
            self.three_solution_abstract_schema_space.initialize_abstract_schema_beliefs(
                self.atomic_schema_space
            )
        else:
            raise ValueError(
                "Incorrect number of solutions found than accounted for by schemas. Cannot update schema belief"
            )

    def update_atomic_schema_beliefs(self, completed_solutions):
        """
        Update the belief in schemas based on the completed solutions
        :param completed_solutions: list of completed solutions executed
        :return: Nothing
        """
        # truncate pushing on the door from solutions
        truncated_solutions = [
            completed_solution[: len(completed_solution) - 1]
            for completed_solution in completed_solutions
        ]
        self.atomic_schema_space.update_atomic_schema_beliefs(truncated_solutions)

    def process_solution(
        self,
        completed_solution_idxs: Sequence[int],
        solution_action_sequence: Sequence[Action],
        solution_change_observed: Sequence[bool],
    ):
        solution_chains = self.causal_chain_space.structure_space.find_causal_chain_idxs_with_actions(
            solution_action_sequence, solution_change_observed
        )
        # I'm just going to assume that you find the correct action sequence. In reality this
        # requires testing one additional chain where you specifically pick the next action that
        # caused a chain from this sequence until it works to identify exactly what the delays are,
        # but I'm fine not modelling that explicitly.
        # For example, let's say you take actions 1 2 3 4 5 in order, and then actions 1 3 and 5
        # caused a change, where actions 2 and 4 didn't. You can disambiguate the chain by taking
        # action 1, then taking action 3 until it works, then taking action 5 until it works. The
        # first time it works, we know exactly what the delay is.
        solution_chain = set(solution_chains).intersection(
            self.causal_chain_space.structure_space.true_chains
        )[0]
        true_chain_idx = self.causal_chain_space.structure_space.true_chains.index(
            solution_chain
        )
        solution_causal_chain_idx = self.causal_chain_space.structure_space.true_chain_idxs[
            true_chain_idx
        ]
        completed_solution_idxs.append(solution_causal_chain_idx)
        # update the state of the model-based planner
        self.model_based_agent.update_state(
            self.env.get_num_solutions_remaining(), solution_action_sequence
        )
        return completed_solution_idxs

    def instantiate_schemas(
        self,
        num_solutions_in_trial,
        completed_solution_idxs,
        solution_action_sequence,
        solution_change_observed,
        excluded_chain_idxs,
        num_solutions_remaining,
        multiproc=True,
    ):
        if num_solutions_remaining <= 0:
            return
        completed_solution_idxs = self.process_solution(
            completed_solution_idxs=completed_solution_idxs,
            solution_action_sequence=solution_action_sequence,
            solution_change_observed=solution_change_observed,
        )

        # TODO(mjedmonds): make every else multiproc compliant - currently there are bugs if other functions are multiprocessed
        (
            instantiated_schemas,
            instantiated_schema_beliefs,
        ) = self.current_abstract_schema_space.instantiate_schemas(
            solutions_executed=completed_solution_idxs,
            n_chains_in_schema=num_solutions_in_trial,
            causal_chain_structure_space=self.causal_chain_space.structure_space,
            excluded_chain_idxs=excluded_chain_idxs,
            multiproc=True,
        )

        self.instantiated_schema_space.structure_space = instantiated_schemas
        self.instantiated_schema_space.belief_space.beliefs = (
            instantiated_schema_beliefs
        )

        self.instantiated_schema_space.belief_space.renormalize_beliefs(
            multiproc=self.multiproc
        )

        # update the top-down beliefs
        self.causal_chain_space.update_top_down_beliefs(
            self.instantiated_schema_space, multiproc=multiproc
        )

        # verify true assignment is in schema space
        assert self.instantiated_schema_space.structure_space.verify_chain_assignment_in_schemas(
            self.causal_chain_space.structure_space.true_chain_idxs
        ), "True chains not in instantiated schemas!"

    def initialize_abstract_schema_space(self, atomic_schema_space, n_solutions):
        if n_solutions == 2:
            self.current_abstract_schema_space = self.two_solution_abstract_schema_space
        elif n_solutions == 3:
            self.current_abstract_schema_space = (
                self.three_solution_abstract_schema_space
            )
        else:
            raise ValueError("Incorrect number of solutions")

        self.current_abstract_schema_space.update_abstract_schema_beliefs(
            atomic_schema_space
        )

    def update_trial(self, trial, is_current_trial):
        if is_current_trial:
            self.logger.cur_trial = trial
        else:
            self.logger.cur_trial = None
            self.logger.trial_seq.append(trial)

    def get_causal_chain_idxs_with_positive_belief(self):
        # get indices with positive belief from top-down and bottom-up
        bottom_up_causal_chain_idxs_with_positive_belief = self.causal_chain_space.bottom_up_belief_space.get_idxs_with_belief_above_threshold(
            print_msg=self.print_messages
        )
        top_down_causal_chain_idxs_with_positive_belief = self.causal_chain_space.top_down_belief_space.get_idxs_with_belief_above_threshold(
            print_msg=self.print_messages
        )
        # actual candidate set is the intersection between top down and bottom up
        causal_chain_idxs_with_positive_belief = list(
            set(bottom_up_causal_chain_idxs_with_positive_belief).intersection(
                set(top_down_causal_chain_idxs_with_positive_belief)
            )
        )

        assert (
            causal_chain_idxs_with_positive_belief
        ), "No causal chains with positive belief in both top-down and bottom-up belief spaces"
        return (
            causal_chain_idxs_with_positive_belief,
            bottom_up_causal_chain_idxs_with_positive_belief,
            top_down_causal_chain_idxs_with_positive_belief,
        )

    def get_true_interventions_and_outcomes(self):
        # Optimal interventions and outcomes
        true_optimal_interventions = [
            solution.actions
            for solution in self.causal_chain_space.structure_space.true_chains
        ]
        true_optimal_outcomes = [
            Outcome(
                solution.states,
                self.intervention_selector.simulate_intervention(
                    solution, solution.actions
                ),
            )
            for solution in self.causal_chain_space.structure_space.true_chains
        ]

        return true_optimal_interventions, true_optimal_outcomes

    def finish_attempt(self):
        """
        finish the attempt in the environment. To be run before the agent updates its internal model
        :return:
        """
        # TODO(mjedmonds): see if this function should be consolidated with finish_attempt_openlock_agent()
        super(OpenLockLearnerAgent, self).finish_attempt()

    def finish_attempt_openlock_agent(
        self, trial_name, prev_num_solutions_remaining, attempt_reward
    ):
        """
        Finishes the attempt for the openlock agent. To be run after the agent has updated its internal model
        :param trial_name:
        :param prev_num_solutions_remaining:
        :param attempt_reward:
        :return:
        """
        num_solutions_remaining = self.env.get_num_solutions_remaining()
        self.rewards[self.current_trial_name].append(attempt_reward)
        self.belief_thresholds_per_attempt[self.current_trial_name].append(
            self.causal_chain_space.bottom_up_belief_space.belief_threshold
        )
        self.num_chains_with_belief_above_threshold_per_attempt[
            self.current_trial_name
        ].append(
            self.causal_chain_space.bottom_up_belief_space.num_idxs_with_belief_above_threshold
        )
        # solution found, add to number of attempts between solutions
        if num_solutions_remaining != prev_num_solutions_remaining:
            num_attempts_since_last_solution = (
                self.env.cur_trial.num_attempts_since_last_solution_found
            )
            self.num_attempts_between_solutions[self.current_trial_name].append(
                num_attempts_since_last_solution
            )

        self.observed_causal_observations = []
        self.attempt_count += 1

    def finish_trial(self, trial_selected, test_trial):
        super(OpenLockLearnerAgent, self).finish_trial(trial_selected, test_trial)

        self.update_atomic_schema_beliefs(self.env.cur_trial.completed_solutions)

        if DEBUGGING:
            self.causal_chain_space.bottom_up_belief_space.attribute_space.pretty_print_global_attributes()
            self.causal_chain_space.bottom_up_belief_space.attribute_space.pretty_print_local_attributes(
                self.current_trial_name
            )
            self.current_abstract_schema_space.structure_space.pretty_print(
                self.current_abstract_schema_space.belief_space
            )
            self.atomic_schema_space.structure_space.pretty_print(
                self.atomic_schema_space.belief_space
            )

        # reset instantiated schema space
        self.instantiated_schema_space.structure_space.reset()
        self.instantiated_schema_space.belief_space.reset()

        # minus 1 because we incremented the attempt count
        self.attempt_count_per_trial[self.current_trial_name] = self.attempt_count - 1
        self.attempt_count = 1
        self.trial_count += 1

    def finish_subject(
        self, strategy="OpenLockLearner", transfer_strategy="OpenLockLearner"
    ):
        """
        Prepare agent to save to JSON. Any data to be read in matlab should be converted to native python lists (instead of numpy) before running jsonpickle
        :param strategy:
        :param transfer_strategy:
        :return:
        """
        agent_cpy = copy.copy(self)

        # we need to deep copy the causal chain space so causal_chains is not deleted (and is usable for other agents)
        causal_chain_space_structure = copy.deepcopy(
            agent_cpy.causal_chain_space.structure_space
        )
        agent_cpy.causal_chain_space.structure_space = causal_chain_space_structure

        # TODO(mjedmonds): this is used to save numpy arrays into json pickle...this should as numpy arrays, but it does not
        agent_cpy.causal_chain_space.bottom_up_belief_space.attribute_space.convert_to_list()
        # cleanup agent for writing to file (deleting causal chain structures and their beliefs)
        # keep bottom_up_belief_space for attributes
        attributes_to_delete = ["structure_space", "top_down_belief_space"]
        for attribute_to_delete in attributes_to_delete:
            if hasattr(agent_cpy.causal_chain_space, attribute_to_delete):
                delattr(agent_cpy.causal_chain_space, attribute_to_delete)
        attributes_to_delete = [
            "outcome_space",
            "cached_outcome_likelihood_sum",
            "cached_outcome_likelihoods_given_intervention_and_chain_times_belief",
        ]
        for attribute_to_delete in attributes_to_delete:
            if hasattr(agent_cpy, attribute_to_delete):
                delattr(agent_cpy, attribute_to_delete)
        # keep bottom_up_belief_space for attribute space
        attributes_to_delete = ["beliefs", "num_idxs_with_belief_above_threshold"]
        for attribute_to_delete in attributes_to_delete:
            if hasattr(
                agent_cpy.causal_chain_space.bottom_up_belief_space, attribute_to_delete
            ):
                delattr(
                    agent_cpy.causal_chain_space.bottom_up_belief_space,
                    attribute_to_delete,
                )

        if hasattr(agent_cpy, "ablation"):
            agent_cpy.ablation = agent_cpy.ablation.__dict__
        if hasattr(agent_cpy, "params") and "ablation" in agent_cpy.params.keys():
            agent_cpy.params["ablation"] = agent_cpy.params["ablation"].__dict__

        super(OpenLockLearnerAgent, self).finish_subject(
            strategy, transfer_strategy, agent_cpy
        )
        # manually mark that we have finished this agent so cleanup() is not called
        # (finished is marked to false on every call to setup_trial
        self.finished = True

    @property
    def current_trial_name(self):
        if self.env is not None and self.env.cur_trial is not None:
            return self.env.cur_trial.name
        else:
            return None

    def convert_outcome_space_to_or_from_ids(self, target_type):
        # convert outcome and intervention spaces to IDs if causal chain space is defined on IDs
        # if causal chain states are defined on ints but outcome state is defined on string, we need to convert IDs
        for i in range(len(self.outcome_space)):
            self.outcome_space[
                i
            ].state_ids = self.causal_chain_space.structure_space.unique_id_manager.convert_states_to_target_type(
                self.outcome_space[i].state_ids, target_type
            )
        if target_type == "int":
            self.outcome_space.using_ids = True
        else:
            self.outcome_space.using_ids = False

    def convert_attribute_space_to_or_from_ids(self, target_type):
        self.causal_chain_space.bottom_up_belief_space.attribute_space.global_attributes.labels[
            "color"
        ] = list(
            self.causal_chain_space.structure_space.unique_id_manager.convert_attribute_tuple_to_target_type(
                self.causal_chain_space.bottom_up_belief_space.attribute_space.global_attributes.labels[
                    "color"
                ],
                target_type,
            )
        )
        self.causal_chain_space.bottom_up_belief_space.attribute_space.global_attributes.labels[
            "position"
        ] = list(
            self.causal_chain_space.structure_space.unique_id_manager.convert_states_to_target_type(
                self.causal_chain_space.bottom_up_belief_space.attribute_space.global_attributes.labels[
                    "position"
                ],
                target_type,
            )
        )
        for (
            key
        ) in (
            self.causal_chain_space.bottom_up_belief_space.attribute_space.local_attributes.keys()
        ):
            local_attributes = self.causal_chain_space.bottom_up_belief_space.attribute_space.local_attributes[
                key
            ]
            local_attributes["color"] = list(
                self.causal_chain_space.structure_space.unique_id_manager.convert_attribute_tuple_to_target_type(
                    local_attributes["color"], target_type
                )
            )
            local_attributes["position"] = list(
                self.causal_chain_space.structure_space.unique_id_manager.convert_states_to_target_type(
                    local_attributes["position"], target_type
                )
            )
        if target_type == "int":
            self.causal_chain_space.bottom_up_belief_space.attribute_space.using_ids = (
                True
            )
        else:
            self.causal_chain_space.bottom_up_belief_space.attribute_space.using_ids = (
                False
            )

    def verify_true_causal_idxs_have_belief_above_threshold(self):
        beliefs = [
            self.causal_chain_space.bottom_up_belief_space[x]
            for x in self.causal_chain_space.structure_space.true_chain_idxs
        ]
        threshold = self.causal_chain_space.bottom_up_belief_space.belief_threshold
        logging.debug(f"Belief threshold={threshold}, belief={beliefs}")
        return all([belief > threshold] for belief in beliefs)

    def attempt_sanity_checks(
        self,
        action_sequences_that_should_be_pruned=None,
        intervention_chain_idxs=None,
        intervention_idxs_executed_this_trial=None,
        chain_idxs_pruned_this_trial=None,
    ):
        assert (
            self.verify_true_causal_idxs_have_belief_above_threshold()
        ), "True causal chain idx had belief drop below 0!"

        if DEBUGGING and chain_idxs_pruned_this_trial:
            assert all(
                [
                    self.causal_chain_space.bottom_up_belief_space[pruned_idx] == 0
                    for pruned_idx in chain_idxs_pruned_this_trial
                ]
            ), "Should have pruned chain that has positive belief"

        # verify all action sequences that should be pruned have 0 belief
        if DEBUGGING and action_sequences_that_should_be_pruned:
            # verify that all chain idxs that should be pruned are actually pruned
            assert all(
                [
                    self.causal_chain_space.bottom_up_belief_space[pruned_idx] == 0
                    for pruned_seq in action_sequences_that_should_be_pruned
                    for pruned_idx in self.causal_chain_space.structure_space.find_causal_chain_idxs_with_actions(
                        pruned_seq
                    )
                ]
            ), "action sequence that should be pruned is not!"

        # verify we don't execute the same intervention twice
        if intervention_chain_idxs and intervention_idxs_executed_this_trial:
            try:
                assert not intervention_chain_idxs.intersection(
                    intervention_idxs_executed_this_trial
                ), "Executing same intervention twice"
            except AssertionError:
                logging.error("problem")
                raise AssertionError("Executing the same intervention twice")

    def determine_first_trial(self):
        return True if self.trial_count == 1 else False

    def pretty_print_policy(self, trial_name, use_global_Q=False):
        if use_global_Q:
            policy_type = "GLOBAL"
        else:
            policy_type = "LOCAL"
        chain_idxs, chain_q_values = self.get_greedy_policy(trial_name, use_global_Q)
        print_message(
            self.trial_count,
            self.attempt_count,
            "GREEDY {} POLICY TO FIND BOTH SOLUTIONS:".format(policy_type),
            self.print_messages,
        )
        self.causal_chain_space.structure_space.pretty_print_causal_chain_idxs(
            chain_idxs, q_values=chain_q_values, print_messages=self.print_messages
        )

    def print_attempt_update(
        self,
        intervention_str,
        attempt_reward,
        num_chains_pruned_this_attempt,
        model_based_solution_chain_idxs,
        map_chains,
    ):
        if DEBUGGING:
            logging.debug(self.env.cur_trial.attempt_seq[-1].action_seq)

        self.plot_reward(attempt_reward, self.total_attempt_count)
        print_message(
            self.trial_count,
            self.attempt_count,
            "{} chains pruned this attempt".format(num_chains_pruned_this_attempt),
            self.print_messages,
        )

        if 0 < len(model_based_solution_chain_idxs) < 20:
            if DEBUGGING:
                print_message(
                    self.trial_count,
                    self.attempt_count,
                    "MODEL-BASED SOLUTION CHAINS",
                    self.print_messages,
                )
                self.causal_chain_space.structure_space.pretty_print_causal_chain_idxs(
                    model_based_solution_chain_idxs,
                    self.causal_chain_space.bottom_up_belief_space,
                    print_messages=self.print_messages,
                )

        if 0 < len(map_chains) < 20:
            if (
                self.causal_chain_space.bottom_up_belief_space.num_idxs_with_belief_above_threshold
                < 20
            ):
                print_message(
                    self.trial_count,
                    self.attempt_count,
                    "CHAINS WITH BELIEF ABOVE {}: {}".format(
                        self.causal_chain_space.bottom_up_belief_space.belief_threshold,
                        self.causal_chain_space.bottom_up_belief_space.num_idxs_with_belief_above_threshold,
                    ),
                    self.print_messages,
                )
                self.causal_chain_space.structure_space.print_chains_above_threshold(
                    self.causal_chain_space.bottom_up_belief_space,
                    self.causal_chain_space.bottom_up_belief_space.belief_threshold,
                )
            if DEBUGGING:
                print_message(
                    self.trial_count,
                    self.attempt_count,
                    "MAP CHAINS",
                    self.print_messages,
                )
                self.causal_chain_space.structure_space.pretty_print_causal_chain_idxs(
                    map_chains,
                    self.causal_chain_space.bottom_up_belief_space,
                    print_messages=self.print_messages,
                )

        if attempt_reward > 0:
            print_message(
                self.trial_count,
                self.attempt_count,
                "Executed {} with reward of {}".format(
                    intervention_str, attempt_reward
                ),
                self.print_messages,
            )

    def print_num_attempts_per_trial(self):
        table = texttable.Texttable()
        max_num_solutions = max(
            [
                len(self.num_attempts_between_solutions[trial_name])
                for trial_name in self.trial_order
            ]
        )
        chain_content = []
        for trial_name, attempt_count in self.attempt_count_per_trial.items():
            new_chain_content = [trial_name, attempt_count]
            num_attempts_between_solutions = [
                x for x in self.num_attempts_between_solutions[trial_name]
            ]
            if len(num_attempts_between_solutions) != max_num_solutions:
                # add in values for 3-lever vs 4-lever, two additional columns for trial name and total attempt count
                num_attempts_between_solutions.append("N/A")
            new_chain_content.extend(num_attempts_between_solutions)
            chain_content.append(new_chain_content)

        headers = ["trial name", "attempt count"]
        addition_header_content = [
            "solution {}".format(i) for i in range(max_num_solutions)
        ]
        headers.extend(addition_header_content)
        alignment = ["l", "r"]
        alignment.extend(["r" for i in range(max_num_solutions)])
        table.set_cols_align(alignment)
        content = [headers]
        content.extend(chain_content)

        table.add_rows(content)

        widths = [30, 20]
        widths.extend([20 for i in range(max_num_solutions)])

        table.set_cols_width(widths)

        logging.info(table.draw())

    def print_agent_summary(self):
        for trial_name in self.trial_order:
            self.causal_chain_space.bottom_up_belief_space.attribute_space.pretty_print_local_attributes(
                trial_name
            )
        logging.info("TWO SOLUTION SCHEMA SPACE:")
        self.two_solution_abstract_schema_space.structure_space.pretty_print(
            self.two_solution_abstract_schema_space.belief_space
        )
        logging.info("THREE SOLUTION SCHEMA SPACE:")
        self.three_solution_abstract_schema_space.structure_space.pretty_print(
            self.three_solution_abstract_schema_space.belief_space
        )

        logging.info("NUMBER OF ATTEMPTS PER TRIAL:")
        self.print_num_attempts_per_trial()

    def print_complete_exploration_message(
        self, causal_chain_idxs_with_positive_belief
    ):
        print_message(
            self.trial_count,
            self.attempt_count,
            "Causal chain space completely explored. Causally plausible chains:",
            self.print_messages,
        )
        self.causal_chain_space.structure_space.pretty_print_causal_chain_idxs(
            causal_chain_idxs_with_positive_belief,
            self.causal_chain_space.bottom_up_belief_space,
        )
        print_message(
            self.trial_count,
            self.attempt_count,
            "Causal chain space completely explored with {} causally plausible chains. Exiting causal learning...".format(
                len(causal_chain_idxs_with_positive_belief)
            ),
            self.print_messages,
        )

    def plot_num_pruned(self, num_chains_pruned):
        self.plot_value(
            "Number of chains pruned (log)",
            math.log(num_chains_pruned[-1]) if num_chains_pruned[-1] > 0 else 0,
            len(num_chains_pruned),
        )

