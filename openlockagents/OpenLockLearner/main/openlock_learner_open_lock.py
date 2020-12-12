import logging
import math
import os
import time

import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from openlock.common import generate_effect_probabilities
from openlock.settings_trial import PARAMS
from openlockagents.common.agent import Agent
from openlockagents.OpenLockLearner.io.causal_structure_io import (
    load_causal_structures_from_file,
)
from openlockagents.OpenLockLearner.learner.OpenLockLearnerAgent import (
    OpenLockLearnerAgent,
)
from openlockagents.OpenLockLearner.main.generate_causal_structures import (
    generate_causal_structures,
)
from openlockagents.OpenLockLearner.util.common import (
    AblationParams,
    parse_arguments,
    setup_structure_space_paths,
)


def plot_num_pruned(num_chains_pruned, filename):
    y = [math.log(x) if x > 0 else -0 for x in num_chains_pruned]
    if len(y) < 2:
        return None
    sns.set_style("dark")
    plt.plot(y)
    plt.ylabel("Num chains pruned (log)")
    plt.xlabel("Step number")
    fig = plt.gcf()
    fig.savefig(filename)
    plt.draw()
    return fig


def main():

    global_start_time = time.time()

    args = parse_arguments()

    logging.basicConfig(
        level=args.verbosity,
        format="%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s %(message)s",
    )
    logging.info(args)

    ablation_params = AblationParams()

    if args.savedir is None:
        data_dir = "/home/joschnei/OpenLock/agent/data/OpenLockLearningResults/cc3-ce4_subjects"
    else:
        data_dir = args.savedir
    if args.scenario is None:
        param_scenario = "CC3-CE4"
    else:
        param_scenario = args.scenario
    if args.bypass_confirmation is None:
        bypass_confirmation = False
    else:
        bypass_confirmation = True
    if args.ablations is None:
        pass
    else:
        # process ablations
        for ablation in args.ablations:
            ablation = ablation.upper()
            if hasattr(ablation_params, ablation):
                setattr(ablation_params, ablation, True)
            else:
                exception_str = "Unknown ablation argument: {}".format(ablation)
                raise ValueError(exception_str)

    params = PARAMS[param_scenario]
    params["data_dir"] = data_dir
    params["train_attempt_limit"] = args.train_attempt_limit
    params["test_attempt_limit"] = args.test_attempt_limit
    # run to the full attempt limit, regardless of whether or not all solutions were found
    params["full_attempt_limit"] = False
    params["intervention_sample_size"] = 10  # doesn't matter
    params["chain_sample_size"] = 1000  # doesn't matter
    params["use_physics"] = False

    # openlock learner params
    params["lambda_multiplier"] = 1
    params["local_alpha_update"] = 1
    params["global_alpha_update"] = 1
    params["epsilon"] = 0.99
    params["epsilon_decay"] = 0.99
    params["epsilon_active"] = False
    # these params were extracted using matlab
    params["intervention_mode"] = "action"
    # setup ablations
    params["ablation_params"] = ablation_params
    params["effect_probabilities"] = generate_effect_probabilities(
        l0=1.0, l1=1.0, l2=1.0, door=1.0
    )

    params["using_ids"] = False
    params["multiproc"] = False
    params["deterministic"] = False
    params["num_agent_runs"] = args.n_replications
    params["src_dir"] = None
    params["print_messages"] = False

    logging.info(params)

    logging.info("Pre-instantiation setup")
    env = Agent.pre_instantiation_setup(params, bypass_confirmation)
    env.lever_index_mode = "position"

    (
        causal_chain_structure_space_path,
        two_solution_schemas_structure_space_path,
        three_solution_schemas_structure_space_path,
    ) = setup_structure_space_paths()

    if not os.path.exists(causal_chain_structure_space_path):
        logging.warning("No hypothesis space files found, generating hypothesis spaces")
        generate_causal_structures(max_delay=params.get("max_delay", 0))

    interventions_predefined = []

    # these are used to advance to the next trial after there have no chains pruned for num_steps_with_no_pruning_to_finish_trial steps
    num_steps_with_no_pruning_to_finish_trial = 500
    num_agent_runs = params["num_agent_runs"]

    logging.info("Loading structure and schemas")
    (
        causal_chain_structure_space,
        two_solution_schemas,
        three_solution_schemas,
    ) = load_causal_structures_from_file(
        causal_chain_structure_space_path,
        two_solution_schemas_structure_space_path,
        three_solution_schemas_structure_space_path,
    )

    logging.info("Starting trials")
    for i in range(num_agent_runs):
        logging.info(f"Starting agent run {i} of {num_agent_runs}")
        agent_start_time = time.time()

        env = Agent.make_env(params)
        env.lever_index_mode = "position"

        # setup agent
        agent = OpenLockLearnerAgent(
            env,
            causal_chain_structure_space,
            params,
            **{
                "two_solution_schemas": two_solution_schemas,
                "three_solution_schemas": three_solution_schemas,
            },
        )

        possible_trials = agent.get_random_order_of_possible_trials(
            params["train_scenario_name"]
        )

        agent.training_trial_order = possible_trials
        logging.info("Training agent")
        for trial_name in possible_trials:
            (
                trial_selected,
                chain_idxs_pruned_from_initial_observation,
            ) = agent.setup_trial(
                scenario_name=params["train_scenario_name"],
                action_limit=params["train_action_limit"],
                attempt_limit=params["train_attempt_limit"],
                specified_trial=trial_name,
            )

            agent.run_trial_openlock_learner(
                trial_selected,
                num_steps_with_no_pruning_to_finish_trial,
                interventions_predefined=interventions_predefined,
                chain_idxs_pruned_from_initial_observation=chain_idxs_pruned_from_initial_observation,
                intervention_mode=params["intervention_mode"],
            )

        # testing
        if params["test_scenario_name"] in ("CE4, CC4, CE4D, CC4D"):
            logging.info("Testing agent")
            (
                trial_selected,
                chain_idxs_pruned_from_initial_observation,
            ) = agent.setup_trial(
                scenario_name=params["test_scenario_name"],
                action_limit=params["test_action_limit"],
                attempt_limit=params["test_attempt_limit"],
            )

            agent.run_trial_openlock_learner(
                trial_selected,
                num_steps_with_no_pruning_to_finish_trial,
                interventions_predefined=interventions_predefined,
                chain_idxs_pruned_from_initial_observation=chain_idxs_pruned_from_initial_observation,
                intervention_mode=params["intervention_mode"],
            )

        agent.print_agent_summary()
        logging.info(
            "Finished agent. Total runtime: {}s".format(time.time() - agent_start_time)
        )
        agent.finish_subject("OpenLockLearner", "OpenLockLearner")

    logging.info(
        "Finished all agents for {}. Total runtime: {}s".format(
            param_scenario, time.time() - global_start_time
        )
    )
    return


if __name__ == "__main__":
    main()
