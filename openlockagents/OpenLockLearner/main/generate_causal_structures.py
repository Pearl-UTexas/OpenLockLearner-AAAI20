from openlockagents.common.agent import Agent
from openlockagents.OpenLockLearner.causal_classes.hypothesis_space import (
    generate_hypothesis_space,
)
from openlockagents.OpenLockLearner.util.common import (
    ACTIONS,
    CAUSAL_CHAIN_EDGES,
    FLUENT_STATES,
    FLUENTS,
    setup_structure_space_paths,
)


def main():
    generate_causal_structures()


def generate_causal_structures(max_delay: int = 0):
    params = dict()
    params["use_physics"] = False
    params["train_scenario_name"] = "CE3D"
    params["src_dir"] = None

    env = Agent.pre_instantiation_setup(params, bypass_confirmation=True)
    env.lever_index_mode = "position"

    attributes = [env.attribute_labels[attribute] for attribute in env.attribute_order]
    structure = CAUSAL_CHAIN_EDGES

    (
        causal_chain_structure_space_path,
        two_solution_schemas_structure_space_path,
        three_solution_schemas_structure_space_path,
    ) = setup_structure_space_paths()
    generate_hypothesis_space(
        env=env,
        structure=structure,
        causal_chain_structure_space_path=causal_chain_structure_space_path,
        two_solution_schemas_structure_space_path=two_solution_schemas_structure_space_path,
        three_solution_schemas_structure_space_path=three_solution_schemas_structure_space_path,
        attributes=attributes,
        actions=ACTIONS,
        fluents=FLUENTS,
        fluent_states=FLUENT_STATES,
        perceptually_causal_relations=None,
        max_delay=max_delay,
    )
    return


if __name__ == "__main__":
    main()
