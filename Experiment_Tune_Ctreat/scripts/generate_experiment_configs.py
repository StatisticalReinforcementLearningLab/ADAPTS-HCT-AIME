import os
import json
import shutil
from itertools import product
import argparse

DESIGN_DIR = "Experiment_Tune_Ctreat/configs/design_decisions"
SHARED_ALGOS_FILE = "Experiment_Tune_Ctreat/configs/shared_algorithms.json"
ANALYSIS_FILE = "Experiment_Tune_Ctreat/analysis.ipynb"

OUTPUT_ROOT = "Experiment_Tune_Ctreat/experiments"


def create_test_STE_config(treatment):

    # Offline value iteration is the optimal policy candidate with highest STE
    algorithms = [
        "OptPolicy_OfflineValueIteration_OnlyFirstSixFeatures_Rep1000",
        "MRT0",
    ]

    mediator_values = [0, 1, 2]
    test_ste_config = {}

    def generate_config(t, mediator, algorithms):
        """Generate JSON file with configurations for the eperiments."""
        key = f"T0_{t}_T1_{t}_T2_{t}_T3_{t}_M_{mediator}"
        config = {
            "treatments": [t, t, t, t],
            "seed": 1,
            "softmax_tem": 100,
            "n": 100,
            "rep": 63,
            "mediator": mediator,
            "algorithms": algorithms,
        }
        return key, config

    for t in treatment:
        for mediator in mediator_values:
            key, config = generate_config(t, mediator, algorithms)
            test_ste_config[key] = config

    test_ste_filename = "Tune_Ctreat.json"

    output_path = os.path.join(DESIGN_DIR, test_ste_filename)
    with open(output_path, "w") as f:
        json.dump(test_ste_config, f, indent=2)

    print(f"Test STE configuration written to {output_path}")


def generate_experiments():
    with open(SHARED_ALGOS_FILE, "r") as f:
        shared_algorithms = json.load(f)

    # Loop through all files (e.g. design_decisions/Pooling_Effect.json)
    for design_file in os.listdir(DESIGN_DIR):
        design_path = os.path.join(DESIGN_DIR, design_file)
        with open(design_path, "r") as f:
            design_config = json.load(f)

        # Convert each file to a directory with config_general_run.json and analysis.ipynb
        for baseline, params in design_config.items():
            output_dir = os.path.join(OUTPUT_ROOT, design_file.split(".")[0], baseline)
            os.makedirs(output_dir, exist_ok=True)

            if "template" in params:
                template_name = params["template"]
                template_path = os.path.join(
                    DESIGN_DIR, f"../environment_template.json"
                )
                with open(template_path, "r") as f:
                    all_template_config = json.load(f)
                template_config = all_template_config[template_name]
                template_params = template_config
                # my template list has the following keys: seed, softmax_tem, treatments, mediator
                # overload these keys with the values from the template
                params["seed"] = template_params.get("seed")
                params["softmax_tem"] = template_params.get("softmax_tem")
                params["treatments"] = template_params.get("treatments")
                if "mediator" in template_params:
                    params["mediator"] = template_params.get("mediator")
                else:
                    params["mediator"] = 0.0

            seed = params.get("seed")
            softmax_tem = params.get("softmax_tem")
            treatments = params.get("treatments")
            n = params.get("n")
            rep = params.get("rep")
            if "mediator" in params:
                mediator = params.get("mediator")
            else:
                mediator = 0.0

            # Map algorithm names to their configurations
            algs_config = {}
            for alg_name in params["algorithms"]:
                if alg_name in shared_algorithms:
                    algs_config[alg_name] = shared_algorithms[
                        alg_name
                    ]  # Skip n and rep
                else:
                    raise ValueError(
                        f"Algorithm '{alg_name}' not found in shared_algorithms.json"
                    )

            config_out = {
                "n": n,
                "rep": rep,
                "seed": seed,
                "softmax_tem": softmax_tem,
                "treatments": treatments,
                "mediator": mediator,
                "algs": algs_config,
            }

            with open(
                os.path.join(output_dir, "config_general_run.json"), "w"
            ) as out_file:
                json.dump(config_out, out_file, indent=4)
            print(f"Generated: {output_dir}/config_general_run.json")


if __name__ == "__main__":
    treatment = [
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.7,
        0.9,
        1,
        1.2,
        1.5,
        1.75,
        2,
        3,
    ]

    create_test_STE_config(treatment)

    generate_experiments()
