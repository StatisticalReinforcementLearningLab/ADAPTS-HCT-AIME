import os
import json
import shutil
from itertools import product
import argparse

DESIGN_DIR = "Experiment_Test_Opt_Policy/configs/design_decisions"
SHARED_ALGOS_FILE = "Experiment_Test_Opt_Policy/configs/shared_algorithms.json"
ANALYSIS_FILE = "Experiment_Test_Opt_Policy/analysis.ipynb"

OUTPUT_ROOT = "Experiment_Test_Opt_Policy/experiments"
TEST_STE_FILENAME = "Test_STE_One_Param.json"

def create_test_STE_config(treatment1, treatment2):

    all_algorithms = [
        "OptPolicy_SingleAgent_Gamma0",
        "OptPolicy_SingleAgent_Gamma0.2",
        "OptPolicy_SingleAgent_Gamma0.5",
        "OptPolicy_SingleAgent_Gamma0.9",
        "OptPolicy_OfflineValueIteration_OnlyFirstThreeFeatures",
        "OptPolicy_OfflineValueIteration_OnlyFirstSixFeatures_Rep1000",
        "OptPolicy_OfflineValueIteration_OnlyFirstFiveFeatures_Rep1000",
        "MRT0",
        "MRT1",
        "MRT0.5", "MRT0.6",
        "MRT0.7",
        "MRT0.8",
        "MRT0.9"
    ]
    mediator_values = [0, 1, 2]
    test_ste_config = {}

    def generate_config(t0, t1, t2, t3, mediator, algorithms=all_algorithms):
        """Helper function to generate a configuration."""
        key = f"T0_{t0}_T1_{t1}_T2_{t2}_T3_{t3}_M_{mediator}"
        config = {
            "treatments": [t0, t1, t2, t3],
            "seed": 1,
            "softmax_tem": 100,
            "n": 100,
            "rep": 63,
            "mediator": mediator,
            "algorithms": algorithms,
        }
        return key, config


    for t1 in treatment1:
        for t2 in treatment2:
            for mediator in mediator_values:
                # Mode 1
                t0 = t1
                t3 = t1
                key, config = generate_config(t0, t1, t2, t3, mediator)
                test_ste_config[key] = config

                # Mode 2
                t0 = t1 / 2
                t3 = t1 / 2
                key, config = generate_config(t0, t1, t2, t3, mediator)
                test_ste_config[key] = config

                # Mode 3
                t0 = t1 * 2
                t3 = t1 * 2
                key, config = generate_config(t0, t1, t2, t3, mediator)
                test_ste_config[key] = config

    output_path = os.path.join(DESIGN_DIR, TEST_STE_FILENAME)
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

            analysis_dest = os.path.join(output_dir, "analysis.ipynb")
            if not os.path.exists(analysis_dest):
                shutil.copy(ANALYSIS_FILE, analysis_dest)
                print(f"Copied analysis.ipynb to: {analysis_dest}")


if __name__ == "__main__":


    treatment1 = [
        0.1,
        0.3,
        0.5,
        0.7,
        0.9,
        1,
        1.2,
        1.5,
        1.75,
        2, 
        3
    ]  
    treatment2 = [
        0.1,
        0.3,
        0.5,
        0.7,
        0.9,
        1,
        1.2,
        1.5,
        1.75,
    ] 

    create_test_STE_config(treatment1, treatment2)

    generate_experiments()
