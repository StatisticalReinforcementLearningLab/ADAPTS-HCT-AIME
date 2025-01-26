import os
import json
import shutil

DESIGN_DIR = "Experiment_Test_Algs/configs/design_decisions"
SHARED_ALGOS_FILE = "Experiment_Test_Algs/configs"
SHARED_ALGOS_COLLABORATION_FILE = "Experiment_Test_Algs/configs/shared_algorithm_test_collaboration.json"
ENV_TEMPLATE_FILE = "Experiment_Test_Algs/configs"
ANALYSIS_FILE = "Experiment_Test_Algs/analysis.ipynb"

OUTPUT_ROOT = "Experiment_Test_Algs/experiments"

def load_and_combine_environment_templates():
    template_files = [f for f in os.listdir(ENV_TEMPLATE_FILE) if f.startswith("environment_template") and f.endswith(".json")]
    combined_templates = {}

    for template_file in template_files:
        template_path = os.path.join(ENV_TEMPLATE_FILE, template_file)
        with open(template_path, "r") as f:
            template_data = json.load(f)
            combined_templates.update(template_data)

    return combined_templates

combined_templates = load_and_combine_environment_templates()
# print(combined_templates)

def load_shared_algorithms():
    template_files = [f for f in os.listdir(SHARED_ALGOS_FILE) if f.startswith("shared_algorithms") and f.endswith(".json")]
    combined_templates = {}

    for template_file in template_files:
        template_path = os.path.join(SHARED_ALGOS_FILE, template_file)
        with open(template_path, "r") as f:
            template_data = json.load(f)
            combined_templates.update(template_data)

    return combined_templates

combined_shared_algorithms = load_shared_algorithms()


def generate_experiments():
    # with open(SHARED_ALGOS_FILE, "r") as f:
    #     shared_algorithms = json.load(f)

    with open(SHARED_ALGOS_COLLABORATION_FILE, "r") as f:
        shared_algorithms_collaboration = json.load(f)

    #Loop through all files (e.g. design_decisions/Pooling_Effect.json)
    for design_file in os.listdir(DESIGN_DIR):
        if design_file == ".DS_Store":
            continue
        design_path = os.path.join(DESIGN_DIR, design_file)
        # print(design_path)
        with open(design_path, "r") as f:
            design_config = json.load(f)

        if "env_list" not in design_config:
            continue

        #Convert each file to a directory with config_general_run.json and analysis.ipynb
        for env in design_config["env_list"]:
            env_config = combined_templates[env]
            output_dir = os.path.join(OUTPUT_ROOT, design_file.split(".")[0], env)
            os.makedirs(output_dir, exist_ok=True)

            n = design_config.get("n")
            rep = design_config.get("rep")
            env_config["n"] = n
            env_config["rep"] = rep

            seed = env_config.get("seed")
            softmax_tem = env_config.get("softmax_tem")
            treatments = env_config.get("treatments")
            mediator = env_config.get("mediator")

            #Map algorithm names to their configurations
            algs_config = {}
            if isinstance(design_config["algorithm_list"], list):
                for alg_name in design_config["algorithm_list"]:
                    if alg_name in combined_shared_algorithms:
                        algs_config[alg_name] = combined_shared_algorithms[alg_name]  #Skip n and rep
                    else:
                        raise ValueError(f"Algorithm '{alg_name}' not found in shared_algorithms.json")
            else:
                algs_config = shared_algorithms_collaboration

            config_out = {
                "n": n,
                "rep": rep,
                "seed": seed,
                "softmax_tem": softmax_tem,
                "treatments": treatments,
                "mediator": mediator,
                "algs": algs_config
            }

            with open(os.path.join(output_dir, "config_general_run.json"), "w") as out_file:
                json.dump(config_out, out_file, indent=4)
            print(f"Generated: {output_dir}/config_general_run.json")

            analysis_dest = os.path.join(output_dir, "analysis.ipynb")
            if not os.path.exists(analysis_dest):
                shutil.copy(ANALYSIS_FILE, analysis_dest)
                print(f"Copied analysis.ipynb to: {analysis_dest}")

if __name__ == "__main__":
    generate_experiments()
