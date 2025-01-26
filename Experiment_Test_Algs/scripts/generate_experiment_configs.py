import os
import json
import shutil

DESIGN_DIR = "Experiment_Test_Algs/configs/design_decisions"
SHARED_ALGOS_FILE = "Experiment_Test_Algs/configs/shared_algorithms.json"
ANALYSIS_FILE = "Experiment_Test_Algs/analysis.ipynb"

OUTPUT_ROOT = "Experiment_Test_Algs/experiments"

def generate_experiments():
    with open(SHARED_ALGOS_FILE, "r") as f:
        shared_algorithms = json.load(f)

    #Loop through all files (e.g. design_decisions/Pooling_Effect.json)
    for design_file in os.listdir(DESIGN_DIR):
        if design_file == ".DS_Store":
            continue
        design_path = os.path.join(DESIGN_DIR, design_file)
        # print(design_path)
        with open(design_path, "r") as f:
            design_config = json.load(f)

        #Convert each file to a directory with config_general_run.json and analysis.ipynb
        for baseline, params in design_config.items():
            output_dir = os.path.join(OUTPUT_ROOT, design_file.split(".")[0], baseline)
            os.makedirs(output_dir, exist_ok=True)

            if "template" in params:
                template_name = params["template"]
                template_path = os.path.join(DESIGN_DIR, f"../environment_template.json")
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


            #Map algorithm names to their configurations
            algs_config = {}
            for alg_name in params["algorithms"]:
                if alg_name in shared_algorithms:
                    algs_config[alg_name] = shared_algorithms[alg_name]  #Skip n and rep
                else:
                    raise ValueError(f"Algorithm '{alg_name}' not found in shared_algorithms.json")

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
