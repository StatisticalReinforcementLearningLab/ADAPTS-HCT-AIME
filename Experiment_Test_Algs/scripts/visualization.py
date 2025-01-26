# 2024-09-27
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import json
import argparse

args = argparse.ArgumentParser()
args.add_argument('--subdir', type=str, nargs='+', default=['/All_Run03_Similar/', '/All_Run015_Similar/', '/All_Run/'])
args.add_argument('--mode', type=str, default='all')
args = args.parse_args()

alg_short_names = {
    "MRT": "",
    "SingleAgent_RwdNaive_Pool_Sigm0.5_Lambda0.75": "SingleAgent",
    "RewardLearning": "",
    "RLSVI_Inf_RwdNaive_Pool_Sigm0.5_Lambd0.75": "",
    "Bayes_RwdNaive_Pool_Sigm0.1_Lambd5_Prior5": "",
    "RLSVI_Inf_RwdMixed_Pool_Sigm0.5_Lambd0.75": "",
    "RewardLearning_Gamma_0.5": "",
    "RLSVI_Inf_Gamma_0.5": "MultiAgent",
    "RewardLearningNewCare": "MultiAgent+SurrogateRwd",
    "SingleAgent_Gamma0.5": "",
}

def visualize_experiment(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    # Extract experiment parameters
    seed = config['seed']
    softmax_tem = config['softmax_tem']
    treatments = config['treatments']
    treat0, treat1, treat2, treat3 = treatments
    n = config['n']
    rep = config['rep']
    mediator = config['mediator']

    # Extract the algorithm names
    algs = list(config['algs'].keys())
    algs.sort()
    # algs = [alg for alg in algs if "UsePrior" not in alg]
    # print(f"Loaded algorithms: {algs}")

    # Initialize a dictionary to store data
    data = {}

    # Adjusted file path logic
    for alg in algs:
        # Construct the file path based on the filename conventio
        mediator_str = str(mediator).rstrip('0').rstrip('.') if '.' in str(mediator) else str(mediator)
        if treat1 == 1:
            treat1 = "1"
        if treat2 == 1:
            treat2 = "1"
        if treat3 == 1:
            treat3 = "1"
        if treat0 == 1:
            treat0 = "1"
        filename = f"Experiment_Test_Algs/results/save_t{treat0}_{treat1}_{treat2}_{treat3}_s{seed}_soft{softmax_tem}_n{n}_rep{rep}_ME{mediator_str}_{alg}.csv"
        # print(f"Loading data for {alg} from {filename}")
        try:
            if "MRT" in alg:
                alg = "MRT"
            data[alg] = pd.read_csv(filename) * 14  # Load and scale the data
        except FileNotFoundError:
            print(f"File not found: {filename}. Skipping {alg}.")

    # Plot the data

    # Generate a color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 11))  # 3 distinct colors for the groups

    # Create a dictionary to map algorithms to their group colors
    color_map = {}
    for alg in algs:
        if "Bandit_Delayed" in alg:
            color_map[alg] = colors[0]
        elif "RLSVI_Inf" in alg:
            color_map[alg] = colors[1]
        elif "UsePrior" in alg:
            color_map[alg] = colors[2]
        elif "Gamma" in alg:
            color_map[alg] = colors[3]
        elif "Bayes" in alg:
            color_map[alg] = colors[4]
        elif "RLSVI_Finite" in alg:
            color_map[alg] = colors[5]
        elif "CareNoLearning" in alg:
            color_map[alg] = colors[6]
        elif "CareMediator" in alg:
            color_map[alg] = colors[7]
        elif "RewardLearningNewCare" in alg:
            color_map[alg] = colors[8]
        elif "RewardLearning" in alg:
            color_map[alg] = colors[8]
        elif "NextWeek" in alg:
            color_map[alg] = colors[9]
        elif "SingleAgent" in alg:
            color_map[alg] = colors[10]
        else:
            color_map[alg] = 'gray'  # Default color for other algorithms

    # Create a list of line styles to differentiate algorithms within groups
    shape_map = {}
    for alg in algs:
        if "Mediator" in alg:
            shape_map[alg] = 'o'
        elif "Naive" in alg:
            shape_map[alg] = 's'
        elif "Delayed" in alg:
            shape_map[alg] = 'D'
        elif "Mixed" in alg:
            shape_map[alg] = 'x'
        else:
            shape_map[alg] = 'o'

    for alg in algs:
        if "Gamma0.1" in alg:
            shape_map[alg] = 's'
        if "Gamma0.3" in alg:
            shape_map[alg] = 'D'
        if "Gamma0.5" in alg:
            shape_map[alg] = 'x'

    line_style_map = {}
    for alg in algs:
        if "NoPool" in alg:
            line_style_map[alg] = '--'
        else:
            line_style_map[alg] = '-'

    # Filter out algorithms that were not loaded
    loaded_algs = [alg for alg in algs if alg in data]
    # print(f"Algorithms with loaded data: {loaded_algs}")
    for names in ['Rewards', 'Distress', 'Relationship']:
        plt.figure(figsize=(4, 2.66666667))
        for alg in loaded_algs:
            # print(alg)

            if alg == "MRT":
                continue
            # print(alg_short_names[alg])
            if alg_short_names[alg] == "":
                continue
            # 
            # Plot data with confidence interval
            plt.plot(
                data[alg]['Mean_' + names] - data["MRT"]['Mean_' + names],
                label=alg_short_names[alg],
                color=color_map[alg],
                linestyle=line_style_map[alg],
                marker=shape_map[alg],
                markersize=4,
                markevery=14
            )

            # Calculate the cumulative standard error
            std = data[alg]['Std_' + names]
            confidence_interval = std / np.sqrt(config['rep'])

            # Plot the confidence range
            plt.fill_between(
                range(len(data[alg])),
                data[alg]['Mean_' + names] - data["MRT"]['Mean_' + names] - confidence_interval,
                data[alg]['Mean_' + names] - data["MRT"]['Mean_' + names] + confidence_interval,
                alpha=0.1,
                color=color_map[alg]
            )
        plt.ylim(-20, 130)
        plt.xlabel('Dyads')
        plt.xticks(range(0, len(data[loaded_algs[0]]), 28), [str(int(x/14)) for x in range(0, len(data[loaded_algs[0]]), 28)])
        plt.ylabel('Cumulative ' + names)
        plt.legend()
        plt.tight_layout(pad=0.1)
        # plt.savefig(f'{os.path.dirname(config_file)}/All.pdf')
        for ext in ['pdf', 'png']:
            plt.savefig(f'{os.path.dirname(config_file)}/All_{names}.{ext}')
        # print(f'Experiment_Test_Algs/experiments/{os.path.basename(os.path.dirname(config_file))}/All.png')
        plt.close()
    # plt.show()

# iterate iteratively through all the files in the experiments folder
if __name__ == "__main__":
    for subdir in args.subdir:
        for config_file in glob.glob(f'Experiment_Test_Algs/experiments/{subdir}/*/config_general_run.json'):
            print(f"Processing {config_file}")
            try:
                visualize_experiment(config_file)
            except Exception as e:
                print(f"Error processing {config_file}: {e}")
                continue
