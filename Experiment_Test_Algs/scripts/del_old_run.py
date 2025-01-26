import glob
import os
import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('--subdir', type=str, default=None)
parser.add_argument('--algorithm', type=str, default=None)
parser.add_argument('--env', type=str, default=None)

args = parser.parse_args()

if __name__ == "__main__":
    if args.subdir is None:
        exit()

    for subdir in [args.subdir]:
        for config_file in glob.glob(f'Experiment_Test_Algs/experiments/{subdir}/*/config_general_run.json'):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    # Extract experiment parameters
                    seed = config['seed']
                    softmax_tem = config['softmax_tem']
                    treatments = config['treatments']
                    treat0, treat1, treat2, treat3 = treatments
                    if treat1 == 1.0:
                        treat1 = "1"
                    if treat2 == 1.0:
                        treat2 = "1"
                    if treat3 == 1.0:
                        treat3 = "1"
                    n = config['n']
                    rep = config['rep']
                    mediator = config['mediator']

                    # Extract the algorithm names
                    algs = list(config['algs'].keys())
                    algs.sort()
                    print(f"Loaded algorithms: {algs}")

                    # Initialize a dictionary to store data
                    data = {}

                    # Adjusted file path logic
                    if args.algorithm is not None:
                        algs = [args.algorithm]
                    for alg in algs:
                        # Construct the file path based on the filename conventio
                        mediator_str = str(mediator).rstrip('0').rstrip('.') if '.' in str(mediator) else str(mediator)
                        filename = f"Experiment_Test_Algs/results/save_t{treat0}_{treat1}_{treat2}_{treat3}_s{seed}_soft{softmax_tem}_n{n}_rep{rep}_ME{mediator_str}_{alg}.csv"
                        print("Deleting", filename)
                        try:
                            os.remove(filename)
                        except FileNotFoundError:
                            print(f"File {filename} not found")
