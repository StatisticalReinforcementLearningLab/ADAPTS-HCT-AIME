import argparse
import os
import json

parser = argparse.ArgumentParser(prog="ADAPT-HCT Simulation Testbed Env")
parser.add_argument("--seed", type=int, default=0)  # random seed
parser.add_argument("-W", type=int, default=14)  # number of weeks in the trial

# Check whether arguments are set correctly
parser.add_argument("--printWhenRead", action="store_true")

# 0.175, 0.125, 0.85, 0.225
# For Environment
parser.add_argument("--mediator", type = float, default=0.0, help="add mediator effect of size me")
parser.add_argument("--add_direct_effect", action="store_true", help="add direct effect from carepartner to target")
parser.add_argument("--extreme_direct_effect", action="store_true", help="add extreme direct effect from carepartner to target")

parser.add_argument("--treat0", type=float, default=1)  # multiplies the main treatment effect
parser.add_argument("--treat1", type=float, default=1)  # multiplies the relationship/treatment interaction effect
parser.add_argument("--treat2", type=float, default=1)  # multiplies the burden/treatment interaction effect
parser.add_argument("--treat3", type=float, default=1)  # multiplies the main treatment effect of game intervention on weekly relationship
parser.add_argument("--bNoise", type=float, default=2.4)  # noise variance of the burden transition

parser.add_argument("--addTrend", action="store_true")  # removing the increasing baseline effects
parser.add_argument("--noRelBurden", action="store_true")  # removing the relationship/burden interaction effect

parser.add_argument("--Warmup", type=int, default=1)  # number of weeks that applies random policy

parser.add_argument("--reinforces", action="store_true")  # whether to add reinforces effect of game intervention

parser.add_argument("--config", type=str, default=None)

parser.add_argument("--store_data", action="store_true")

args, unknown = parser.parse_known_args()

if args.config is not None:
    argparse_dict = vars(args)
    if os.path.isfile(args.config):
        with open(args.config, "r") as f:
            # parser.set_defaults(**json.load(f))
            json_dict = json.load(f)
            if 'alg1' in json_dict.keys():
                del json_dict['alg1']
                del json_dict['alg2']
                del json_dict['alg3']
            argparse_dict.update(json_dict)
    # Reload arguments to override config file values with command line values
    args = argparse.Namespace(**argparse_dict)