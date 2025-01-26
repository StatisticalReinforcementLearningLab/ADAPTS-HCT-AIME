import argparse
import os
import json

parser = argparse.ArgumentParser(prog="ADAPT-HCT Simulation Testbed Alg")

# For reward construction
parser.add_argument("--reward_type", type=str, default="naive")
parser.add_argument("--delayed_weight", type=float, default=0.2)
parser.add_argument("--naive_weight", type=float, default=1)

parser.add_argument("--n", type=int, default=10)


parser.add_argument("--config", type=str, default=None)

parser.add_argument(
    "--softmax_tem", type=float, default=100
)

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