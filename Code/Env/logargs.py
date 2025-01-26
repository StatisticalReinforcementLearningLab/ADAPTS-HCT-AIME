import argparse
import os
import json

parser = argparse.ArgumentParser(prog="ADAPT-HCT Simulation Testbed Logger")

parser.add_argument("--logEnabled", action="store_true")  
# whether to log
parser.add_argument("--logPath", type=str, default="./log.txt")  
# path for logger
parser.add_argument("--config", type=str, default=None)

args, unknown = parser.parse_known_args()

if args.config is not None:
    argparse_dict = vars(args)
    if os.path.isfile(args.config):
        with open(args.config, "r") as f:
            # parser.set_defaults(**json.load(f))
            json_dict = json.load(f)
            argparse_dict.update(json_dict)
    # Reload arguments to override config file values with command line values
    args = argparse.Namespace(**argparse_dict)