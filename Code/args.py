import argparse
import os
import json

parser = argparse.ArgumentParser(prog="ADAPT-HCT Simulation Testbed")
parser.add_argument("--seed", type=int, default=0)  # random seed
parser.add_argument("-W", type=int, default=14)  # number of weeks in the trial

# List of arguments:
# 1. pooling
#   - bandit component pooling across dyads
#   - Episodic MDP component (AYA and caregiver) pooling across dyads
#   - bandit component pooling across weeks
# 2. feature construction. Try multiple options


############################

# For experiment
parser.add_argument(
    "--recruitProb", type=float, default=0.1
)  # probability of recruiting a dyad each week

parser.add_argument("--identical", action = "store_true", help="if true, only run experiments with alg1=alg2=alg3")

parser.add_argument("--sequential", action = "store_true", help="if we recruit based on order")

parser.add_argument("--alg1",
    default={
        "MRT1": {
            "class": "MRT",
            "parameters": {"gameProb": 0.3, "targetProb": 1.0, "caregiverProb": 0.7},
        },
        "MRT0": {
            "class": "MRT",
            "parameters": {"gameProb": 0.3, "targetProb": 0.0, "caregiverProb": 0.7},
        },
        "DyadicRL": {
            "class": "DyadicRL",
            "parameters": {
                "lambda_": 1,
                "sigma": 1,
                "update": 1,
                "poolAcrossDyads": True,
            },
        },
        "DyadicRLNoPool": {
            "class": "DyadicRL",
            "parameters": {
                "lambda_": 1,
                "sigma": 1,
                "update": 1,
                "poolAcrossDyads": False,
            },
        },
        "Bandit": {
            "class": "Bandit",
            "parameters": {
                "lambda_": 1,
                "sigma": 1,
                "update": "week",
                "maxN": None,
                "incrUpdate": True,
                "poolAcrossDyads": True,
            },
        },
        "BanditNoPool": {
            "class": "Bandit",
            "parameters": {
                "lambda_": 1,
                "sigma": 1,
                "update": "week",
                "maxN": None,
                "incrUpdate": True,
                "poolAcrossDyads": False,
            },
        },
        "Deterministic1":{
            "class": "Deterministic",
            "parameters":{
                "mode": 1
            }
        },
        "Deterministic2":{
            "class": "Deterministic",
            "parameters":{
                "mode": 2
            }
        },
        "Deterministic3":{
            "class": "Deterministic",
            "parameters":{
                "mode": 3
            }
        },
        "Deterministic0":{
            "class": "Deterministic",
            "parameters":{
                "mode": 0
            }
        }
    },
)  # this algorithm is a dictionary. Can be specified as a str

parser.add_argument("--alg2",
    default={
        "MRT1": {
            "class": "MRT",
            "parameters": {"gameProb": 0.0, "targetProb": 0.0, "caregiverProb": 1.0},
        },
        "MRT0": {
            "class": "MRT",
            "parameters": {"gameProb": 0.0, "targetProb": 0.0, "caregiverProb": 0.0},
        },
        "DyadicRL": {
            "class": "DyadicRL",
            "parameters": {
                "lambda_": 1,
                "sigma": 1,
                "update": 1,
                "poolAcrossDyads": True,
            },
        },
        "DyadicRLNoPool": {
            "class": "DyadicRL",
            "parameters": {
                "lambda_": 1,
                "sigma": 1,
                "update": 1,
                "poolAcrossDyads": False,
            },
        },
        "Bandit": {
            "class": "Bandit",
            "parameters": {
                "lambda_": 1,
                "sigma": 1,
                "update": "week",
                "maxN": None,
                "incrUpdate": True,
                "poolAcrossDyads": True,
            },
        },
        "BanditNoPool": {
            "class": "Bandit",
            "parameters": {
                "lambda_": 1,
                "sigma": 1,
                "update": "week",
                "maxN": None,
                "incrUpdate": True,
                "poolAcrossDyads": False,
            },
        },
        "Deterministic1":{
            "class": "Deterministic",
            "parameters":{
                "mode": 1
            }
        },
        "Deterministic2":{
            "class": "Deterministic",
            "parameters":{
                "mode": 2
            }
        },
        "Deterministic3":{
            "class": "Deterministic",
            "parameters":{
                "mode": 3
            }
        },
        "Deterministic0":{
            "class": "Deterministic",
            "parameters":{
                "mode": 0
            }
        }
    },
)  # this algorithm is a dictionary. Can be specified as a str

parser.add_argument("--alg3",
    default={
        "Bandit": {
            "class": "Bandit",
            "parameters": {
                "lambda_": 1,
                "sigma": 1,
                "update": "week",
                "maxN": None,
                "incrUpdate": True,
            },
        }
    },
)  # this algorithm is a dictionary. Can be specified as a str

# allow us to directly specify algorithm combinations
parser.add_argument("--algs", default=None, help=""" Example
    {
        "comb1": [
            {
                "class": "MRT",
                "parameters": {"gameProb": 0.3, "targetProb": 1.0, "caregiverProb": 0.7},
            },
            {
                "class": "MRT",
                "parameters": {"gameProb": 0.3, "targetProb": 1.0, "caregiverProb": 0.7},
            },
            {
                "class": "MRT",
                "parameters": {"gameProb": 0.3, "targetProb": 1.0, "caregiverProb": 0.7},
            }
        ]
    }
    """
)

parser.add_argument("--run_one", default=None, help="run one specific algorithm. the algorithm must be an element of algs")
parser.add_argument("--fix_id", type=int, default=-1, help="fix the dyad id. -1 means random")

parser.add_argument("--rep", type=int, default=2)  # number of replication
parser.add_argument(
    "-n", type=int, default=10
)  # number of dyads recruited in a single trial

# For logging and saving

parser.add_argument("--save", type=str, default=None)  # where the results are saved

parser.add_argument("--horizon", type=int, default=14)  # horizon length for computing optimal policy

parser.add_argument("--plot", action = "store_true", help="if true, plot")

# algorithm
parser.add_argument("--config", type=str, default=None)  # algorithm configs

args, unknown = parser.parse_known_args()
if args.config is not None:
    argparse_dict = vars(args)
    if os.path.isfile(args.config):
        print(f"Loading config: {args.config}")
        with open(args.config, "r") as f:
            # parser.set_defaults(**json.load(f))
            json_dict = json.load(f)
            argparse_dict.update(json_dict)
    # Reload arguments to override config file values with command line values
    args = argparse.Namespace(**argparse_dict)