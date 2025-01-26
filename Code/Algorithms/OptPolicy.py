from .Algorithm import Algorithm
from .Policy import Policy
import numpy as np
import os.path
import sys
import subprocess
import pickle
import time
from utils import opt_policy

# import Env.envargs as envargs


class OptPolicy(Algorithm):
    def __init__(
        self,
        n=100,
        rep=100,
        horizon=14,
        person="Target",
        reward_type="naive",
        delayed_weight=None,
        naive_weight=None,
        optimal_policy="singleagent",
        single_agent_gamma=None
    ):
        print(f"Initializing OptPolicy with n={n}, rep={rep}, horizon={horizon}, "
              f"person={person}, reward_type={reward_type}, delayed_weight={delayed_weight}, "
              f"naive_weight={naive_weight}", flush=True)
        super().__init__(
            "OptPolicy",
            person=person,
            reward_type=reward_type,
            delayed_weight=delayed_weight,
            naive_weight=naive_weight,
        )
        self.treat0 = None
        self.n = n
        self.rep = rep
        self.horizon = horizon
        self.optimal_policy = optimal_policy
        self.single_agent_gamma = single_agent_gamma

    def loadPolicy(self, envargs, linear=False):

        print(f"Loading policy with parameters: treat0={envargs.treat0}, "
              f"treat1={envargs.treat1}, treat2={envargs.treat2}, "
              f"treat3={envargs.treat3}, bNoise={envargs.bNoise}, mediator={envargs.mediator}, "
              f"optimal_policy={self.optimal_policy}", flush=True)
        self.treat0 = envargs.treat0
        self.treat1 = envargs.treat1
        self.treat2 = envargs.treat2
        self.treat3 = envargs.treat3
        self.bNoise = envargs.bNoise
        self.mediator = envargs.mediator
        self.optimal_policy = self.optimal_policy
        self.single_agent_gamma = self.single_agent_gamma
        save_dir = "./Opt_Policy/policy_%.2f_%.2f_%.2f_%.2f_%.2f_%.1f_poltype_%s_%.2f.pkl" % (
            self.treat0,
            self.treat1,
            self.treat2,
            self.treat3,
            self.bNoise,
            float(self.mediator),
            self.optimal_policy,
            self.single_agent_gamma if self.single_agent_gamma is not None else 0.0  # Handle None case
        )
        print(f"Policy save directory: {save_dir}", flush=True)
        

        if not os.path.isfile(save_dir):

            print(f"Policy file not found. Generating new policy at {save_dir}", flush=True)
            command = [
                sys.executable,
                "./Code/compute_opt.py",
                "--treat0", "%.2f" % self.treat0,
                "--treat1", "%.2f" % self.treat1,
                "--treat2", "%.2f" % self.treat2,
                "--treat3", "%.2f" % self.treat3,
                "--bNoise", "%.2f" % self.bNoise,
                "--n", str(self.n),
                "--rep", str(self.rep),
                "--horizon", str(self.horizon),
                "--optimal_policy", self.optimal_policy,
                "--mediator", str(self.mediator),
                "--single_agent_gamma", str(self.single_agent_gamma)
            ]
            print("Running command:", " ".join(command), flush=True)
                    
            subprocess.run(command, check=True)
        with open(save_dir, "rb") as f:
            print(f"Loading policy from {save_dir}", flush=True)
            self.policy = pickle.load(f)
            print(f"Policy successfully loaded from {save_dir}", flush=True)
    def addData(self, data, h: int, person: str, idx: int):
        pass

    def __str__(self):
        policy_str = (f"OptPolicy: treat0={self.treat0}, treat1={self.treat1}, "
                      f"treat2={self.treat2}, treat3={self.treat3}, "
                      f"bNoise={self.bNoise}, mediator={self.mediator}")
        return policy_str

    def getPolicy(self, dyad) -> Policy:
        if self.treat0 is None:
            self.loadPolicy(dyad.target.args)
        state = dyad.getState()
        actions = self.policy.policy(
            self.person, state, (dyad.currentTime - dyad.startTime).getAbsoluteTime()
        )
        action = {"Target": actions[0], "Care": actions[1], "Game": actions[2]}[
            self.person
        ]
        return action
