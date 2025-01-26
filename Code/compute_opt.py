import os
import sys
import argparse
import time
import pickle
import random
import numpy as np
from collections import defaultdict
from args import *
from Env import Trial, envargs
from Algorithms import MRT, Bandit
from Algorithms.SingleAgentOptPolicy import SingleAgent
from utils import opt_policy, generate_discretization, STATE_DIM

# Setup paths and seed
sys.path.append("./Code/")
args.seed = int(time.time())
np.random.seed(args.seed)
random.seed(args.seed)


FEATURE_CONFIGS = {
    "only_first_three_features": 3,
    "only_first_five_features": 5,
    "only_first_six_features": 6,
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute optimal policy")
    parser.add_argument("--treat0", type=float, required=True)
    parser.add_argument("--treat1", type=float, required=True)
    parser.add_argument("--treat2", type=float, required=True)
    parser.add_argument("--treat3", type=float, required=True)
    parser.add_argument("--bNoise", type=float, required=True)
    parser.add_argument("--mediator", type=float, required=True)
    parser.add_argument("--optimal_policy", type=str, required=True)
    parser.add_argument("--single_agent_gamma", type=lambda v: None if v == "None" else float(v), default=None)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=193)
    return parser.parse_args()

def run_single_agent_trial(args):
    agent = SingleAgent(person="Target", gamma=args.single_agent_gamma)
    for _ in range(args.rep):
        trial = Trial.Trial(algorithm=[agent, agent, agent], args=envargs.args)
        for _ in range(args.n):
            trial.employNewSubject()
            while trial.isGoing():
                trial.progressTime()
    return agent

def run_mrt_trial(feature_count, args):
    new_envargs = envargs.args
    new_envargs.store_data = True
    agents = [MRT.MRT(person="Target"), MRT.MRT(person="Care"), MRT.MRT(person="Game")]
    all_data = []
    
    for _ in range(args.rep):
        trial = Trial.Trial(algorithm=agents, args=new_envargs)
        for _ in range(args.n):
            trial.employNewSubject()
            while trial.isGoing():
                trial.progressTime()
        for dyad in trial.subjects:
            all_data.extend(dyad.data)
    
    data = np.array(all_data)[:, :feature_count]
    dis_ind = [1 if i < 3 else 0 for i in range(feature_count)]  # Example discretization pattern
    return discretize_data(data, dis_ind), dis_ind

def discretize_data(data, dis_ind):
    quantiles = []
    for i, should_discretize in enumerate(dis_ind):
        if should_discretize:
            dis_s, quantile = generate_discretization(data[:, i], 10)
            data[:, i] = dis_s
            quantiles.append(quantile)
        else:
            quantiles.append(0)
    return data, quantiles

def offline_value_iteration(data, feature_count, horizon):
    Q = defaultdict(lambda: defaultdict(float))
    state_action_next = defaultdict(list)
    all_states = set()
    all_actions = set()

    for i, entry in enumerate(data):
        h = entry[-1] % horizon
        state = tuple(entry[:feature_count])
        action = tuple(entry[STATE_DIM:(STATE_DIM + 3)])
        reward = entry[-2]
        
        all_states.add(state)
        all_actions.add(action)
        
        next_state = tuple(data[i+1, :feature_count]) if h < horizon-1 and i < len(data)-1 else state
        state_action_next[(state, action, h)].append((reward, next_state))

    for h in reversed(range(horizon)):
        for state in all_states:
            for action in all_actions:
                outcomes = state_action_next.get((state, action, h), [])
                if not outcomes:
                    continue
                
                total = sum(reward + (max(Q[h+1][(next_state, a)] for a in all_actions) if h < horizon-1 else 0)
                            for reward, next_state in outcomes)
                Q[h][(state, action)] = total / len(outcomes)

    return Q, all_actions

def save_policy(args, policy_data, policy_type):
    treat_values = '_'.join(f"{getattr(args, f'treat{i}'):.2f}" for i in range(4))
    save_dir = (f"./Opt_Policy/policy_{treat_values}_{args.bNoise:.2f}_"
                f"{args.mediator}_poltype_{policy_type}_"
                f"{args.single_agent_gamma or 0.0:.2f}.pkl")
    with open(save_dir, 'wb') as f:
        pickle.dump(policy_data, f)
    print(f"Policy saved to {save_dir}")

def main():
    args = parse_arguments()
    print(f"Computing optimal policy: {args.optimal_policy}", flush=True)

    if args.optimal_policy == "singleagent":
        agent = run_single_agent_trial(args)
        policy = opt_policy(
            Q=None, all_actions=agent.action_space, horizon=args.horizon,
            quantiles=None, dis_ind=None, w=agent.theta
        )
        save_policy(args, policy, "singleagent")
        
    elif args.optimal_policy in FEATURE_CONFIGS:
        feature_count = FEATURE_CONFIGS[args.optimal_policy]
        data, dis_ind = run_mrt_trial(feature_count, args)
        discretized_data, quantiles = discretize_data(data, dis_ind)
        Q, all_actions = offline_value_iteration(discretized_data, feature_count, args.horizon)
        policy = opt_policy(Q, all_actions, args.horizon, quantiles, dis_ind, feature_count)
        save_policy(args, policy, args.optimal_policy)
        
    else:
        raise ValueError(f"Unknown policy type: {args.optimal_policy}")

if __name__ == "__main__":
    main()