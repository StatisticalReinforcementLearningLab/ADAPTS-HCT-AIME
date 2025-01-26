from args import *
import Env.envargs as envargs
import Algorithms.algargs as algargs
from Env import Trial
import pandas as pd
# from Algorithms import MRT, Bandit, DyadicRL, Deterministic, OptPolicy, RLSVIH
from Algorithms import MRT, Bandit, OptPolicy, RLSVIH, SingleAgent, Bayes, RewardLearning, RewardLearningLocal, RewardLearningNewCare
import numpy as np
import random
import time
import matplotlib.pyplot as plt

DYAD_NUM = 63

random.seed(args.seed)
np.random.seed(args.seed)

ordering = [np.random.choice(range(DYAD_NUM)) for _ in range(args.n * args.rep + 1)]

# args.seed = int(time.time())
# set seed


algDict = {
    "MRT": MRT.MRT,
    "Bandit": Bandit.Bandit,
    "OptPolicy": OptPolicy.OptPolicy,
    "RLSVIH": RLSVIH.RLSVIH,
    "SingleAgent": SingleAgent.SingleAgent,
    "Bayes": Bayes.Bayes,
    "RewardLearning": RewardLearning.RewardLearning,
    "RewardLearningLocal": RewardLearningLocal.RewardLearningLocal,
    "RewardLearningNewCare": RewardLearningNewCare.RewardLearningNewCare
}


if type(args.alg1) is not dict:
    if type(args.alg1) is str:
        args.alg1 = {"OptPolicy":{"class": args.alg1, "parameters": {}}}
if type(args.alg2) is not dict:
    if type(args.alg2) is str:
        args.alg2 = {"OptPolicy":{"class": args.alg2, "parameters": {}}}
if type(args.alg3) is not dict:
    if type(args.alg3) is str:
        args.alg3 = {"OptPolicy":{"class": args.alg3, "parameters": {}}}


def run(alg: list[dict]):
    print("Running (%s, %s, %s)" % (alg[0], alg[1], alg[2]))
    res = []
    curRes = []
    curAs = []
    curDs = []
    curRels = []

    for i in range(args.rep):
        print("\t Running %d-th round" % (i))
        agentList = []
        if alg[0]["class"] == "SingleAgent":
            agent0 = getattr(
                globals()[alg[0]["class"]], alg[0]["class"]
            )(**alg[0]["parameters"])
            agentList = [agent0, agent0, agent0] # use the same agent for all three roles
        else:
            agent0 = getattr(
                globals()[alg[0]["class"]], alg[0]["class"]
            )(**alg[0]["parameters"])
            agentList.append(agent0)
            agent1 = getattr(
                globals()[alg[1]["class"]], alg[1]["class"]
            )(**alg[1]["parameters"])
            agentList.append(agent1)
            agent2 = getattr(
                globals()[alg[2]["class"]], alg[2]["class"]
            )(**alg[2]["parameters"])
            agentList.append(agent2)

        CurTrial = Trial.Trial(algorithm=agentList)
        # run sequentially for rep times
        for j in range(args.n):
            if args.sequential:
                CurTrial.employNewSubject(idx = i)
            elif args.fix_id is not -1:
                CurTrial.employNewSubject(idx = args.fix_id)
            else:
                CurTrial.employNewSubject(idx = ordering[j + args.n * i])
            while CurTrial.isGoing():
                CurTrial.progressTime()
        for agent in agentList:
            print(agent.endSave())
        curRes.append(CurTrial.avgRs) # list of np arrays of shape (n_dyads, n_weeks)
        curAs.append(CurTrial.avgAs) # list of np arrays of shape (n_dyads, n_weeks, n_arms)
        curDs.append(CurTrial.avgDs) # list of np arrays of shape (n_dyads, n_weeks)
        curRels.append(CurTrial.avgRels) # list of np arrays of shape (n_dyads, n_weeks)
    return curRes, curAs, curDs, curRels

if args.save:
    print("Saving to %s" % (args.save))
    with open(args.save, "w") as f:
        f.write(str(args) + "\n")
        f.write(str(envargs.args) + "\n")
        f.write(str(algargs.args) + "\n")
else:
    print(args)
    print(envargs.args)
    print(algargs.args)


allResults = []
if args.algs is None:
    for alg1 in args.alg1.keys():
        for alg2 in args.alg2.keys():
            for alg3 in args.alg3.keys():
                args.alg1[alg1]['parameters']['person'] = 'Target'
                args.alg2[alg2]['parameters']['person'] = 'Care'
                args.alg3[alg3]['parameters']['person'] = 'Game'
                if args.identical and (alg1 != alg2):
                    continue
                
                algs = [args.alg1[alg1], args.alg2[alg2], args.alg3[alg3]]
                res = run(args.n, args.rep, algs)
                res.append(alg1)
                res.append(alg2)
                res.append(alg3)
                if args.save is None:
                    print(res)
                else:
                    with open(args.save, mode="a") as file:
                        file.write(str(res) + "\n")
else:
    if args.plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    for key in args.algs.keys():
        if args.run_one is not None:
            if key != args.run_one:
                continue
        algs = args.algs[key]
        algs[0]['parameters']['person'] = 'Target'
        algs[1]['parameters']['person'] = 'Care'
        algs[2]['parameters']['person'] = 'Game'
        res = run(algs)
        if args.save is None:
            print_rwd = np.mean(res[0])
            acts = np.array(res[1])
            print_act = np.mean(acts, axis=(0, 1, 2))
            print([print_rwd, print_act, key])
        else:
            rwds = np.array(res[0]).reshape(args.rep, -1)
            acts = np.array(res[1])
            acts = np.transpose(acts, (0, 2, 1, 3))  # Exchanging the second and third axes
            
            # print(np.array(res[2]).shape)
            ds = np.array(res[2]).reshape(args.rep, -1)
            rels = np.array(res[3]).reshape(args.rep, -1)
            
            cum_rwd = np.cumsum(rwds, axis=1)
            mean_rwd = np.mean(cum_rwd, axis=0)
            std_rwd = np.std(cum_rwd, axis=0)

            cum_ds = np.cumsum(ds, axis=1)
            mean_ds = np.mean(cum_ds, axis=0)
            std_ds = np.std(cum_ds, axis=0)

            cum_rels = np.cumsum(rels, axis=1)
            mean_rels = np.mean(cum_rels, axis=0)
            std_rels = np.std(cum_rels, axis=0)

            mean_act = np.mean(acts, axis=0).reshape(3, -1)
            df_res = pd.DataFrame({"Mean_Rewards": mean_rwd, "Std_Rewards": std_rwd, "Mean_AYA_Actions": mean_act[0, :], "Mean_Care_Actions": mean_act[1, :], "Mean_Game_Actions": mean_act[2, :], "Mean_Distress": mean_ds, "Std_Distress": std_ds, "Mean_Relationship": mean_rels, "Std_Relationship": std_rels})
            # Save rewards and regrets to separate CSV files
            df_res.to_csv(args.save.split(".txt")[0] + "_" + key + ".csv", index=False)
            # Visualize Mean Rewards
            if args.plot:
                ax1.plot(df_res['Mean_Rewards'], label=key)
                # ax1.plot([0, len(df_res['Mean_Rewards']) - 1], [0, df_res['Mean_Rewards'].iloc[-1]], 'r--', label='Linear Trend')

                # Plot Actions
                if args.run_one is not None:
                    ax2.plot(np.cumsum(df_res['Mean_AYA_Actions']) / (np.arange(len(df_res['Mean_AYA_Actions'])) + 1), label='AYA Actions')
                    ax2.plot(np.cumsum(df_res['Mean_Care_Actions']) / (np.arange(len(df_res['Mean_Care_Actions'])) + 1), label='Care Actions')
                    ax2.plot(np.cumsum(df_res['Mean_Game_Actions']) / (np.arange(len(df_res['Mean_Game_Actions'])) + 1), label='Game Actions')
                    ax2.plot(df_res['Mean_AYA_Actions'], label='AYA Cummean', alpha=0.2)
                    ax2.plot(df_res['Mean_Care_Actions'], label='Care Cummean', alpha=0.2)
                    ax2.plot(df_res['Mean_Game_Actions'], label='Game Cummean', alpha=0.2)
                    ax2.set_xlabel('Time Step')
                    ax2.set_ylabel('Mean Action Probability')
                    ax2.legend()
                    ax2.grid(True)
                    ax2.set_title('Mean Actions')
    if args.plot:
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Mean Reward')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('Mean Rewards')
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
