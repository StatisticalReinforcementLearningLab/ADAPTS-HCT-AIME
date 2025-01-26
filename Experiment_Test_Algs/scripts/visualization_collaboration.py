# 2024-09-27
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import json

path = "Experiment_Test_Algs/experiments/Test_Collaboration/"
env = "save_t0.5_1_1_0.5_s1_soft100_n25_rep1000_ME1"

# generate a list of p values
p1 = [0.25, 0.4]
p2 = [0.5]
p3 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

results1 = []
for p_1 in p1:
    res = []
    for p_2 in p2:
        for p_3 in p3:
            # if p_3 == 0.5:
                # name1 = f"Experiment_Test_Algs/results/{env}_MRT_{p_1}_{p_2}_{p_3}_true.csv"
            # else:
            name1 = f"Experiment_Test_Algs/results/{env}_MRT_{p_1}_{p_2}_{p_3}.csv"
            df = pd.read_csv(name1)
            end_reward = df.iloc[-1, 0] / 25
            res.append(end_reward)
    results1.append(res)
results1 = np.array(results1)

p1 = [0.5]
p2 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
p3 = [0.25, 0.75]

results2 = []
for p_3 in p3:
    res = []
    for p_2 in p2:
        for p_1 in p1:
            name1 = f"Experiment_Test_Algs/results/{env}_MRT_{p_1}_{p_2}_{p_3}.csv"
            df = pd.read_csv(name1)
            end_reward = df.iloc[-1, 0] / 25
            res.append(end_reward)
    results2.append(res)
results2 = np.array(results2)

def visualize_collaboration():
    print(results1.shape)
    print(results2.shape)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3))  # Create multiple subplots
    red1 = '#FF5733'  # Coral color for AYA Prob = 0.25
    red2 = '#C70039'  # Crimson color for AYA Prob = 0.75
    blue1 = '#1E90FF'  # DodgerBlue color for Game Prob = 0.25
    blue2 = '#00BFFF'  # DeepSkyBlue color for Game Prob = 0.75
    axs[0].plot(np.arange(0, 1.1, 0.1), results1[0, :], color=red1)
    axs[0].axvline(x=1, color=red1, linestyle='--')  # Add a vertical line at x = 1

    axs[1].plot(np.arange(0, 1.1, 0.1), results1[1, :], color=red2)
    axs[1].axvline(x=0, color=red2, linestyle='--')  # Add a vertical line at x = 1

    axs[0].set_title("(a) AYA Prob = 0.25, Care Prob = 0.5")
    axs[1].set_title("(b) AYA Prob = 0.75, Care Prob = 0.5")
    axs[0].set_ylabel("Average Weekly Reward")
    axs[0].set_xlabel("Game Prob")
    axs[1].set_ylabel("Average Weekly Reward")
    axs[1].set_xlabel("Game Prob")

    axs[2].plot(np.arange(0, 1.1, 0.1), results2[0, :], color=blue1)
    axs[2].axvline(x=0.6, color=blue1, linestyle='--')  # Add a vertical line at x = 1

    axs[3].plot(np.arange(0, 1.1, 0.1), results2[1, :], color=blue2)
    axs[3].axvline(x=0.5, color=blue2, linestyle='--')  # Add a vertical line at x = 1

    axs[2].set_title("(c) AYA Prob = 0.5, Game Prob = 0.25")
    axs[3].set_title("(d) AYA Prob = 0.5, Game Prob = 0.75")
    axs[2].set_xlabel("Care Prob")
    axs[2].set_ylabel("Average Weekly Reward")
    axs[3].set_xlabel("Care Prob")
    axs[3].set_ylabel("Average Weekly Reward")
    # axs[1, 0].set_title("Game Prob Results")
    # axs[1, 0].legend()
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(f"{path}/MRT_collaboration.pdf")
    plt.show()
    plt.close()

# iterate iteratively through all the files in the experiments folder
if __name__ == "__main__":
    visualize_collaboration()
