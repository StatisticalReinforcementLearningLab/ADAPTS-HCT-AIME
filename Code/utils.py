import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

STATE_DIM = 11


class State:
    def __init__(self):
        self.adh = [0]
        self.distress = [0]
        self.ayaBurden = [0]
        self.careBurden = [0]
        self.relationship = [0]
        self.weekA = [0]
        self.dayA = [0]
        self.tarA = [0]
        self.wAveAdh = [0]
        self.wAveDistress = [0]
        self.time_of_day = [0]

        self.gamma_adh = 13/14
        self.gamma_distress = 6/7
        

        self.STATE_DIM = STATE_DIM

    def getState(self):
        cur_state = np.array([
            1, self.adh[-1], self.ayaBurden[-1], self.distress[-1],
            self.careBurden[-1], self.relationship[-1], self.weekA[-1],
            self.dayA[-1], self.wAveAdh[-1], self.wAveDistress[-1], self.time_of_day[-1]
        ])
        assert len(cur_state) == self.STATE_DIM
        return cur_state

    def updateAveAdh(self):
        self.wAveAdh.append(self.wAveAdh[-1] * self.gamma_adh + self.adh[-1])
    def updateAveDistress(self):
        self.wAveDistress.append(self.wAveDistress[-1] * self.gamma_distress + self.distress[-1])

def generate_discretization(s, nq = 4):
    dis_s = pd.qcut(s, q=nq, labels=False, duplicates='drop')
    quantiles = np.quantile(s, np.linspace(0, 1, nq + 1))
    print(f"Feature values: {s[:10]}")  # Print first 10 values of the feature
    print(f"Quantiles: {quantiles}")
    print(f"Discretized values: {dis_s[:10]}")  # Print first 10 discretized values
    return dis_s, quantiles
    
def apply_discretization(s, quantiles):
    dis_s = pd.cut([s], bins=quantiles, labels=False, include_lowest=True, duplicates='drop')
    return dis_s[0]


class opt_policy:
    def __init__(self, Q, all_actions, horizon, quantiles, dis_ind, w = None, subset_feature = None):
        self.Q = Q
        self.all_actions = list(all_actions)
        self.horizon = horizon
        self.quantiles = quantiles
        self.dis_ind = dis_ind
        self.w = w
        self.feature_fn = getFeatureSingleAgent
        self.subset_feature = subset_feature
    def policy(self, person, s, t):
        if self.w is not None:
            #SingleAgent
            return self.linear_policy(person, s, t)
        else:
            #Offline value iteration
            return self.q_based_policy(person, s, t)
    def q_based_policy(self, person, s, t):
        # t is the absolute time
        # print("in q_based_policy")
        h = t % self.horizon
        s = s[:self.subset_feature]
        for i in range(len(self.dis_ind)):
            if self.dis_ind[i] == 1:
                dis_s = apply_discretization(s[i], quantiles=self.quantiles[i])
                s[i] = dis_s        
        if (tuple(s), tuple(self.all_actions[0])) not in self.Q[h].keys():
            print("State not found in Q-table. Choosing random action.")
            print(tuple(s), tuple(self.all_actions[0]))
            idx = np.random.choice(range(len(self.all_actions)))
            return self.all_actions[idx]
        tmp_Q = np.array([self.Q[h][(tuple(s), tuple(a))] for a in self.all_actions])
        # print(self.Q[h].keys())
        for i, a in enumerate(self.all_actions):
            # print(a)
            # aa = integer_to_binary(a[0], 3)
            if person == "Target" and a[1] + a[2] > 0:
                tmp_Q[i] = -np.inf
            if person == "Care" and a[0] + a[2] > 0:
                tmp_Q[i] = -np.inf
            if person == "Game" and a[0] + a[1] > 0:
                tmp_Q[i] = -np.inf
        return self.all_actions[np.argmax(tmp_Q)]
    def linear_policy(self, person, s, t):
        values = []
        for a in self.all_actions:
            if person == "Target" and (a[1] + a[2]) > 0:
                values.append(-np.inf)
                continue
            if person == "Care" and (a[0] + a[2]) > 0:
                values.append(-np.inf)
                continue
            if person == "Game" and (a[0] + a[1]) > 0:
                values.append(-np.inf)
                continue

            phi_sa = self.feature_fn(s, a)
            q_val = np.dot(self.w, phi_sa)
            values.append(q_val)

        best_idx = np.argmax(values)
        return self.all_actions[best_idx]

def _Bernoulli(p):
    return np.random.binomial(1, p)


def _standard_logistic(x):
    return 1 / (1 + np.exp(-x))


def Bernoulli_logistic(p):
    return _Bernoulli(_standard_logistic(p))


def integer_to_binary(n, length):
    return np.array([int(x) for x in bin(n)[2:].zfill(length)])


def binary_to_integer(b):
    return int("".join(str(x) for x in b), 2)

def getFeature(state, action):
    # construct feature from data
    # this feature has interaction between each action and state!
    oneHot = integer_to_binary(action, 3)
    feature = np.outer(state, oneHot, out=None).reshape(-1)
    return feature

def getFeatureSingleAgent(state, action):
    #Same as getFeature, but modified to match the SingleAgent getFeature function. 
    oneHot = np.append([1], action)            #Add intercept, making it length 4
    feature = np.outer(state, oneHot).reshape(-1)
    return feature

def compute_mean_var_burden(coeOfBurden, constantOfBurden, bNoise):
    burden = 0
    all_burdens = []
    for week in range(1000):
        a2 = np.random.choice([0, 1])
        for j in range(14):
            all_burdens.append(burden)
            a1 = np.random.choice([0, 1])
            burden =  constantOfBurden + burden* coeOfBurden[0] + coeOfBurden[1] * a1 + coeOfBurden[2] * a2 + np.random.normal(0, bNoise)
        
    return np.mean(all_burdens), np.var(all_burdens)


def compute_feature_variances(data, feature_names):
    """Compute variances of state features and return the results."""
    variances = np.var(data, axis=0)
    print("Feature Variances:")
    for feature_name, variance in zip(feature_names, variances):
        print(f"{feature_name}: {variance:.4f}")
    return variances


def plot_feature_distributions(data, feature_names, output_path="feature_distributions.png"):
    """Plot and save feature distributions."""
    plt.figure(figsize=(15, 10))
    for i, feature_name in enumerate(feature_names):
        plt.subplot(3, 4, i + 1)
        plt.hist(data[:, i], bins=20, alpha=0.7, color='blue')
        plt.title(feature_name)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved as {output_path}")


def discretize_features(data, dis_ind, n_bins):
    """Discretize features based on the indicator array for constructing the optimal policy."""
    quantiles = []
    for i, flag in enumerate(dis_ind):
        if flag == 1:  #Check if the feature needs discretization
            dis_s, quantile = generate_discretization(data[:, i], n_bins)
            data[:, i] = dis_s
            quantiles.append(quantile)
        else:
            quantiles.append(0)
    return data, quantiles


def analyze_discretized_states(data, filter_states, feature_names, output_path="state_distributions.png"):
    """Count and plot the distribution of num samples per discretized state."""
    filter_indices = [i for i, keep in enumerate(filter_states) if keep == 1]
    discretized_states = [tuple(entry[filter_indices]) for entry in data]

    discretized_state_counts = Counter(discretized_states)
 
    state_counts = list(discretized_state_counts.values())
    print("Summary statistics for number of samples per state:")
    print(f"Min samples in a state: {min(state_counts)}")
    print(f"Max samples in a state: {max(state_counts)}")
    print(f"Mean samples per state: {np.mean(state_counts):.2f}")
    print(f"Median samples per state: {np.median(state_counts):.2f}")

    plt.figure(figsize=(10, 6))
    plt.hist(state_counts, bins=70, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Distribution of Number of Samples per State")
    plt.xlabel("Number of Samples per State")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved as {output_path}")

    return discretized_state_counts

def compute_rare_state_similarities(discretized_state_counts, rare_threshold=5):
    """Get other states that are similar to the rarely occuring states."""
    rare_states = [state for state, count in discretized_state_counts.items() if count < rare_threshold]
    all_states = list(discretized_state_counts.keys())

    rare_to_all_similarities = {}
    for rare_state in rare_states:
        similarities = []
        for state in all_states:
            distance = np.linalg.norm(np.array(rare_state) - np.array(state))  # Euclidean distance
            similarities.append((state, distance))
        rare_to_all_similarities[tuple(rare_state)] = similarities

    pairwise_distances = [dist for similarities in rare_to_all_similarities.values() for _, dist in similarities]
    return rare_states, pairwise_distances


def plot_pairwise_similarities(pairwise_distances, output_path="pairwise_similarities.png"):
    """Plot distribution of pairwise similarities of rare states."""
    plt.figure(figsize=(8, 6))
    plt.hist(pairwise_distances, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Pairwise Similarities (Rare to All States)")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved as {output_path}")
    threshold = np.percentile(pairwise_distances, 90)
    print(f"Suggested similarity threshold: {threshold:.2f}")
    return threshold
