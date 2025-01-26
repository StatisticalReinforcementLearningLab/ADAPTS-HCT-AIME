import numpy as np
import scipy.special as sp
from scipy.linalg.blas import dger
from .algargs import args

# Incremental implementation: refer to
# https://timvieira.github.io/blog/post/2021/03/25/fast-rank-one-updates-to-matrix-inverse/
# def incInverse(B, u, v):
#     # Warning: `overwrite_a=True` silently fails when B is not an order=F array!
#     assert B.flags["F_CONTIGUOUS"]
#     Bu = B @ u
#     alpha = -1 / (1 + v.T @ Bu)
#     dger(alpha, Bu, v.T @ B, a=B, overwrite_a=1)
# B -= B @ u @ v.T @ B / (1 + v.T @ B @ u)

def incInverse(B,u,v):
    return B - np.outer(B @ u, v.T @ B) / (1 + v.T @ B @ u)

def samplePolicy(possibleGain: np.ndarray, softmax_tem = None):
    # Calculate the softmax probabilities for each possible gain.
    # Softmax is used to convert the gains into probabilities that sum to 1.
    if softmax_tem is None:
        prob = sp.softmax(possibleGain * args.softmax_tem)
    else:
        prob = sp.softmax(possibleGain * softmax_tem)

    # Randomly select an index based on the calculated probabilities.
    # This simulates making a choice based on the given probabilities.
    return np.random.choice(range(len(possibleGain)), p=prob)


def _softmaxWithTem(x: np.ndarray):
    # tem for temperature 4 now
    p = sp.softmax(x * args.softmax_tem)
    return p


def integer_to_binary(n, length = 3):
    return np.array([int(x) for x in bin(n)[2:].zfill(length)])
    # first dim: game action, second dim: care action, third dim: AYA action

def getFeature(state, action):
    # construct feature from data
    # this feature has interaction between each action and state!
    oneHot = integer_to_binary(action, 3)
    feature = feature = np.concatenate(
        [
            state,
            oneHot[0] * state,
            oneHot[1] * state,
            oneHot[2] * state,
            oneHot[0] * oneHot[1] * state,
            oneHot[0] * oneHot[2] * state,
        ]
    )
    return feature


WeekLength = 14
DayLength = 2



def RLSVI(horizon: int, data, lambda_, sigma):
    # Iterate from horizon to 1
    theta = []
    for h in range(horizon, 0, -1):
        X = []
        y = []
        h_index = h - 1
        for j in range(data[h_index][0].shape[0]):  # j as l
            X.append(data[h_index][0][j])
            reward = data[h_index][1][j]
            if h != horizon and j < len(data[h_index + 1][0]):
                d_after = data[h_index + 1][0][j]
                theta_after = theta[-1]
                possible_gain = []
                if h_index % WeekLength == 0:
                    actions = [0, 1, 2, 3, 4, 5, 6, 7]
                elif h_index % DayLength == 0:
                    actions = [0, 2, 4, 6]
                else:
                    actions = [0, 4]
                for action in actions:
                    row_after = getFeature(d_after[:8], action)
                    # row_after = getFeature(d_after, action)\
                    possible_gain.append(np.dot(theta_after, row_after))
                # possible_gain = np.asarray(possible_gain)
                probOfActions = _softmaxWithTem(possible_gain)
                reward += probOfActions @ possible_gain
            y.append(reward)

        X = np.array(X)
        y = np.array(y)
        variance = np.linalg.inv(
            lambda_ * np.eye(X.shape[1]) + 1 / (sigma**2) * np.dot(X.T, X)
        )
        mean = (1 / (sigma**2) * np.dot(variance, X.T @ y)).reshape(-1)
        try:
            theta.append(np.random.multivariate_normal(mean, variance))
        except np.linalg.LinAlgError:
            print(mean)
            print(variance)
            eigenvalues, _ = np.linalg.eig(variance)
            if np.all(eigenvalues >= 0):
                print("positive definite")
            else:
                print("not positive definite")
                # Find the minimum eigenvalue
            min_eigenvalue = np.min(eigenvalues)
            print("min eigenvalue: ", min_eigenvalue)
            return None
    return theta
