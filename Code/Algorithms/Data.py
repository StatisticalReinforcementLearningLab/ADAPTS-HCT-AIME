import numpy as np
from numpy.typing import NDArray
from .algutils import integer_to_binary

STATE_DIM = 10

class Data:
    def __init__(self, maxN = None):
        self.maxN = maxN

    def addData(self, data):
        raise NotImplementedError("Subclass must implement abstract method")

class BayesData(Data):
    # We need a user-specific data structure
    # Bandit only needs tuple like (s,a,r)
    # s: state, a: action, r: reward
    # Action is modeled in binary form as integer
    # E.g. 000 for do nothing, 001 for act to target
    # 011 for act to both target and caregiver 101 for act to both target and game

    def __init__(self, maxN = None):
        super().__init__(maxN)
        self.data_by_idx = {}
        # self.X = np.array([])
        # self.y = np.array([])
        

    def addData(self, data, idx: int):
        # s: NDArray, a: int, r):
        # self.data.append((s, a, r))
        if idx not in self.data_by_idx:
            self.data_by_idx[idx] = {"X": np.array([]), "y": np.array([])}
        # self.data_by_idx[idx].append(data)
        X = self.getFeature(data[0], data[1])
        r = data[2]
        if self.data_by_idx[idx]["X"].shape[0] == 0:
            self.data_by_idx[idx]["X"] = np.append(self.data_by_idx[idx]["X"], X).reshape(1, -1)
            self.data_by_idx[idx]["y"] = np.append(self.data_by_idx[idx]["y"], r)
        else:
            self.data_by_idx[idx]["X"] = np.vstack([self.data_by_idx[idx]["X"], X])
            self.data_by_idx[idx]["y"] = np.append(self.data_by_idx[idx]["y"], r)
        if self.maxN is not None:
            n = self.data_by_idx[idx]["X"].shape[0]
            self.data_by_idx[idx]["X"] = self.data_by_idx[idx]["X"][(n-self.maxN):n, :]
            self.data_by_idx[idx]["y"] = self.data_by_idx[idx]["y"][(n-self.maxN):n]

    def isEmpty(self, idx: int):
        if idx not in self.data_by_idx:
            return True
        return self.data_by_idx[idx]["X"].shape[0] == 0

    def getFeature(self, state, action):
        # construct feature from data
        # this feature has interaction between each action and state!
        oneHot = np.append([1], integer_to_binary(action, 3))
        feature = np.outer(state, oneHot, out=None).T.reshape(-1)
        return feature

    def getFeatureDim(self):
        dummy_state = np.zeros(STATE_DIM)  
        return self.getFeature(dummy_state, 0).shape[0]
     

class BanditData:
    # Bandit only needs tuple like (s,a,r)
    # s: state, a: action, r: reward
    # Action is modeled in binary form as integer
    # E.g. 000 for do nothing, 001 for act to target
    # 011 for act to both target and caregiver 101 for act to both target and game

    def __init__(self, maxN = None):
        self.X = np.array([])
        self.y = np.array([])
        super().__init__(maxN)

    def addData(self, data):
        # s: NDArray, a: int, r):
        # self.data.append((s, a, r))
        X = self.getFeature(data[0], data[1])
        r = data[2]
        if self.X.shape[0] == 0:
            self.X = np.append(self.X, X).reshape(1, -1)
            self.y = np.append(self.y, r)
        else:
            self.X = np.vstack([self.X, X])
            self.y = np.append(self.y, r)
        if self.maxN is not None:
            n = self.X.shape[0]
            self.X = self.X[(n-self.maxN):n, :]
            self.y = self.y[(n-self.maxN):n]

    def isEmpty(self):
        return self.X.shape[0] == 0

    def getFeature(self, state, action):
        # construct feature from data
        # this feature has interaction between each action and state!
        oneHot = np.append([1], integer_to_binary(action, 3))
        feature = np.outer(state, oneHot, out=None).T.reshape(-1)
        return feature

    def getFeatureDim(self):
        dummy_state = np.zeros(STATE_DIM)  
        return self.getFeature(dummy_state, 0).shape[0]
        

class RLSVIHData:
    def __init__(self, lenOfTimeBlock, maxN = None):
        self.lenOfTimeBlock = lenOfTimeBlock
        # if see a observation tuple as an element
        self.data = [
            [np.array([]), np.array([])] for _ in range(self.lenOfTimeBlock)
        ]  # lenOfTimeBlock x l
        super().__init__(maxN)
    def isEmpty(self):
        ret = False
        for x in self.data:
            ret = ret or (x[0].shape[0] == 0)
        return ret

    def reshapedData(self):
        ret = []
        for i, x in enumerate(self.data):
            if i % self.lenOfTimeBlock == 0:
                ret.append([])
            ret[-1].append(x)
        return ret

    def getFeature(self, state, action):
        # construct feature from data
        # this feature has interaction between each action and state!
        oneHot = integer_to_binary(action, 3)
        # feature = np.outer(state, oneHot, out=None).reshape(-1)
        feature = np.concatenate([state, oneHot[0] * state, oneHot[1] * state, oneHot[2] * state, oneHot[0] * oneHot[1] * state, oneHot[0] * oneHot[2] * state])
        return feature
    def getFeatureDim(self):
        dummy_state = np.zeros(STATE_DIM)
        return self.getFeature(dummy_state, 0).shape[0]

    def addData(self, s: NDArray, a: int, r, h):
        if self.data[h][0].shape[0] == 0:
            self.data[h][0] = np.append(self.data[h][0], self.getFeature(s, a))
            self.data[h][1] = np.append(self.data[h][1], r)
            self.data[h][0] = self.data[h][0].reshape((1, -1))
        else:
            self.data[h][0] = np.vstack([self.data[h][0], self.getFeature(s, a)])
            self.data[h][1] = np.vstack([self.data[h][1], r])


class DyadicRLData:
    lenOfTimeBlock = 14

    def __init__(self, maxN = None):
        # if see a observation tuple as an element
        self.data_high = [np.array([]), np.array([])]  # 1 x l
        self.data_low = [
            [np.array([]), np.array([])] for _ in range(self.lenOfTimeBlock)
        ]  # lenOfTimeBlock x l
        super().__init__(maxN)
    def isEmpty(self, low=True):
        if low:
            ret = False
            for x in self.data_low:
                ret = ret or (x[0].shape[0] == 0)
            return ret
        else:
            return self.data_high[0].shape[0] == 0

    def reshapedData(self, low=True):
        if low:
            ret = []
            for i, x in enumerate(self.data_low):
                if i % self.lenOfTimeBlock == 0:
                    ret.append([])
                ret[-1].append(x)
            return ret
        else:
            ret = []
            for i, x in enumerate(self.data_high):
                ret.append([x])
            return ret

    def getFeature(self, state, action):
        # construct feature from data
        # this feature has interaction between each action and state!
        oneHot = integer_to_binary(action, 3)
        # feature = np.outer(state, oneHot, out=None).reshape(-1)
        feature = np.concatenate([state, oneHot[0] * state, oneHot[1] * state, oneHot[2] * state, oneHot[0] * oneHot[1] * state, oneHot[0] * oneHot[2] * state])
        return feature

    def addDataLow(self, s: NDArray, a: int, r, h):
        if self.data_low[h][0].shape[0] == 0:
            self.data_low[h][0] = np.append(self.data_low[h][0], self.getFeature(s, a))
            self.data_low[h][1] = np.append(self.data_low[h][1], r)
            self.data_low[h][0] = self.data_low[h][0].reshape((1, -1))
        else:
            self.data_low[h][0] = np.vstack(
                [self.data_low[h][0], self.getFeature(s, a)]
            )
            self.data_low[h][1] = np.vstack([self.data_low[h][1], r])

    def addDataHigh(self, s: NDArray, a: int, r):
        # self.data_high[0].append((s, a, r))
        if self.data_high[0].shape[0] == 0:
            self.data_high[0] = np.append(self.data_high[0], self.getFeature(s, a))
            self.data_high[1] = np.append(self.data_high[1], r)
            self.data_high[0] = self.data_high[0].reshape((1, -1))
        else:
            self.data_high[0] = np.vstack([self.data_high[0], self.getFeature(s, a)])
            self.data_high[1] = np.vstack([self.data_high[1], r])


"""
class FullRLData():
    def __init__(self):
        self.data = []
        self.lambda_ = 1
        self.sigma = 1

    def addData(self, s: NDArray, a: int, r, whichDyad:int):
        if whichDyad >= len(self.data):
            self.data.append([])
        self.data[whichDyad].append((s, a, r))

    def RLSVI(self, horizen:int):
        # Iterate from horizen to 1
        theta = []
        for h in range(horizen, 0, -1):
            X = []
            y = []
            for j,ds in enumerate(self.data):
                if j >= h:
                    break
                d = ds[h]
                oneHot = integer_to_binary(d[1], 3)
                row = np.concatenate([d[0], oneHot])
                X.append(row)
                reward = d[2]
                if h != horizen:
                    d_after = self.data[j+1]
                    theta_after = theta[-1]
                    possible_gain = []
                    if h % WeekLength == 0:
                        actions = [0, 1, 2, 3, 4, 5, 6, 7]
                    elif h % DayLength == 0:
                        actions = [0, 1, 2, 3]
                    else:
                        actions = [0, 1]
                    for action in actions:
                        oneHot_after = integer_to_binary(action, 3)
                        row_after = np.concatenate([d_after[0], oneHot_after])
                        possible_gain.append(np.dot(theta_after, row_after))
                    reward += np.max(possible_gain)
                y.append(reward)
            X = np.array(X)
            y = np.array(y)
            variance = np.linalg.inv(self.lambda_ * np.eye(X.shape[1]) + 1/(self.sigma**2) * np.dot(X.T, X))
            mean = 1/(self.sigma**2) * np.dot(variance, X.T@y)
            theta.append(np.random.multivariate_normal(mean, variance))

    def makeXAndy(self, len:int=-1):
        X = []
        y = []
        for i, d in enumerate(self.data):
            if len != -1 and i >= len:
                break
            oneHot = integer_to_binary(d[1], 3)
            row = np.concatenate([d[0], oneHot])
            X.append(row) 
            y.append(d[2])
        return np.array(X), np.array(y)
"""


if __name__ == "__main__":
    pass
