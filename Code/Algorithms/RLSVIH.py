from .Algorithm import Algorithm
import numpy as np
from .algutils import samplePolicy, _softmaxWithTem, incInverse, integer_to_binary
from .Policy import Policy
from copy import deepcopy
from utils import STATE_DIM

class RLSVIH(Algorithm):
    def __init__(self, lambda_=1, sigma=1, update=1, poolAcrossDyads=True, person="Target", reward_type = "naive", delayed_weight = None, naive_weight = None, gamma = None):
        super().__init__("DyadicRL", poolAcrossDyads, person, reward_type, delayed_weight, naive_weight)
        self.lambda_ = lambda_
        self.sigma = sigma
        self.person = person
        # self.maxN = maxN

        if self.person == "Game":
            raise NotImplementedError("Not implemented for Game agent!")    
        self.lenOfTimeBlock = 14 if self.person == "Target" else 7
        self.update = update  # update theta's every `update` weeks

        self.thetas = None
        self.X_raw = [np.array([]) for _ in range(self.lenOfTimeBlock)]
        self.X = [np.array([]) for _ in range(self.lenOfTimeBlock)]
        self.y = [np.array([]) for _ in range(self.lenOfTimeBlock)]
        self.variance = [np.eye(self.getFeatureDim()) / self.lambda_ for _ in range(self.lenOfTimeBlock)]

    def getFeature(self, state, action):
        oneHot = np.append([1], action)
        if self.person == "Target":
            # short_state = state[[0, 1, 2, 10]]
            short_state = state[[0, 1, 2, 5, 6, 10]]
        elif self.person == "Care":
            short_state = state[[0, 3, 4, 5, 6, 10]]
        else:
            short_state = state[[0, 2, 4, 5, 8, 9]]
        feature = np.outer(short_state, oneHot, out=None).T.reshape(-1)
        return feature

    def getFeatureDim(self):
        dummy_state = np.zeros(STATE_DIM)  
        return self.getFeature(dummy_state, 0).shape[0]

    def RLSVI(self):
        thetas = []
        for h in range(self.lenOfTimeBlock, 0, -1):
            h_index = h - 1
            X = deepcopy(self.X[h_index])
            y = deepcopy(self.y[h_index])

            if h != self.lenOfTimeBlock:
                for j in range(X.shape[0]):
                    if j < self.X[h_index + 1].shape[0]:
                        d_after = self.X_raw[h_index + 1][j, :]
                        theta_after = thetas[-1]
                        possible_gain = []
                        for action in [0, 1]:
                            row_after = self.getFeature(d_after, action)
                            possible_gain.append(np.dot(theta_after, row_after))
                        possible_gain = np.array(possible_gain)
                        probOfActions = _softmaxWithTem(possible_gain)
                        y[j] += probOfActions @ possible_gain

            variance = self.variance[h_index]
            mean = (1 / (self.sigma**2) * np.dot(variance, X.T @ y)).reshape(-1)
            try:
                # print(np.dot(variance, X.T @ y).shape, X.shape, y.shape, mean.shape, variance.shape)
                thetas.append(np.random.multivariate_normal(mean, variance))
            except np.linalg.LinAlgError:
                print(f"LinAlgError at h={h}")
                print("mean:", mean)
                print("variance:", variance)
                eigenvalues = np.linalg.eigvals(variance)
                print("min eigenvalue:", np.min(eigenvalues))
                return None
        return thetas

    def getPolicy(self, dyad) -> int:
        if dyad.startTime is None or dyad.currentTime is None:
            raise ValueError("The subject has not been initialized.")
        elapsedTime = dyad.currentTime - dyad.startTime
        h_index = elapsedTime.getAbsoluteTime() % 14 if self.person == "Target" else elapsedTime.getAbsoluteTime() % 7
        if (self.X[-1].shape[0] == 0) or (self.thetas is None and not elapsedTime.isNewWeek()):
            action = np.random.randint(0, 2)
            return action
        
        if elapsedTime.isNewWeek():
            if elapsedTime.week % self.update == 0 or self.thetas is None:
                thetas = self.RLSVI()
                if thetas is not None:
                    self.thetas = thetas

        predictionReward = []
        actions = [0, 1]

        state = dyad.getState()
        for action in actions:
            x = self.getFeature(state, action)
            predictionReward.append(self.thetas[h_index] @ x)
        action = actions[samplePolicy(np.array(predictionReward))]
        return action

    def addData(self, data: list, abs_time: int, person: str, idx: int):
        h_index = abs_time % 14 if self.person == "Target" else abs_time % 7
        X = self.getFeature(data[0], data[1])
        X_raw = data[0]
        r = data[2]

        if self.X[h_index].shape[0] == 0:
            self.X[h_index] = X.reshape(1, -1)
            self.y[h_index] = np.array([r])
            self.X_raw[h_index] = X_raw.reshape(1, -1)
        else:
            self.X[h_index] = np.vstack([self.X[h_index], X])
            self.y[h_index] = np.append(self.y[h_index], r)
            self.X_raw[h_index] = np.vstack([self.X_raw[h_index], X_raw])
        rank1Update = X / self.sigma
        self.variance[h_index] = incInverse(self.variance[h_index], rank1Update, rank1Update)

if __name__ == "__main__":  
    rl = RLSVIH()
    rl.addData((np.zeros(FEATURE_DIM), 0, 1), 1)
    rl.addData((np.zeros(FEATURE_DIM), 0, 1), 2)
    rl.addData((np.zeros(FEATURE_DIM), 0, 1), 3)
    rl.addData((np.zeros(FEATURE_DIM), 0, 1), 4)
    rl.addData((np.zeros(FEATURE_DIM), 0, 1), 5)
    rl.addData((np.zeros(FEATURE_DIM), 0, 1), 6)
