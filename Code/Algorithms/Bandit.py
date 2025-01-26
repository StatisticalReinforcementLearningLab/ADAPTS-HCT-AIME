from .Algorithm import Algorithm
from .Policy import Policy
import numpy as np
from .algutils import samplePolicy, incInverse, _softmaxWithTem
from utils import State

dumpy_state = State()
STATE_DIM = dumpy_state.STATE_DIM

class Bandit(Algorithm):
    def __init__(
        self,
        lambda_=1,
        sigma=1,
        update=1, # update every week by default
        maxN=None,
        incrUpdate=True,
        poolAcrossDyads=True,
        person = "Target",
        gamma = None,
        reward_type = "naive",
        delayed_weight = None,
        naive_weight = None,
        softmax = None,
        pure_exploration = False,
        save_coefficients = False
    ):
        super().__init__("Bandit", poolAcrossDyads, person, reward_type, delayed_weight, naive_weight, save_coefficients)
        # Parameters for Thompson Sampling
        # 0 for target, 1 for caregiver, 2 for game
        self.lambda_ = lambda_
        self.sigma = sigma
        self.var = None
        self.mean = None

        # overload the default softmax temperature if not None
        self.softmax = softmax
        # whether to only use softmax for exploration
        self.pure_exploration = pure_exploration

        # these are to make bandit run faster!
        self.update = update
        self.maxN = maxN
        self.incrUpdate = incrUpdate

        self.X = np.array([])
        self.raw_X = np.array([])
        self.y = np.array([])
        self.dim = self.getFeatureDim()

        self.variance = np.eye(self.dim) / self.lambda_

        self.theta = np.zeros(self.dim)

        # discount factor
        if gamma is None:
            if self.person == 'Target':
                self.gamma = 13/14
            elif self.person == 'Care':
                self.gamma = 6/7
            else:
                self.gamma = 0
        else:
            self.gamma = gamma

        self.w = np.zeros(self.dim)

    def __thompsonSampling(self):
        if self.gamma > 0:
            X = []
            y = []
            for j in range(self.X.shape[0]-1):  # j as l
                X.append(self.X[j, :])
                reward = self.y[j]
                raw_X_next = self.raw_X[j+1, :]
                possible_gain = []
                for action in [0, 1]:
                    row_after = self.getFeature(raw_X_next, action)
                    possible_gain.append(np.dot(self.theta, row_after))
                possible_gain = np.array(possible_gain)
                probOfActions = _softmaxWithTem(possible_gain)
                reward += self.gamma * (probOfActions @ possible_gain)
                y.append(reward)

            X = np.array(X)
            y = np.array(y)
        else:
            X = self.X
            y = self.y

        mean = (1 / (self.sigma**2) * np.dot(self.variance, X.T @ y)).reshape(-1)
        try:
            if not self.pure_exploration:
                self.w = np.random.multivariate_normal(self.gamma * self.w, (1-self.gamma**2) * self.variance)
            else:
                self.w = np.zeros(self.dim)
            self.theta = mean + self.w
            
            
        except np.linalg.LinAlgError:
            print(mean)
            print(self.variance)
            eigenvalues, _ = np.linalg.eig(self.variance)
            if np.all(eigenvalues >= 0):
                print("positive definite")
            else:
                print("not positive definite")
                # Find the minimum eigenvalue
            min_eigenvalue = np.min(eigenvalues)
            print("min eigenvalue: ", min_eigenvalue)
            return self.theta
        return self.theta

    def getFeature(self, state, action):
        oneHot = np.append([1], action)
        # use truncated state for better performance
        if self.person == "Target":
            # short_state = state[[0, 1, 2, 10]]
            short_state = state[[0, 1, 2, 5, 6, 10]]
        elif self.person == "Care":
            short_state = state[[0, 3, 4, 5, 6, 10]]
        else:
            short_state = state[[0, 2, 4, 5, 8, 9]]
        feature = np.outer(short_state, oneHot, out=None).T.reshape(-1)
        return feature
    def endSave(self):
        if self.save_coefficients:
            return self.theta

    def getFeatureDim(self):
        dummy_state = np.zeros(STATE_DIM)  
        return self.getFeature(dummy_state, 0).shape[0]

    def getPolicy(self, dyad) -> Policy:
        if dyad.startTime is None or dyad.currentTime is None:
            raise ValueError("The dyad has not been initialized.")
        elapsedTime = dyad.currentTime - dyad.startTime

        # Algorithm can not be used without data
        # In default, use a random to warm-up
        state = dyad.getState()
        # force a one week warm-up
        if self.X.shape[0] == 0 or (not self.poolAcrossDyads and elapsedTime.week <= 1):
            choice = np.random.randint(0, 2)
            return choice

        updateCoefficient = False
        if dyad.currentTime.isNewWeek() and dyad.currentTime.week%self.update == 0:
            updateCoefficient = True
        if updateCoefficient or self.theta is None:
            self.theta = self.__thompsonSampling()
        
        predictionReward = []
        for action in [0, 1]:
            x = self.getFeature(state, action)
            predictionReward.append(np.dot(self.theta, x))
        action = [0, 1][samplePolicy(np.array(predictionReward), self.softmax)]
        return action

    def addData(self, data: list, abs_time: int, person: str, idx: int):
        X = self.getFeature(data[0], data[1])
        raw_X = data[0]
        r = data[2]
        if self.X.shape[0] == 0:
            self.X = np.append(self.X, X).reshape(1, -1)
            self.y = np.append(self.y, r)
            self.raw_X = np.append(self.raw_X, raw_X).reshape(1, -1)
        else:
            self.X = np.vstack([self.X, X])
            self.y = np.append(self.y, r)
            self.raw_X = np.vstack([self.raw_X, raw_X])
        if self.maxN is not None:
            n = self.X.shape[0]
            self.X = self.X[(n-self.maxN):n, :]
            self.y = self.y[(n-self.maxN):n]
            self.raw_X = self.raw_X[(n-self.maxN):n, :]

        rank1Update = self.X[-1, :] / self.sigma
        self.variance = incInverse(self.variance, rank1Update, rank1Update)

if __name__ == "__main__":
    bandit = Bandit()
    bandit.addData((np.zeros(STATE_DIM), 0, 1), (np.zeros(STATE_DIM), 0, 1))
    print(bandit.getPolicy())
