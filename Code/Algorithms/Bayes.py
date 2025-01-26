from .Algorithm import Algorithm
from .Policy import Policy
import numpy as np
from .algutils import samplePolicy, incInverse, _softmaxWithTem
from utils import State
from .algargs import args

dumpy_state = State()
STATE_DIM = dumpy_state.STATE_DIM

class Bayes(Algorithm):
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
        pure_exploration = True,
        prior_weight = 1,
        simple_pool = False,
        save_coefficients = False
    ):
        super().__init__("Bayes", poolAcrossDyads, person, reward_type, delayed_weight, naive_weight, save_coefficients)
        # Parameters for Thompson Sampling
        # 0 for target, 1 for caregiver, 2 for game
        self.lambda_ = lambda_
        self.sigma = sigma

        self.simple_pool = simple_pool

        # overload the default softmax temperature if not None
        self.softmax = softmax
        self.prior_weight = prior_weight
        # whether to only use softmax for exploration
        self.pure_exploration = pure_exploration

        # these are to make bandit run faster!
        self.update = update
        self.maxN = maxN
        self.incrUpdate = incrUpdate

        self.dim = self.getFeatureDim()
        self.data_by_seq_idx = {}

        for seq_idx in range(args.n):
            self.data_by_seq_idx[seq_idx] = {
                "X": np.array([]), 
                "y": np.array([]), 
                "n": 1,
                "raw_X": np.array([]), 
                "variance": np.eye(self.dim) / self.lambda_,
                "variance_inv": np.eye(self.dim) * self.lambda_,
                "theta": np.zeros(self.dim),
                "post_mean": np.zeros(self.dim),
                "w": np.zeros(self.dim)
            }

        # self.X = np.array([])
        # self.raw_X = np.array([])
        # self.y = np.array([])
        

        # initialize prior
        self.prior_mean = np.zeros(self.dim)
        self.prior_cov = np.zeros((self.dim, self.dim)) 

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

    def __thompsonSampling(self, seq_idx: int):
        # update the theta for a specific dyad
        cur_data = self.data_by_seq_idx[seq_idx]
        if self.gamma > 0:
            X = []
            y = []
            for j in range(cur_data["X"].shape[0]-1):  # j as l
                X.append(cur_data["X"][j, :])
                reward = cur_data["y"][j]
                raw_X_next = cur_data["raw_X"][j+1, :]
                possible_gain = []
                for action in [0, 1]:
                    row_after = self.getFeature(raw_X_next, action)
                    possible_gain.append(np.dot(cur_data["theta"], row_after))
                possible_gain = np.array(possible_gain)
                probOfActions = _softmaxWithTem(possible_gain)
                reward += self.gamma * (probOfActions @ possible_gain)
                y.append(reward)

            X = np.array(X)
            y = np.array(y)
        else:
            X = cur_data["X"]
            y = cur_data["y"]

        mean = (1 / (self.sigma**2) * np.dot(cur_data["variance"], X.T @ y)).reshape(-1)
        try:
            if not self.pure_exploration:
                cur_data["w"] = np.random.multivariate_normal(self.gamma * cur_data["w"], (1-self.gamma**2) * cur_data["variance"])
            else:
                cur_data["w"] = np.zeros(self.dim)
            cur_data["theta"] = mean + cur_data["w"]
            
            
        except np.linalg.LinAlgError:
            print(mean)
            print(cur_data["variance"])
            eigenvalues, _ = np.linalg.eig(cur_data["variance"])
            if np.all(eigenvalues >= 0):
                print("positive definite")
            else:
                print("not positive definite")
                # Find the minimum eigenvalue
            min_eigenvalue = np.min(eigenvalues)
            print("min eigenvalue: ", min_eigenvalue)
            return cur_data["theta"]
        return cur_data["theta"]

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

    def getFeatureDim(self):
        dummy_state = np.zeros(STATE_DIM)
        return self.getFeature(dummy_state, 0).shape[0]
    def updatePrior(self):
        
        if len(self.data_by_seq_idx.keys()) == 0:
            return 
        
        self.prior_mean = np.zeros(self.dim)
        self.prior_cov = np.zeros((self.dim, self.dim))
        # if self.simple_pool:
        #     for seq_idx in self.data_by_seq_idx.keys():
        #         cur_data = self.data_by_seq_idx[seq_idx]
        #         self.prior_mean += cur_data["theta"]
        #         self.prior_cov += cur_data["variance"]
        #     self.prior_mean = self.prior_mean / len(self.data_by_seq_idx.keys())
        #     self.prior_cov = self.prior_cov / len(self.data_by_seq_idx.keys())
        #     return
    
        weights = np.zeros((self.dim, self.dim))
        for seq_idx in self.data_by_seq_idx.keys():
            cur_data = self.data_by_seq_idx[seq_idx]
            self.prior_mean += cur_data["variance_inv"] @ cur_data["theta"]
            weights += cur_data["variance_inv"]
        inv_weights = np.linalg.inv(weights)
        self.prior_mean =  inv_weights @ self.prior_mean

        cnt = 0
        for seq_idx in self.data_by_seq_idx.keys():
            cur_data = self.data_by_seq_idx[seq_idx]
            if cur_data["n"] <= 10:
                continue
            self.prior_cov += np.outer(cur_data["theta"] - self.prior_mean, cur_data["theta"] - self.prior_mean) # + np.linalg.inv(cur_data["variance"])/cur_data["n"]
            cnt += 1
        self.prior_cov = self.prior_cov / cnt + 0.01 * np.eye(self.dim)
        self.prior_cov_inv = np.linalg.inv(self.prior_cov)
        # self.prior_cov = np.eye(self.dim)
        return 
        try:
            if cnt < self.dim:
                self.prior_cov = np.linalg.inv(self.prior_cov / cnt + self.lambda_ * np.eye(self.dim))  #
            else:
                self.prior_cov = np.linalg.inv(self.prior_cov / cnt)  #
        except np.linalg.LinAlgError:
            self.prior_cov = np.linalg.inv(self.prior_cov / cnt + self.lambda_ * np.eye(self.dim))
        # self.prior_cov = np.linalg.inv(inv_weights @ self.prior_cov + inv_weights) #+ 1/self.lambda_ * np.eye(self.dim))

    def updateUserPosterior(self, seq_idx: int):
        cur_data = self.data_by_seq_idx[seq_idx]
        # print(self.prior_cov)
        # print(seq_idx, np.linalg.norm(cur_data["variance"]* cur_data["n"]), np.linalg.norm(self.prior_cov))
        post_mean = np.linalg.inv(cur_data["variance_inv"] + self.prior_cov_inv * self.prior_weight) @\
            (cur_data["variance_inv"] @ cur_data["theta"] + self.prior_cov_inv @ self.prior_mean * self.prior_weight) #/ (self.prior_weight + 1)
        # post_mean = cur_data["theta"]
        # post_mean = np.linalg.inv(cur_data["variance"] + self.prior_cov * self.prior_weight) @ (self.prior_cov @ self.prior_mean * self.prior_weight + cur_data["variance"] @ cur_data["theta"]) #/ (self.prior_weight + 1)
        cur_data["post_mean"] = post_mean

    def getPolicy(self, dyad) -> Policy:
        seq_idx = dyad.seq_idx
        if dyad.startTime is None or dyad.currentTime is None:
            raise ValueError("The dyad has not been initialized.")
        elapsedTime = dyad.currentTime - dyad.startTime

        # Algorithm can not be used without data
        # In default, use a random to warm-up
        state = dyad.getState()
        # force a one week warm-up
        if len(self.data_by_seq_idx.keys()) == 0 or (not self.poolAcrossDyads and elapsedTime.week <= 1):
            choice = np.random.randint(0, 2)
            return choice

        # update the theta for a specific dyad
        updateCoefficient = False
        if dyad.currentTime.isNewWeek() and dyad.currentTime.week%self.update == 0:
            updateCoefficient = True
        if updateCoefficient or self.data_by_seq_idx[seq_idx]["theta"] is None:
            self.data_by_seq_idx[seq_idx]["theta"] = self.__thompsonSampling(seq_idx)
        self.updatePrior()
        self.updateUserPosterior(seq_idx)
        
        predictionReward = []
        for action in [0, 1]:
            x = self.getFeature(state, action)
            predictionReward.append(np.dot(self.data_by_seq_idx[seq_idx]["post_mean"], x))
        action = [0, 1][samplePolicy(np.array(predictionReward), self.softmax)]
        return action

    def addData(self, data: list, abs_time: int, person: str, seq_idx: int):
        X = self.getFeature(data[0], data[1])
        self.data_by_seq_idx[seq_idx]["n"] += 1
        raw_X = data[0]
        r = data[2]
        if seq_idx not in self.data_by_seq_idx.keys():
            self.data_by_seq_idx[seq_idx] = self.getNewDict()
        if self.data_by_seq_idx[seq_idx]["X"].shape[0] == 0:
            self.data_by_seq_idx[seq_idx]["X"] = np.append(self.data_by_seq_idx[seq_idx]["X"], X).reshape(1, -1)
            self.data_by_seq_idx[seq_idx]["y"] = np.append(self.data_by_seq_idx[seq_idx]["y"], r)
            self.data_by_seq_idx[seq_idx]["raw_X"] = np.append(self.data_by_seq_idx[seq_idx]["raw_X"], raw_X).reshape(1, -1)
        else:
            self.data_by_seq_idx[seq_idx]["X"] = np.vstack([self.data_by_seq_idx[seq_idx]["X"], X])
            self.data_by_seq_idx[seq_idx]["y"] = np.append(self.data_by_seq_idx[seq_idx]["y"], r)
            self.data_by_seq_idx[seq_idx]["raw_X"] = np.vstack([self.data_by_seq_idx[seq_idx]["raw_X"], raw_X])
        if self.maxN is not None:
            n = self.data_by_seq_idx[seq_idx]["X"].shape[0]
            self.data_by_seq_idx[seq_idx]["X"] = self.data_by_seq_idx[seq_idx]["X"][(n-self.maxN):n, :]
            self.data_by_seq_idx[seq_idx]["y"] = self.data_by_seq_idx[seq_idx]["y"][(n-self.maxN):n]
            self.data_by_seq_idx[seq_idx]["raw_X"] = self.data_by_seq_idx[seq_idx]["raw_X"][(n-self.maxN):n, :]

        rank1Update = self.data_by_seq_idx[seq_idx]["X"][-1, :] / self.sigma
        self.data_by_seq_idx[seq_idx]["variance_inv"] = self.data_by_seq_idx[seq_idx]["variance_inv"] + np.outer(rank1Update, rank1Update)
        self.data_by_seq_idx[seq_idx]["variance"] = incInverse(self.data_by_seq_idx[seq_idx]["variance"], rank1Update, rank1Update)

if __name__ == "__main__":
    bandit = Bandit()
    bandit.addData((np.zeros(STATE_DIM), 0, 1), (np.zeros(STATE_DIM), 0, 1))
    print(bandit.getPolicy())
