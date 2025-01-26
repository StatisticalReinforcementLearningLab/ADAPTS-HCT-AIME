from .Algorithm import Algorithm
from .Policy import Policy
import numpy as np
from .algutils import samplePolicy, incInverse, _softmaxWithTem
from utils import State

# from args import args


dumpy_state = State()
STATE_DIM = dumpy_state.STATE_DIM

class RewardLearning(Algorithm):
    def __init__(
        self,
        lambda_=1,
        sigma=1,
        update=1, # update every week by default
        maxN=None, # deprecated!
        incrUpdate=True,
        poolAcrossDyads=True,
        person = "Target",
        gamma = None,
        reward_type = "naive",
        delayed_weight = None,
        naive_weight = None,
        softmax = None,
        rwd_lambda = 0.75,
        pure_exploration = False,
        save_param = False,
        next_week_weight = 0.5,
        care_mediator_weight = 0.0,
        care_learning = True,
        use_prior = False,
        save_coefficients = False
    ):
        super().__init__("Bandit", poolAcrossDyads, person, reward_type, delayed_weight, naive_weight, save_coefficients)
        # Parameters for Thompson Sampling
        # 0 for target, 1 for caregiver, 2 for game
        self.lambda_ = lambda_
        self.sigma = sigma
        self.var = None
        self.mean = None
        self.next_week_weight = next_week_weight
        self.save_param = save_param
        self.care_mediator_weight = care_mediator_weight
        self.care_learning = care_learning
        self.use_prior = use_prior

        # overload the default softmax temperature if not None
        self.softmax = softmax
        # whether to only use softmax for exploration
        self.pure_exploration = pure_exploration

        # these are to make bandit run faster!
        self.update = update
        self.maxN = maxN
        self.incrUpdate = incrUpdate

        self.rwd_update_count = 0

        self.X = np.array([])
        self.raw_X = np.array([])
        self.y = np.array([])
        self.dim = self.getFeatureDim()

        self.rwd_X = np.array([])
        self.rwd_y = np.array([])
        self.rwd_dim = self.getRewardFeatureDim()

        self.theta_rwd = np.zeros(self.rwd_dim)
        self.theta = np.zeros(self.dim)

        self.rwd_lambda = rwd_lambda
        self.rwd_theta_prior = np.zeros(self.rwd_dim)
        if self.person == "Target":
            if self.use_prior:
                self.rwd_theta_prior = np.array([0, 1, -0.5, 1])
            self.rwd_prior_var = np.eye(self.rwd_dim) / self.rwd_lambda
            self.rwd_variance = np.eye(self.rwd_dim) / 0.1
            self.rwd_variance_inv = np.eye(self.rwd_dim) * 0.00001
        elif self.person == "Care":
            if self.use_prior:
                self.rwd_theta_prior = np.array([1, -1, -1, 1, -1, -1])
            self.rwd_prior_var = np.eye(self.rwd_dim) / self.rwd_lambda
            self.rwd_variance = np.eye(self.rwd_dim) / 0.1
            self.rwd_variance_inv = np.eye(self.rwd_dim) * 0.00001
        else:
            if self.use_prior:
                self.rwd_theta_prior = np.array([1, -0.5, -0.5, 1, -0.5, -0.5, -0.5, 0])
            self.rwd_prior_var = np.eye(self.rwd_dim) / self.rwd_lambda
            self.rwd_variance = np.eye(self.rwd_dim) / 0.1
            self.rwd_variance_inv = np.eye(self.rwd_dim) * 0.00001

        self.variance = np.eye(self.dim) / self.lambda_
        # self.rwd_variance = np.eye(self.rwd_dim) / self.lambda_

        # if self.reward_type != "naive":
        #     raise NotImplementedError("Reward learning algorithm is not implemented for non-naive reward types")

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
                if self.person == "Target":
                    reward = self.y[j]
                elif self.person == "Care":
                    if self.care_learning:
                        reward = self.theta_rwd @ self.rwd_X[j+1, :]
                        if self.care_mediator_weight > 0:
                            reward -= self.care_mediator_weight * self.raw_X[j+1, 3]
                    else:
                        reward = self.y[j]
                else:
                    reward = self.theta_rwd @ self.rwd_X[j, :] + self.next_week_weight * self.theta_rwd @ self.rwd_X[j+1, :]
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
            X = []
            y = []
            for j in range(self.X.shape[0]-1):
                X.append(self.X[j, :])
                if self.person == "Target":
                    reward = self.y[j]
                elif self.person == "Care":
                    reward = self.theta_rwd @ self.rwd_X[j, :]
                else:
                    reward = self.theta_rwd @ self.rwd_X[j, :] + self.next_week_weight * self.theta_rwd @ self.rwd_X[j+1, :]
                y.append(reward)
            y = np.array(y)
            X = np.array(X)

        # print(X.shape, y.shape, self.variance.shape)
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
    def getRewardFeature(self, state, action):
        oneHot = np.append([1], action)
        if self.person == "Target":
            short_state = state[[0, 1, 2]] # these is no need to learn for the target
            feature = short_state
        elif self.person == "Care":
            short_state = state[[0, 3, 4]] # care-giver uses its stress and current burden
            feature = np.outer(short_state, oneHot, out=None).T.reshape(-1)
        else:
            short_state = state[[0, 2, 4, 5]]
            feature = np.outer(short_state, oneHot, out=None).T.reshape(-1)
        return feature
    
    def endSave(self):
        if self.save_coefficients:
            # print(self.theta)
            return self.theta
        else:
            return None
    
    def reward_learning(self):
        # run a regression to predict the reward
        tmp_y = self.rwd_y
        tmp_X = self.rwd_X

        # if self.person == "Game":
        #     self.theta_rwd = np.array([0, -1, -0.3])
        if self.person == "Care":
            tmp_y = self.raw_X[7:, 5] # predict future relationship
            tmp_X = self.rwd_X[:-7]
            self.theta_rwd = np.linalg.inv(self.rwd_variance_inv+self.rwd_prior_var) @ \
                (self.rwd_variance_inv @ (1 / (self.sigma**2) * np.dot(self.rwd_variance, tmp_X.T @ tmp_y)).reshape(-1) + self.rwd_prior_var @ self.rwd_theta_prior)
        if self.person == "Game":
            self.theta_rwd = np.linalg.inv(self.rwd_variance_inv+self.rwd_prior_var) @ \
                (self.rwd_variance_inv @ (1 / (self.sigma**2) * np.dot(self.rwd_variance, tmp_X.T @ tmp_y)).reshape(-1) + self.rwd_prior_var @ self.rwd_theta_prior)
        
        self.rwd_update_count += 1
        # if self.save_param and self.rwd_update_count % 1 == 0 and self.person in ["Care", "Game"]:
        #     save_dir = args.save.split(".txt")[0]
        #     if self.rwd_update_count == 1:
        #         with open(f"{save_dir}_theta_rwd_{self.person}.csv", 'w') as f:
        #             np.savetxt(f, self.theta_rwd.reshape(1, -1), delimiter=",")
        #     else:
        #         with open(f"{save_dir}_theta_rwd_{self.person}.csv", 'a') as f:
        #             np.savetxt(f, self.theta_rwd.reshape(1, -1), delimiter=",")

        # if self.X.shape[0] % 10 == 0:
            # print(self.theta_rwd)
    def getFeatureDim(self):
        dummy_state = np.zeros(STATE_DIM)  
        return self.getFeature(dummy_state, 0).shape[0]
    
    def getRewardFeatureDim(self):
        dummy_state = np.zeros(STATE_DIM)  
        return self.getRewardFeature(dummy_state, 0).shape[0]

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
            self.reward_learning()
            self.theta = self.__thompsonSampling()
        
        predictionReward = []
        for action in [0, 1]:
            x = self.getFeature(state, action)
            predictionReward.append(np.dot(self.theta, x))
        action = [0, 1][samplePolicy(np.array(predictionReward), self.softmax)]
        # return np.random.randint(0, 2) ## TODO: remove this
        return action

    def addData(self, data: list, abs_time: int, person: str, idx: int):
        X = self.getFeature(data[0], data[1])
        rwd_X = self.getRewardFeature(data[0], data[1])
        raw_X = data[0]
        r = data[2]
        if self.X.shape[0] == 0:
            self.X = np.append(self.X, X).reshape(1, -1)
            self.y = np.append(self.y, r)
            self.raw_X = np.append(self.raw_X, raw_X).reshape(1, -1)
            self.rwd_X = np.append(self.rwd_X, rwd_X).reshape(1, -1)
            self.rwd_y = np.append(self.rwd_y, r)
        else:
            self.X = np.vstack([self.X, X])
            self.y = np.append(self.y, r)
            self.raw_X = np.vstack([self.raw_X, raw_X])
            self.rwd_X = np.vstack([self.rwd_X, rwd_X])
            self.rwd_y = np.append(self.rwd_y, r)
        if self.maxN is not None:
            n = self.X.shape[0]
            self.X = self.X[(n-self.maxN):n, :]
            self.y = self.y[(n-self.maxN):n]
            self.raw_X = self.raw_X[(n-self.maxN):n, :]
            self.rwd_X = self.rwd_X[(n-self.maxN):n, :]
            self.rwd_y = self.rwd_y[(n-self.maxN):n]

        rank1Update = self.X[-1, :] / self.sigma
        rank1Update_rwd = self.rwd_X[-1, :] / self.sigma
        self.variance = incInverse(self.variance, rank1Update, rank1Update)
        self.rwd_variance = incInverse(self.rwd_variance, rank1Update_rwd, rank1Update_rwd)
        self.rwd_variance_inv += np.outer(rank1Update_rwd, rank1Update_rwd)

if __name__ == "__main__":
    bandit = Bandit()
    bandit.addData((np.zeros(STATE_DIM), 0, 1), (np.zeros(STATE_DIM), 0, 1))
    print(bandit.getPolicy())
