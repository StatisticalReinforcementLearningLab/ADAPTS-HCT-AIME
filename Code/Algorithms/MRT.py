from .Algorithm import Algorithm
from .Policy import Policy
import numpy as np


class MRT(Algorithm):
    def __init__(self, p=0.5, person = "Target", saveData = False, reward_type = "naive", nonRandom = False):
        super().__init__("MRT", person = person, reward_type = reward_type)
        self.gameProb = p
        self.targetProb = p
        self.caregiverProb = p
        self.saveData = saveData
        self.X = np.array([])
        self.y = np.array([])
        self.nonRandom = nonRandom
        self.p_state = 0

    def addData(self, data, abs_time: int, person: str, idx: int):
        if self.saveData:
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

    def __str__(self):
        return f"MRT: gameProb={self.gameProb}, targetProb={self.targetProb}, caregiverProb={self.caregiverProb}"

    def getPolicy(self, subject) -> Policy:
        if self.nonRandom and self.gameProb == 0.5:
            targetPolicy = self.p_state
            caregiverPolicy = self.p_state
            gamePolicy = self.p_state
            self.p_state = (self.p_state + 1) % 2
        else:
            targetPolicy = np.random.choice(
                [0, 1], size=1, p=[1 - self.targetProb, self.targetProb]
            )[0]
            caregiverPolicy = np.random.choice(
                [0, 1], size=1, p=[1 - self.caregiverProb, self.caregiverProb]
            )[0]
            gamePolicy = np.random.choice(
                [0, 1], size=1, p=[1 - self.gameProb, self.gameProb]
            )[0]

        if self.person == "Target":
            return targetPolicy
        elif self.person == "Care":
            return caregiverPolicy
        else:
            return gamePolicy
