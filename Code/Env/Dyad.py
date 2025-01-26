import numpy as np
import pandas as pd
from Algorithms import MRT, Policy
from . import People, TrialTime, my_logger, envutils
from utils import State
# Constants
PATH = "./Model_Fitting/residuals_heter"
COEFFI_WEEKLY = pd.read_csv(f"{PATH}/coeffi_weekly.csv")
STD_WEEKLY = COEFFI_WEEKLY["Relationship"].std()


class Dyad:
    def __init__(self, target: People.Target, caregiver: People.Carepartner):
        self.algorithm = None
        self.state = State()
        self.data = []
        self.target = target
        self.caregiver = caregiver
        self.target.state = self.state
        self.caregiver.state = self.state
        self.startTime = None
        self.currentTime = None
        self.idx = target.idx
        self.seq_idx = None
        self.args = target.args

        coe = COEFFI_WEEKLY.iloc[target.idx, :]
        self.gamma = 6/7
        self.constantOfRel = coe.iloc[0]
        self.coeOfRel = coe.iloc[1]
        self.coeOfTarget = coe.iloc[2]
        self.coeOfCaregiver = min(0, coe.iloc[3])
        # if target.args.mediator_effect:
        if target.args.mediator>=0:
            self.coeOfCaregiver = 0 - target.args.mediator * np.abs(self.coeOfRel) / 2
        else:
            self.coeOfCaregiver = 0 - np.random.normal(0, 1) * np.abs(self.coeOfRel) / 2 # when mediator effect < 0, we use random noise to simulate the effect
        self.coeOfGame = np.abs(self.coeOfRel) * target.args.treat3

        self.logger = my_logger.Logger(caller=f"Dyad {target.idx}")
        self.logDetail = False

        # initialize old states
        self.oldGameState = [self.getState()]
        self.oldCaregiverState = [self.getState()]
        self.oldTargetState = [self.getState()]

        self.lastGameState = self.getState()
        self.lastCaregiverState = self.getState()
        self.lastTargetState = self.getState()
    
    def initialize_treatment_effect(self):
        self.coeOfGame = np.abs(self.coeOfRel) + np.random.normal(0, STD_WEEKLY)
        self.target.initialize_treatment_effect()
        self.caregiver.initialize_treatment_effect()

    def getState(self):
        return self.state.getState()

    def __getCoeOfGame(self):
        if not self.args.noRelBurden:
            return max(0, self.coeOfGame * self.args.treat3 - (self.state.ayaBurden[-1] + self.state.careBurden[-1]) * self.coeOfGame * self.args.treat2 / 2)
        else:
            return max(0, self.coeOfGame)
    
    def __weeklyTransitionOfRel(self):
        x = self.constantOfRel
        x += self.coeOfRel * self.state.relationship[-1]
        x += self.coeOfTarget * self.state.wAveAdh[-1]
        x += self.coeOfCaregiver * self.state.wAveDistress[-1]
        if self.args.extreme_direct_effect:
            x -= self.coeOfRel * np.sum(self.state.dayA[-7:]) * 100
        x += self.__getCoeOfGame() * self.state.weekA[-1]

        newRel = envutils.Bernoulli_logistic(x)
        self.state.relationship.append(newRel)

    def employed(self):
        message = f"{self.target.idx}th Dyad employed."
        if self.logDetail:
            message += (f"\ncoeOfRel: {self.coeOfRel}"
                        f"\ncoe4RelOfTarget: {self.coeOfTarget}"
                        f"\ncoe4RelOfCaregiver: {self.coeOfCaregiver}"
                        f"\ncoe4RelOfGame: {self.coeOfGame}"
                        f"\nstartTime:{self.startTime}")
        self.logger.log(message)

    def logState(self):
        self.logger.log(f"Current Mood: {self.state.relationship[-1]}")

    def setAlgorithm(self, algorithms: list):
        self.algorithm = algorithms

    def getWho(self):
        who = 4
        if self.currentTime.isNewDay():
            who += 2
        if self.currentTime.isNewWeek():
            who += 1
        return who

    def getPolicy(self, person: str):
        # return a binary action for a given person
        if person not in ["Target", "Care", "Game"]:
            raise ValueError("person must be one of 'Target', 'Care', or 'Game'")
        if self.algorithm is None:
            raise ValueError("Algorithm not set.")
        i = {"Target": 0, "Care": 1, "Game": 2}[person]
        tem = self.algorithm[i].getPolicy(self)
        if type(tem) == list:
            tem = tem[i]
        self.logger.log(f"Policy for {person}: {tem}")
        return tem

    def getRandomPolicy(self, person: str):
        if person not in ["Target", "Care", "Game"]:
            raise ValueError("person must be one of 'Target', 'Care', or 'Game'")
        random_policy = np.random.randint(0, 2)
        self.logger.log(f"Random Policy for {person}: {random_policy}")
        return random_policy

    def progressTo(self, currentTime):
        self.currentTime = currentTime
        self.target.currentTime = currentTime
        self.caregiver.currentTime = currentTime

        
        if self.currentTime.isNewWeek():
            # progress previous week's relationship
            self.__weeklyTransitionOfRel()
            self.passData("Game")
        if self.currentTime.isNewDay():
            # progress previous day's relationship
            self.caregiver.progressTo()
            self.state.updateAveDistress()
            self.passData("Care")
        # this is the state decisions at this time is based on
        self.oriState = self.getState()
        if self.currentTime.isNewWeek():
            # game agent makes decision
            weekA = self.getPolicy("Game") if currentTime.week > self.args.Warmup else self.getRandomPolicy("Game")
            self.state.weekA.append(weekA)
            self.lastGameState = self.getState()
            # used to generate weekly summary
            self.oldGameState = []
            
        
        if self.currentTime.isNewDay():
            # caregiver agent makes decision
            dayA = self.getPolicy("Care") if currentTime.week > self.args.Warmup else self.getRandomPolicy("Care")
            self.state.dayA.append(dayA)
            # used to generate daily summary
            self.lastCaregiverState = self.getState()
            self.oldCaregiverState = []

        # target agent makes decision
        tarA = self.getPolicy("Target") if currentTime.week > self.args.Warmup else self.getRandomPolicy("Target")
        self.lastTargetState = self.getState()
        self.state.tarA.append(tarA)
        self.state.time_of_day.append(self.currentTime.isNewDay())

        # progress this decision time
        self.target.progressTo()
        self.state.updateAveAdh()
        self.passData("Target")

        self.oldGameState.append(self.getState())
        self.oldCaregiverState.append(self.getState())

        # Store data if required
        if self.args.store_data:
            self.data.append(self.oriState.tolist()+ [self.state.tarA[-1], self.state.dayA[-1], self.state.weekA[-1],
                                                        self.state.adh[-1], (self.currentTime - self.startTime).getAbsoluteTime()])

    def passData(self, person: str):
        if person == "Game":
            reward = self.algorithm[2].rewardMappings.map(self.oldGameState, self.getState())
            # pass a list of states with one week, their weekly action, reward and absolute time
            self.algorithm[2].addData([self.lastGameState, self.lastGameState[6], reward], (self.currentTime - self.startTime).getAbsoluteTime(), person, self.seq_idx)
        elif person == "Care":
            reward = self.algorithm[1].rewardMappings.map(self.oldCaregiverState, self.getState())
            self.algorithm[1].addData([self.lastCaregiverState, self.lastCaregiverState[7], reward], (self.currentTime - self.startTime).getAbsoluteTime(), person, self.seq_idx)
        elif person == "Target":
            reward = self.algorithm[0].rewardMappings.map([self.getState()], self.getState())
            self.algorithm[0].addData([self.lastTargetState, self.state.tarA[-1], reward], (self.currentTime - self.startTime).getAbsoluteTime(), person, self.seq_idx)


