import numpy as np
import pandas as pd
from . import TrialTime, envutils, my_logger
from utils import compute_mean_var_burden


# ori path = "./../Model_Fitting/residuals_heter"
path = "./Model_Fitting/residuals_heter"

coeffi_AYA = pd.read_csv("%s/coeffi_AYA.csv" % path)
coeffi_care = pd.read_csv("%s/coeffi_care.csv" % path)

std_AYA_AM = coeffi_AYA.iloc[:, 1].std()
std_AYA_PM = coeffi_AYA.iloc[:, 5].std()

std_care = coeffi_care["Distress"].std()

sleep_init = pd.read_csv("%s/sleep_initial.csv" % path, dtype=float)
residual_care = pd.read_csv("%s/residual_care.csv" % path)


class People:
    def __init__(self, idx, args):
        self.startTime = None
        self.currentTime = None
        self.args = args

        # index of the dyad in the dataset idx is in [0, 62]
        self.idx = idx

        self.state = None

        self.logger = None
        # Should be initialized in the subclass.

    # Since this function for 2 subclass is very different, we choose not to implement it here.
    # def transitionToNextPhase(self):
    #     raise NotImplementedError("This method should be implemented in the subclass.")

    def logState(self):
        if self.logger is None:
            raise ValueError("Logger is not initialized.")
        self.logger.log(f"Current state: {self.state}")


class Target(People):

    def __init__(self, idx, args):
        super().__init__(idx=idx, args=args)

        # Binary, represents adherence
        # self.state: int = 0
        # self.historicalState = self.state

        # parameters for adherence transition
        coe = coeffi_AYA.iloc[self.idx, :]
        self.constantOfAdherence = np.array([coe.iloc[0], coe.iloc[4]])
        self.coeOfAdh = np.array([coe.iloc[1], coe.iloc[5]])
        self.coeOfRel = np.array([max(0, coe.iloc[2]), max(0, coe.iloc[6])])
        self.coeOfStress = np.array([min(0, coe.iloc[3]), min(0, coe.iloc[7])])

        # if args.mediator:
        if self.args.mediator>=0:
            self.coeOfRel = np.array([0, 0]) + self.args.mediator * np.abs(self.coeOfAdh) / 2
        else:
            self.coeOfRel = np.array([0, 0]) + np.random.normal(0, 1) * np.abs(self.coeOfAdh) / 2 # when mediator effect < 0, we use random noise to simulate the effect
        if self.args.add_direct_effect:
            # use gaussian
            self.coeOfStress = np.array([np.random.normal(0, np.abs(self.coeOfAdh) / 2) / 5, np.random.normal(0, np.abs(self.coeOfAdh) / 2) / 5]) * self.args.mediator 

        # parameters for burden transition
        self.coeOfBurden = np.array(
            [1 - 1.0 / 14, 1.0, 0.2]
        )

        self.bNoise = self.args.bNoise

        c_high = 0.6
        c_low =  0.2
        incr = (c_high - c_low) / 195
        if not args.addTrend:
            self.constantOfBurden: np.ndarray = np.array([c_low]*200)
        else:
            self.constantOfBurden: np.ndarray = np.arange(c_low, c_high + incr * 2, incr)

        # self.mean_burden, self.var_burden = compute_mean_var_burden(self.coeOfBurden, c_low, self.bNoise)
        self.mean_burden, self.var_burden = (11.145000272462855, 43.574579072819276)

        # print(f"Mean Burden: {mean_burden}, Var Burden: {var_burden}")
        # logger
        self.logger = my_logger.Logger(caller="Target")

        # Class variables
        self.baseCoeOfGame = np.abs(self.coeOfAdh) / 5
        self.baseCoeOfIntv = np.abs(self.coeOfAdh)
        # self.discount4Rel: float = self.args.DiscountR
        # self.discount4Burden: float = self.args.DiscountB
    def initialize_treatment_effect(self):
        self.baseCoeOfIntv = np.abs(self.coeOfAdh) + np.array([np.random.normal(0, std_AYA_AM), np.random.normal(0, std_AYA_PM)])
        self.baseCoeOfGame = np.abs(self.coeOfAdh) / 5 + np.array([np.random.normal(0, std_AYA_AM), np.random.normal(0, std_AYA_PM)]) / 5

    def __getTreatmentEffect(self):
        m = self.currentTime.timeOfDay
        # coeOfGame = self.baseCoeOfGame[m] * (1 - self.discount4Rel * (1 - self.rel))
        coeOfGame = max(0, self.baseCoeOfGame[m] * self.args.treat0 + self.state.relationship[-1] * self.baseCoeOfGame[m] * self.args.treat1 - self.state.ayaBurden[-1] * self.baseCoeOfGame[m] * self.args.treat2)
        coeOfIntv = max(0, self.baseCoeOfIntv[m] * self.args.treat0 + self.state.relationship[-1] * self.baseCoeOfIntv[m] * self.args.treat1 - self.state.ayaBurden[-1] * self.baseCoeOfIntv[m] * self.args.treat2)
        if not self.args.reinforces:
            coeOfGame  = 0
        return coeOfGame, coeOfIntv

    def __transitionOfBurden(self):
        d = (self.currentTime - self.startTime).getAbsoluteTime()
        x = self.constantOfBurden[d]
        
        # generate the vector for the transition of burden
        vec = np.array([self.state.ayaBurden[-1] * np.sqrt(self.var_burden) + self.mean_burden, self.state.tarA[-1], self.state.weekA[-1]])
        x += self.coeOfBurden @ vec
        x += np.random.normal(loc=0.0, scale=self.bNoise)
        x = (x - self.mean_burden) / np.sqrt(self.var_burden)
        return x

    def __transitionOfAdherence(self):
        m = self.currentTime.timeOfDay
        x = self.constantOfAdherence[m]
        x += self.coeOfAdh[m] * self.state.adh[-1]
        x += self.coeOfStress[m] * self.state.distress[-1]
        x += self.coeOfRel[m] * self.state.relationship[-1]
        coeOfGame, coeOfIntv = self.__getTreatmentEffect()
        x += coeOfGame * self.state.weekA[-1]
        x += coeOfIntv * self.state.tarA[-1]

        if self.args.extreme_direct_effect:
            x -= self.coeOfAdh[m] * self.state.weekA[-1] * 100
        return envutils.Bernoulli_logistic(x)

    def __transitionToNextPhase(self):
        newAdherence = self.__transitionOfAdherence()
        newBurden = self.__transitionOfBurden()
        # update new state for the target
        self.state.adh.append(newAdherence)
        self.state.ayaBurden.append(newBurden)

    def progressTo(self):
        # self.state.tarA.append(tarA)
        self.__transitionToNextPhase()
        self.logState()


class Carepartner(People):

    def __init__(self, idx, args):
        super().__init__(idx=idx, args=args)

        self.state: float = sleep_init.iloc[self.idx, 1] 
        # self.historicalState = self.state

        # parameters for burden transition

        self.coeOfBurden = np.array(
            [1 - 1.0 / 7, 1, 0.2]
        )
        c_high = 0.6
        c_low = 0.2
        incr = (c_high - c_low) / 98
        if not args.addTrend:
            self.constantOfBurden: np.ndarray = np.array([c_low]*200)
        else:
            self.constantOfBurden: np.ndarray = np.arange(c_low, c_high + incr * 2, incr)
        self.bNoise = self.args.bNoise
        # self.mean_burden, self.var_burden = compute_mean_var_burden(self.coeOfBurden, c_low, self.bNoise)
        self.mean_burden, self.var_burden = (5.923000858825525, 22.47418236742726)

        # parameters for stress transition
        coe = coeffi_care.iloc[self.idx, :]
        self.coeOfStress: float = coe.iloc[1]
        self.coe4StressOfADH: float = coe.iloc[2]
        self.coe4StressOfRel: float = coe.iloc[3]
        self.constantOfStress: float = coe.iloc[0]

        # residual
        self.residual = residual_care.iloc[self.idx, ::-1].values.tolist()

        # Class variables
        self.baseCoeOfGame: float = -np.abs(self.coeOfStress) / 5
        self.baseCoeOfIntv: float = -np.abs(self.coeOfStress)

        self.bNoise = self.args.bNoise / ((0.6 + 1 + 0.5) * 7)
        # self.discount4Rel: float = self.args.DiscountR
        # self.discount4Burden: float = self.args.DiscountB

        self.logger = my_logger.Logger(caller="Caregiver")
    def initialize_treatment_effect(self):
        self.baseCoeOfIntv = -np.abs(self.coeOfStress) + np.random.normal(0, std_care)
        self.baseCoeOfGame = -np.abs(self.coeOfStress) / 5 + np.random.normal(0, std_care / 5)

    def __transitionOfBurden(self):
        d = (self.currentTime - self.startTime).getAbsoluteTime() // 2
        x = self.constantOfBurden[d]
        # vec = np.array([self.burden, self.Intv, self.dyadIntv])
        vec = np.array([self.state.careBurden[-1] * np.sqrt(self.var_burden) + self.mean_burden, self.state.dayA[-1], self.state.weekA[-1]])
        x += self.coeOfBurden @ vec
        x += np.random.normal(loc = 0, scale=self.bNoise)
        x = (x - self.mean_burden) / np.sqrt(self.var_burden)
        return x

    def __getTreatmentEffect(self):
        base_effect = self.args.treat0
        relationship_effect = self.state.relationship[-1] * self.args.treat1
        burden_effect = -self.state.careBurden[-1] * self.args.treat2
        
        coeOfGame = min(0, self.baseCoeOfGame * (base_effect + relationship_effect + burden_effect)) if not self.args.reinforces else 0
        coeOfIntv = min(0, self.baseCoeOfIntv * (base_effect + relationship_effect + burden_effect))
        return coeOfGame, coeOfIntv

    def __transitionOfStress(self):
        x = self.constantOfStress
        x += self.coeOfStress * self.state.distress[-1]
        x += self.coe4StressOfRel * self.state.relationship[-1]
        x += self.coe4StressOfADH * self.state.adh[-1]
        coeOfGame, coeOfIntv = self.__getTreatmentEffect()
        x += coeOfGame * self.state.weekA[-1]
        x += coeOfIntv * self.state.dayA[-1]
        x += self.residual.pop()  # pop one residual
        return x

    def __transitionToNextPhase(self):
        newStress = self.__transitionOfStress()
        newBurden = self.__transitionOfBurden()
        self.state.distress.append(newStress)
        self.state.careBurden.append(newBurden)

    def progressTo(self):
        self.__transitionToNextPhase()
        self.logState()
