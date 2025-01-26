from . import DyadPool, TrialTime, envargs, my_logger
import copy
import numpy as np
from typing import List

argsDefault = envargs.args
if argsDefault.printWhenRead:
    print(argsDefault)


class Trial:
    def __init__(self, algorithm: List, args=argsDefault):
        self.time = TrialTime.TrialTime()

        self.pool = DyadPool.DyadPool(args=args)
        self.subjects = []
        self.__removed = []

        self.algorithm = algorithm

        self.avgRs = []
        self.avgAs = []
        self.avgDs = []
        self.avgRels = []
        # the average reward of a dyad across the trial

        self.args = args

        self.logger = my_logger.Logger(caller="Trial")

    def employNewSubject(self, poolAcrossSubjects=True, idx = None):
        # print("Employing new subject %d" % idx)
        dyad = self.pool.getDyad(idx)
        dyad.seq_idx = len(self.subjects)
        dyad.startTime = copy.deepcopy(self.time)
        dyad.currentTime = copy.deepcopy(self.time)
        # The following 2 should be exact same thing of that of dyad.
        dyad.target.startTime = dyad.startTime
        dyad.caregiver.startTime = dyad.startTime
        self.subjects.append(dyad)
        self.logger.log(
            f"New subject employed. Now we have (with removed) {len(self.subjects)} subjects."
        )

        alg4Dyad = []
        for alg in self.algorithm:
            if alg.poolAcrossDyads:
                alg4Dyad.append(alg)
            else:
                alg4Dyad.append(copy.deepcopy(alg))
        dyad.setAlgorithm(alg4Dyad)
        dyad.employed()

    def progressTime(self):
        self.logger.log(f"Time progressed to {self.time}.")

        for i, subject in enumerate(self.subjects):
            elapsedTime = self.time - subject.startTime
            if elapsedTime.week >= self.args.W:
                if i not in self.__removed:
                    self.logger.log(
                        f"Subject {i+1} has reached the end of the trial, removing..."
                    )
                    ss = 1 if i == 0 else 0
                    subject.avgR = np.nanmean(np.array(subject.state.adh[ss:]).reshape(-1, 14), axis = 1)  # weekly mean reward
                    subject.As = [np.nanmean(np.array(subject.state.tarA[ss:]).reshape(-1, 14), axis = 1),
                                np.nanmean(np.array(subject.state.dayA[ss:]).reshape(-1, 7), axis = 1), 
                                np.array(subject.state.weekA[ss:])]  # each dyad's mean reward
                    subject.distress = np.nanmean(np.array(subject.state.distress[ss:]).reshape(-1, 7), axis = 1)
                    subject.relationship = np.nanmean(np.array(subject.state.relationship[ss:]).reshape(-1, 1), axis = 1)
                    self.avgRs.append(subject.avgR)
                    self.avgAs.append(subject.As)
                    self.avgDs.append(subject.distress)
                    self.avgRels.append(subject.relationship)
                    self.__removed.append(i)
                continue

            self.logger.log(f"Progressing subject {i+1}...")

            subject.progressTo(self.time)

        self.time.getNextTime()

    def isGoing(self):
        return len(self.subjects) > len(self.__removed)

    def summarize(self):
        return (np.nanmean(self.avgRs), np.nanstd(self.avgRs))

    # I think this has no meaning now
    # Since avgRs is not a chronological sequence
    def drawR(self):
        import matplotlib.pyplot as plt

        plt.plot(self.avgRs)
        plt.show()
