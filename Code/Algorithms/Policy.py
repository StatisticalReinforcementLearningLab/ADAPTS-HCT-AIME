from .algutils import integer_to_binary
import numpy as np

class Policy:
    def __init__(self):
        # We always denote in the order of
        # the target as 0, caregiver as 1, and dyad as 2.
        self.target = np.nan
        self.caregiver = np.nan
        self.dyad = np.nan

    def getInt(self):
        return self.target * 4 + self.caregiver * 2 + self.dyad

    def getList(self):
        return [self.target, self.caregiver, self.dyad]

    def setByInt(self, value):
        oneHot = integer_to_binary(value, 3)
        self.target = oneHot[0]
        self.caregiver = oneHot[1]
        self.dyad = oneHot[2]

    def set(self, index, value):
        if index == 0:
            self.target = value
        elif index == 1:
            self.caregiver = value
        elif index == 2:
            self.dyad = value
        else:
            raise ValueError("Invalid who value.")
