from .RewardMapping import RewardMapping
from .Policy import Policy
# from .algargs import args

class Algorithm:
    def __init__(self, name, poolAcrossDyads=True, person = "Target", reward_type = "naive", delayed_weight = None, naive_weight = None, save_coefficients = False):
        self.name = name
        self.reward_type = reward_type
        self.person = person
        self.rewardMappings = RewardMapping(self.person, self.reward_type, delayed_weight, naive_weight)
        self.poolAcrossDyads = poolAcrossDyads
        self.save_coefficients = save_coefficients

    def __str__(self):
        return f"Using algorithm: {self.name}\n"

    def getPolicy(self) -> int:
        # Policy are in [target, caregiver, weekly]
        raise NotImplementedError("Subclass must implement abstract method")

    def addData(self, data, h: int, idx: int):
        raise NotImplementedError("Subclass must implement abstract method")
    def getPerson(self):
        return self.person
    def endSave(self):
        pass

    def getRewardType(self):
        return self.reward_type
if __name__ == "__main__":
    pass
