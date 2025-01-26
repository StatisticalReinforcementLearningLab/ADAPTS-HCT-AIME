import numpy as np
from .algargs import args
from utils import State, STATE_DIM


def linearly(weight: np.ndarray, stateObsed: np.ndarray):
    # Obs for observe.
    return weight @ stateObsed


rewardDict = {
    "linearly": linearly,
}

# args.delayed_weight: weight for the burden

def preload_weight(delayed_weight, naive_weight):
    return {
        "Target_naive": np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0,0]),
        "Target_delayed": np.array([0, 1, -delayed_weight, 0, 0, 0, 0, 0, 0, 0,0]),
        "Target_delayed0": np.array([0, 1, -delayed_weight, 0, 0, 0, 0, 0, 0, 0,0]),
        "Target_mediator": np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0,0]),
        "Target_mixed": np.array([0, 1, -delayed_weight, 0, 0, 0, 0, 0, 0, 0,0]),
        "Care_naive": np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0,0]),
        "Care_delayed": np.array([0, 0, 0, -1, -delayed_weight, 0, 0, 0, 0, 0,0]),
        "Care_delayed0": np.array([0, 1, 0, 0, -delayed_weight, 0, 0, 0, 0, 0,0]),
        "Care_mediator": np.array([0, 0, 0, -1, 0, 0, 0, 0, 0, 0,0]),
        "Care_mixed": np.array([0, naive_weight, 0, -1, -delayed_weight, 0, 0, 0, 0, 0,0]),
        "Game_naive": np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0,0]),
        "Game_delayed": np.array([0, 0,-delayed_weight/5,0,-delayed_weight/5,1,0,0,0,0,0]),
        "Game_delayed0": np.array([0, 1,-delayed_weight/5,0,-delayed_weight/5,0,0,0,0,0,0]),
        "Game_mediator": np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0,0]),
        "Game_mixed": np.array([0, naive_weight, -delayed_weight/5,0,-delayed_weight/5,1,0,0,0,0,0])
    }

class RewardMapping:
    def __init__(self, person: str, reward_type: str, delayed_weight = None, naive_weight = None):
        if delayed_weight is not None:
            self.delayed_weight = delayed_weight
        else:
            self.delayed_weight = args.delayed_weight
        if naive_weight is not None:
            self.naive_weight = naive_weight
        else:
            self.naive_weight = args.naive_weight

        self.preload_weight = preload_weight(self.delayed_weight, self.naive_weight)
        self.person = person
        if person not in ["Game", "Care", "Target"]:
            raise ValueError("person must be one of 'Game', 'Care', or 'Target'")
        self.reward_type = reward_type
        if self.reward_type not in ["delayed", "mediator", "mixed", "naive", "delayed0"]:
            raise ValueError("reward_type must be one of 'delayed', 'mediator', 'mixed', or 'naive'")
        self.weight = self.preload_weight[f"{self.person}_{self.reward_type}"]

        self.mapping = linearly # use linear for now

    def map(self, stateObsed: list, curState: State):
        

        # if not target person, we use average adherence over the week/day
        if not self.person == "Target":
            curState[1] = sum([state[1] for state in stateObsed]) / len(stateObsed)
        if self.person == "Game":
            # average game burden over the week
            curState[2] = sum([state[2] for state in stateObsed]) / len(stateObsed)
            curState[4] = sum([state[4] for state in stateObsed]) / len(stateObsed)
        return self.mapping(self.weight, curState)


if __name__ == "__main__":
    reward_mapping = RewardMapping("Target", "naive")
    print(reward_mapping.weight)
