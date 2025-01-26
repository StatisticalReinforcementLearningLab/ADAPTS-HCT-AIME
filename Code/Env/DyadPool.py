from . import Dyad, People
import random
import copy

DYAD_NUM = 63

class DyadPool:
    def __init__(self, args):
        # Here keeps all the dyads we created according to existing dataset.
        self.dyads = []
        self.args = args
        # Just add all-zero dyads for now.
        # we have in total 63 dyads
        for i in range(DYAD_NUM):
            target = People.Target(idx=i, args=args)
            caregiver = People.Carepartner(idx=i, args=args)
            self.dyads.append(Dyad.Dyad(target, caregiver))

    def getDyad(self, idx = None):
        if idx is None:
            new_idx = random.choice(range(DYAD_NUM))
            new_dyad = copy.deepcopy(self.dyads[new_idx])
            new_dyad.initialize_treatment_effect()
            return new_dyad
        else:
            new_dyad = copy.deepcopy(self.dyads[idx%DYAD_NUM])
            new_dyad.initialize_treatment_effect()
            return new_dyad
