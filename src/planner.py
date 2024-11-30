"""
Greedy-like and Multi-Agent Deep REINFORCE Markov Decision Process optimizers

RH: TODO:
    - Implement greedy-like stochastic planner (MDP without policy learning basically)
    - Implement Deep on-policy MDP RL algorithm
"""

import numpy as np
from crowd_sim_cons import *

# Single instance per agent
# class IPlanner:
#     def __init__(self, local_density_map=None) -> None:
#         # Initial transition matrix
#         self._tm = np.ones((local_density_map.size, local_density_map.size))
#         self._tm = self._tm / np.sum(self._tm)


class Manager:
    def __init__(self, planners, simulator) -> None:
        self.planners = sorted(planners) # sorts on underlying agent ID
        self.simulator = simulator
        self.STEP_NUMBER = -1

    def step(self):
        # states = self.simulator.get_sim_state()
        # for state, p in zip(states, self.planners):
        for p in self.planners:
            action = p.choose_action() # For greedy planner, will choose stochastically using belief map PDF as state transition matrix
            p.do_action(action)

        # Step the simulator to next frame
        self.simulator.advance = True

        self.STEP_NUMBER += 1

# Single instance per agent?
class GreedyPlanner:
    def __init__(self, agent, init_st_mat='uniform') -> None:
        self.agent = agent
        self.rng = np.random.default_rng()

    def __lt__(self, other):
        # Compare underlying agent ids
        return self.agent.id < other.agent.id

    # Functions for choosing 
    def choose_action(self):
        # Select FLAT index from belief map given probabilities of state transition matrix
        flatind = self.rng.choice(np.arange(self.agent.map.size), p=self.agent.map.flat)
        return (self.agent.map.gridX.flat[flatind], self.agent.map.gridY.flat[flatind])

    def do_action(self, action):
        # Action should already be the target position as an X,Y tuple or something similar
        self.agent.patrol(action) # Gives the drone a point to try to travel to


###############################################
# Borrowing some code from the tutorial site here

# class DeepMDPPlanner(REINFORCE):
#     pass

if __name__=="__main__":
    # Init Crowd Simulator
    
    # Init Manager

    # Run loop


    pass