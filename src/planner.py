"""
Greedy-like and Multi-Agent Deep REINFORCE Markov Decision Process optimizers

RH: TODO:
    - Implement greedy-like stochastic planner (MDP without policy learning basically)
    - Implement Deep on-policy MDP RL algorithm
"""

import numpy as np
from crowd_sim_cons import *
from crowd_sim import Simulation
from recorder import PygameRecord
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import torch.nn.functional as F

import warnings
from tqdm import tqdm
import json
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class Record:
    def __init__(self, l):
        self.observations = [None]*l
        self.actions = np.zeros(l)
        self.rewards = np.zeros(l)
    def as_dict(self, tiny=True):
        if tiny:
            return {"position": np.vstack(self.observations)[:,:2].tolist(), 
                    "actions": self.actions.tolist(),
                    "rewards": self.rewards.tolist()
                    }
        else:
            return {"observations": np.vstack(self.observations).tolist(), 
                    "actions": self.actions.tolist(),
                    "rewards": self.rewards.tolist()
                    }

class Records:
    def __init__(self, planners, max_num_steps):
        self.records = {p:Record(max_num_steps) for p in planners}
    def __getitem__(self, name):
        return self.records[name]

class Manager:
    def __init__(self, planners, simulator) -> None:
        self.planners = sorted(planners) # sorts on underlying agent ID
        self.simulator = simulator
        # self.STEP_NUMBER = -1

    def save(self, history=None, note=None):
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        if not os.path.exists(f'models/{timestamp}'):
            os.makedirs(f'models/{timestamp}', exist_ok=True)
        
        for p in self.planners:
            p.policy.save(f'models/{timestamp}/drone_{p.agent.id}_policy_model.ptm')

        if history is not None:
            # Get each Record as a dictionary, within a records dictionary, within a list, and convert to json?
            with open(f'models/{timestamp}/history.json', 'w') as f:
                j = json.dump({"note": note, "history": [{k.agent.id: v.as_dict() for k,v in records.records.items()} for records in history]}, f)

            cumrewards = np.asarray([[np.sum(pr.rewards) for pr in records.records.values()] for records in history])

            # Plot and save the reward curve
            plt.plot(cumrewards)
            plt.title('Cumulative Rewards Trace')
            plt.xlabel('Episode')
            plt.ylabel('Total Episode Reward')
            plt.legend()
            plt.savefig(f'models/{timestamp}/cumulative_rewards.png')

    def step(self):
        # states = self.simulator.get_sim_state()
        # for state, p in zip(states, self.planners):
        for p in self.planners:
            p.step(self.simulator.global_map.instantaneous_occupancy_map)

        # Step the simulator to next frame
        self.simulator.advance = True

        # self.STEP_NUMBER += 1

    # Similar to PolicyGradientExecutor.execute
    def train(self, episodes=100, max_num_steps=50):
        # This should run the training loop for all policies simultaneously, updating policies at the end of each episode, and performing some total number of episodes
        # Might need training step function on each DroneMDPPlanner instance which captures record of reward, cost, etc
        training_history = [None]*episodes

        for episode in tqdm(range(episodes)):
            # Init drones at beginning of episode in RANDOM LOCATIONS, but NOT the crowd (basically allow for burn-in)
            x = np.random.uniform(DRONE_RADIUS, WIDTH - DRONE_RADIUS, size=NUM_DRONES)
            y = np.random.uniform(DRONE_RADIUS, HEIGHT - DRONE_RADIUS, size=NUM_DRONES)
            for id in range(NUM_DRONES):
                self.simulator.drones[id].position.x = x[id]
                self.simulator.drones[id].position.y = y[id]

            records = Records(self.planners, max_num_steps)
            for step in range(max_num_steps):
                self.simulator.step()
                for p in self.planners:
                    observation, action = p.step(self.simulator.global_map.instantaneous_occupancy_map, train=True)
                    r = records[p]
                    r.observations[step] = observation
                    r.actions[step] = action
                    r.rewards[step] = (self.reward(p.agent) - self.cost(p.agent, self.simulator.drones)) #* p.get_discount_factor() ** step
        
                self.simulator.advance = True

            # For each observer, update its corresponding policy by ID
            for p in self.planners:
                # deltas = np.asarray(self.calculate_deltas(p, records[p].rewards))
                # p.policy.update(np.vstack(records[p].observations), records[p].actions, deltas)
                
                # Changing this to just do the policy update with Pytorch implementation all at once
                p.policy.update(np.vstack(records[p].observations), records[p].actions, records[p].rewards, p.get_discount_factor())

            # Store historical records for episode
            training_history[episode] = records

        # Return the raw data which we can then investigate and plot to show convergence of policy over time
        return training_history

    # A better name for this is the 'return' which is discounted future rewards (cumsum of reversed rewards with gamma applied to each step *relative* to step t from which each sum is performed)
    def calculate_deltas(self, mdp, rewards):
        # """
        # Generate a list of the discounted future rewards at each step of an episode
        # Note that discounted_reward[T-2] = rewards[T-1] + discounted_reward[T-1] * gamma.
        # We can use that pattern to populate the discounted_rewards array.
        # """
        # T = len(rewards)
        # discounted_future_rewards = np.zeros(T)

        # # The final discounted reward is the reward you get at that step
        # discounted_future_rewards[-1] = rewards[-1]
        # for t in reversed(range(0, T - 1)):
        #     discounted_future_rewards[t] = (
        #         rewards[t]
        #         + discounted_future_rewards[t + 1] * mdp.get_discount_factor()
        #     )
        # deltas = (mdp.get_discount_factor() ** np.arange(T)) * discounted_future_rewards
        # return deltas

        # GOING TO TRY PYTORCH VERSION OF THIS
        gamma = mdp.get_discount_factor()

    
    def cost(self, agent, agents):
        cost = 0
        for other in agents:
            if other.id != agent.id:
                cost += len(set.intersection(agent.current_observed_particles, other.current_observed_particles))
        return cost
    
    def reward(self, agent):
        reward = len(agent.current_observed_particles)
        return reward


# Single instance per agent?
class GreedyPlanner:
    def __init__(self, agent, init_st_mat='uniform') -> None:
        self.agent = agent
        self.rng = np.random.default_rng()

    def __lt__(self, other):
        # Compare underlying agent ids
        return self.agent.id < other.agent.id

    def step(self, global_inst_occ_map):
        self.agent.map.instantaneous_occupancy_map = global_inst_occ_map
        action = self.choose_action()
        self.do_action(action)

    # Functions for choosing 
    def choose_action(self):
        # Select FLAT index from belief map given probabilities of state transition matrix
        current_density_map = self.agent.map.get_density_map()
        flatind = self.rng.choice(np.arange(current_density_map.size), p=current_density_map.flat)
        return flatind

    def do_action(self, action):
        # Convert action to a target position as an X,Y tuple
        self.agent.patrol((self.agent.map.coordinate_mesh[0].flat[action], self.agent.map.coordinate_mesh[1].flat[action])) # Gives the drone a point to try to travel to


class DroneMDPPlanner:
    def __init__(
        self,
        agent,
        policy,
        state_space,
        action_space,
        discount_factor=0.9,
    ):
        self.agent = agent
        self.policy = policy
        self.discount_factor = discount_factor

        # indices of our state grid
        self.state_space = state_space
        self.action_space = action_space
        self.rlobsvec_prealloc = np.zeros(self.state_space)

        self._reset()

    # Initial setup (run as part of episode reset!)
    def _reset(self):
        # Clear all other environment state and records
        self.rlobsvec_prealloc = np.zeros(self.state_space)

    # # We should overload the execute_policy function to just run this once per episode
    # def run_episode(self, policies, max_steps):
    #     self._reset()

    #     # Run loop (THINK OF THIS AS A SINGLE EPISODE)
    #     EPISODE = 0
    #     hms[..., 0] = crowd_sim.global_map.get_density_map()
    #     (vm, vp) = crowd_sim.global_map.get_velocity_mag_and_phase_map()
    #     vmag[..., 0] = vm
    #     vphase[..., 0] = vp

    #     while EPISODE < max_steps:
    #         crowd_sim.step()
    #         self.step(crowd_sim.global_map.instantaneous_occupancy_map)
    #         EPISODE += 1

    #     # Here's where we record all episode information (observers have reward, cost, and cumulative information)
    #     return 

    def get_discount_factor(self):
        return self.discount_factor

    def __lt__(self, other):
        # Compare underlying agent ids
        return self.agent.id < other.agent.id

    # Non-training forward pass!
    def step(self, global_inst_occ_map, train=False):
        self.agent.map.instantaneous_occupancy_map = global_inst_occ_map
        action = self.choose_action()
        self.do_action(action)

        if train:
            return self.rlobsvec_prealloc, action

    # SINGLE STEP EXECUTION WITH ACTIVE POLICY FOR SINGLE AGENT!
    def choose_action(self):
        # construct observational state vector
        self.rlobsvec_prealloc[...] = np.concatenate([[self.agent.position.x/WIDTH, self.agent.position.y/HEIGHT], self.agent.map.get_density_map().flat])

        # choose new action
        action = self.policy.select_action(self.rlobsvec_prealloc) # action is a flattened index corresponding to action space aka coordinate mesh
        return action

    def do_action(self, action):
        # THIS EXPECTS A FLAT INDEX AS THE ACTION
        target = (self.agent.map.coordinate_mesh[0].flat[action], self.agent.map.coordinate_mesh[1].flat[action])
        self.agent.patrol(target)

# No changes needed probably!!!
class DeepNeuralNetworkPolicy:
    """
    An implementation of a policy that uses a PyTorch (https://pytorch.org/) 
    deep neural network to represent the underlying policy.
    """

    def __init__(self, state_space, action_space, hidden_dim=64, alpha=0.001, stochastic=True):
        self.state_space = state_space
        self.action_space = action_space

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the policy structure as a sequential neural network.
        self.policy_network = nn.Sequential(
            nn.Linear(in_features=self.state_space, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=self.action_space),
        )

        # Move network to GPU!
        self.policy_network.to(self.device)

        # Initialize weights using Xavier initialization and biases to zero
        self._initialize_weights()

        # The optimiser for the policy network, used to update policy weights
        self.optimiser = Adam(self.policy_network.parameters(), lr=alpha)

        # Whether to select an action stochastically or deterministically
        self.stochastic = stochastic

    def _initialize_weights(self):
        for layer in self.policy_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Ensure the last layer outputs logits close to zero
        last_layer = self.policy_network[-1]
        if isinstance(last_layer, nn.Linear):
            with torch.no_grad():
                last_layer.weight.fill_(0)
                last_layer.bias.fill_(0)

    """ Select an action using a forward pass through the network """

    def select_action(self, state):
        # I like the 'observation' terminology better, where this incoming state is an observation of the global environment state concatenated with our agent data (position index)
        # i.e. obs = flattened global density grid + flattened max uncertainty grid + current agent position (boresight index) = state vector 'state'

        # Convert the state into a tensor so it can be passed into the network
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action_logits = self.policy_network(state) # This should be an observation of the entire state space, possibly concatenated with our current position or affected by a cosine^4 or 1/r^2 falloff or something centered on our location??????

        action_distribution = Categorical(logits=action_logits)
        # Sample an action according to the probability distribution
        return action_distribution.sample().item() # This should give us the new position of the boresight!!!!

    """ Get the probability of an action being selected in a state """
    def get_probability(self, state, action):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_logits = self.policy_network(state)

        # A softmax layer turns action logits into relative probabilities
        probabilities = F.softmax(input=action_logits, dim=-1).tolist()
        # Convert from a tensor encoding back to the action space
        return probabilities[action]

    # This function evaluates a full episode trajectory!!!!
    def evaluate_actions(self, states, actions):
        action_logits = self.policy_network(states)
        action_distribution = Categorical(logits=action_logits)
        log_prob = action_distribution.log_prob(actions.squeeze(-1))
        return log_prob.view(1, -1)

    # def update(self, states, actions, deltas):
    #     # Convert to tensors to use in the network
    #     deltas = torch.as_tensor(deltas, dtype=torch.float32, device=self.device)
    #     states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
    #     actions = torch.as_tensor(actions, device=self.device)

    #     action_log_probs = self.evaluate_actions(states, actions)

    #     # Construct a loss function, using negative because we want to descend,
    #     # not ascend the gradient
    #     loss = -(action_log_probs * deltas).sum()
    #     self.optimiser.zero_grad()
    #     loss.backward()

    #     # Take a gradient descent step
    #     self.optimiser.step()

    def update(self, states, actions, rewards, gamma):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        
        R = 0
        policy_loss = []
        returns = deque()
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.appendleft(R)
        returns = torch.tensor(returns)

        eps = np.finfo(np.float32).eps.item()
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Compute log_probs of actions given states (evaluate_actions?)
        log_probs = self.evaluate_actions(states, actions)

        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimiser.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimiser.step()

    def save(self, filename):
        torch.save(self.policy_network.state_dict(), filename)

    @classmethod
    def load(cls, state_space, action_space, filename):
        policy = cls(state_space, action_space)
        policy.policy_network.load_state_dict(torch.load(filename))
        return policy

# class MarkovDecisionProcessDeepPolicyGradientPlanner:
#     def __init__(self, agent, mdp, policies) -> None:
#         self.agent = agent
#         self.mdp = mdp
#         self.policies = policies

#         if len(self.mdp.observers) != len(self.policies):
#             raise Exception(f"Number of observers managed by the MDP ({len(self.mdp.observers)}) is different from the number of policies supplied ({self.policies}). These must match.")

#     """ Generate and store an entire episode trajectory to use to update the policy """

#     def execute(self, episodes=100, max_num_steps_per_episode=50):

#         for episode in tqdm(range(episodes)):

#             # This appends data to self.mdp.episode_metadata list of all episode observer records for every episode
#             records = self.mdp.run_episode(self.policies, max_num_steps_per_episode)

#             # For each observer, update its corresponding policy by ID
#             for record in records:
#                 rewards = record.rewards
#                 states = record.rl_observation_vectors # This should be the chain of observation vectors!!!!!
#                 actions = record.actions
                
#                 deltas = self.calculate_deltas(rewards)

#                 self.policies[record.id].update(states, actions, deltas)

#         # Return the raw data which we can then investigate and plot to show convergence of policy over time
#         return self.mdp.episode_metadata

#     def calculate_deltas(self, rewards):
#         """
#         Generate a list of the discounted future rewards at each step of an episode
#         Note that discounted_reward[T-2] = rewards[T-1] + discounted_reward[T-1] * gamma.
#         We can use that pattern to populate the discounted_rewards array.
#         """
#         T = len(rewards)
#         discounted_future_rewards = np.zeros(T)

#         # The final discounted reward is the reward you get at that step
#         discounted_future_rewards[-1] = rewards[-1]
#         for t in reversed(range(0, T - 1)):
#             discounted_future_rewards[t] = (
#                 rewards[t]
#                 + discounted_future_rewards[t + 1] * self.mdp.get_discount_factor()
#             )
#         deltas = (self.mdp.get_discount_factor() ** np.arange(T)) * discounted_future_rewards
#         return deltas

##########################################################################################

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
def viz_stack(hms, typeof='density', to_disk=False, filename='test'):
    fig, ax = plt.subplots()
    ax.set_title(f"Crowd {typeof} at $t=${0}", color='black')
    ax.set_ylabel(r"$y$ [m]", color='black')
    ax.set_xlabel(r"$x$ [m]", color='black')
    ax.imshow(hms[:,:,0], cmap="inferno", origin='lower', vmax=hms.max())

    # Define the animation function
    def update(frame):
        ax.cla()
        ax.set_title(f"Crowd {typeof} at $t=${frame}", color='black')
        ax.set_ylabel(r"$y$ [m]", color='black')
        ax.set_xlabel(r"$x$ [m]", color='black')
        ax.imshow(hms[:,:,frame], cmap="inferno", origin='lower', vmax=hms.max())

        return ax

    # Create the animation object
    ani = FuncAnimation(fig, update, frames=list(range(1,hms.shape[-1]))) #, interval=20, blit=True)
    doc = ani.to_jshtml()
    if to_disk:
        with open(filename+'.html', 'w') as f:
            f.write(doc)
    return doc

if __name__=="__main__":
    import sys
    import time
    import matplotlib.pyplot as plt

    # Load optional note
    note = ''.join(sys.argv[1:])

    # Init Crowd Simulator
    Training_mode = True
    PLANNER_TYPE = 'mdp' # Takes either 'greedy' | 'mdp'
    MODEL_FILE = None # "some_file.v1.ptm"

    crowd_sim = Simulation(Training_mode, headless=True)
    
    # with PygameRecord("output.gif", FRAME_RATE) as recorder:
    
    # Init Manager
    if PLANNER_TYPE == 'greedy':

        manager = Manager([GreedyPlanner(agent) for agent in crowd_sim.drones], crowd_sim)

    elif PLANNER_TYPE == 'mdp':

        state_space = crowd_sim.global_map.density.size + 2
        action_space = crowd_sim.global_map.density.size
        manager = Manager([DroneMDPPlanner(agent, DeepNeuralNetworkPolicy(state_space, action_space, 256), state_space, action_space, discount_factor=0.91) for agent in crowd_sim.drones], crowd_sim)

        if MODEL_FILE:
            manager.load(MODEL_FILE) # load weights from some file
        else:
            history = manager.train(2000, 15) # otherwise, train with some number of episodes

        # Save all policies + the history!
        manager.save(history, note)

    # Make Cumulative Reward plot
    cumrewards = np.asarray([[np.sum(pr.rewards) for pr in records.records.values()] for records in history])
    plt.plot(np.convolve(cumrewards, np.ones(15)/15., 'same'))
    plt.legend(labels=range(len(crowd_sim.drones)))
    plt.title('Smoothed Reward Trace')
    plt.xlabel('Episode #')
    plt.ylabel('Episode Return')
    plt.show()

    # # Practice for heuristic calculations
    # crowd_sim.step()
    # manager.step()

    # # Velocity (mag and phase)
    # hm = crowd_sim.global_map.get_density_map()
    # (vm, vp) = crowd_sim.global_map.get_velocity_mag_and_phase_map()
    # # (curl, div) = crowd_sim.global_map.get_curl_div_map()
    # # (skew, kurt) = crowd_sim.global_map.get_skew_curt_map()

    # # Run loop (THINK OF THIS AS A SINGLE EPISODE)
    # EPOCH = 0
    # hms = np.zeros((crowd_sim.global_map.mesh_shape[0], crowd_sim.global_map.mesh_shape[1], 100))
    # vmag = np.zeros((crowd_sim.global_map.mesh_shape[0], crowd_sim.global_map.mesh_shape[1], 100))
    # vphase = np.zeros((crowd_sim.global_map.mesh_shape[0], crowd_sim.global_map.mesh_shape[1], 100))

    # hms[..., 0] = manager.simulator.global_map.get_density_map()
    # (vm, vp) = crowd_sim.global_map.get_velocity_mag_and_phase_map()
    # vmag[..., 0] = vm
    # vphase[..., 0] = vp

    # while EPOCH < 99:
    #     crowd_sim.step()
    #     manager.step()
    #     EPOCH += 1

    #     hms[..., EPOCH] = manager.simulator.global_map.get_density_map()
    #     (vm, vp) = crowd_sim.global_map.get_velocity_mag_and_phase_map()        
    #     vmag[..., EPOCH] = vm
    #     vphase[..., EPOCH] = vp

    # viz_stack(hms, True, 'Density')
    # viz_stack(vmag, True, 'VelocityMag')
    # viz_stack(vphase, True, 'VelocityPhase')

    # ###########################
    # # Curl and Density Test

