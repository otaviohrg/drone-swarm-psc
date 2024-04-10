#RL tools
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType

#Math tools
import numpy as np



#SPG tools
from spg.playground import Playground
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle

import collections

def replace_values(d):
    for k, v in d.items():
        if isinstance(v, collections.OrderedDict):
            d[k] = replace_values(v)
        else:
            d[k] = d[k] * 0
    return d


class SwarmEnv(gym.Env):
    """
        This is a wrapper class for creating a gym environment to train the drones giving a map.
        For a startup, we will consider just one drone and implement RL algorithms to train it.
    """
    def __init__(self,
                 playground: Playground):
        self.map = playground
        self.map.reset()
    
    @property
    def action_space(self):
        return spaces.Dict(
            {agent.name: agent.agent_action_space for agent in self.map.agents}
        )
    
    @property
    def observation_space(self):
        return spaces.Dict(
            {agent.name: agent.agent_observation_space for agent in self.map.agent}
        )
    
    def reset(self):
        """
            Reset the map
            return the map
        """
        self.map.reset()
        return self.map._compute_observations()

    def zero_action_space(self):
        action = self.action_space.sample()
        zero_dict = replace_values(action)
        return zero_dict
    

    #implement compute observation
    #implement compute reward
    
    def compute_observation(self):
        return map._compute_observation()

    def _compute_reward(self):
        return {agent: agent.reward for agent in self.map.agents}

    def step(self, action:ActType):
        """
            Take a step in the map
            return the observation, reward, done, info
        """
        self.map.step()
        #To complete

    