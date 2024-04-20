""" import sys
sys.path.insert(0, '../../swarm_rescue' ) """


# Importation of the necessary libraries
import math
import numpy as np
from typing import Optional


#RL tools
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType

# SB3 tools
from stable_baselines3.common.env_checker import check_env

#SPG tools

from maps.map_simple import MyMapSimple
#from maps.map_intermediate_01 import MyMapIntermediate01

# Drone type to instantiate
from swarm_rescue.solutions.my_drone_forward import MyDroneForward
from swarm_rescue.solutions.sensors import process_semantic_sensor
from swarm_rescue.solutions.sensors import process_lidar_sensor

from swarm_rescue.solutions.state_machine import Activity, update_state



class DroneEnv(gym.Env):
    """
        DroneEnv is a Gym environment which will be used to train drones.
        The agent in this environment is a drone which will use his sensors to explore the map.
        The map is a playground which contains objects (walls, wounded persons, rescue center,...) and even other drones.
        But for a startup, we will consider that there is just one drone in the map and one wounded person.
        They will always be one rescue center in every case and the drone position is initialied near the rescue center.
    """

    def __init__(self):

        super(DroneEnv, self).__init__()

        # create the map which contains all the information about the playground as the position of walls and number of drones,...
        self.map = MyMapSimple()
        self.playground = self.map.construct_playground(MyDroneForward)

        # define observations and actions spaces

        # action space
        self.action_space = spaces.Dict({
            "forward":spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) ,
            "lateral": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "rotation": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "grasper": spaces.Discrete(2, shape=(1,))
        })

        # observation space 
        # The observations are made of the sensors data of the drone

        LIDAR_RESOLUTION = 181
        SEMANTIC_RESOLUTION = 35
        map_shape = self.map._size_area
        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(low=0, high=300, shape=(LIDAR_RESOLUTION,), dtype=np.float32),
            "semantic": spaces.Dict({"found_wounded": spaces.Discrete(2, shape=(1,)), "found_rescue_center": spaces.Discrete(2, shape=(1,))}),
            "gps": spaces.Box(low=[-map_shape[0]/2, -map_shape[1]/2], high=[map_shape[0]/2,map_shape[1]/2], shape=(2,), dtype=np.float32),
            "compass": spaces.Box(low=-math.PI, high=math.PI, shape=(1,), dtype=np.float32),
            "odometer": spaces.Box(low=[0,-math.PI, -math.PI], high=[2*np.sqrt(map_shape[0]**2 + map_shape[1]**2), math.PI, math.PI], shape=(3,), dtype=np.float32)
        })

    def reset(self, seed=None):
        # replace the drone in initial position, reset the wounded person position and return the initial observation
        self.map = MyMapSimple()
        self.playground = self.map.construct_playground(MyDroneForward)
        self.playground.reset()

        # Get the observation from the drone
        
        observation = self.get_observation()
        info = None
        return observation, info

    def get_observation(self):
        map_shape = self.map._size_area

        # get the observation from the drone
        lidar_values = self.map._drones[0].lidar_values()
        lidar_values = np.clip(lidar_values, 0, 300)

        gps_position = self.map._drones[0].measured_gps_position()
        gps_position = [np.clip(gps_position[0],-map_shape[0]/2, map_shape[0]/2), np.clip(gps_position[1],-map_shape[1]/2, map_shape[1]/2)]

        compass_angle = self.map._drones[0].measured_compass_angle()
        compass_angle = np.clip(compass_angle, -math.PI, math.PI)

        odometer = self.map._drones[0].odometer_values()
        odometer = [np.clip(odometer[0],0, 2*np.sqrt(map_shape[0]**2 + map_shape[1]**2)), np.clip(odometer[1],-math.PI,math.PI), np.clip(odometer[2],-math.PI,math.PI)]

        found_wounded, found_rescue_ctr, _ = process_semantic_sensor(self.map._drones[0])

        observation = {
            "lidar": lidar_values,
            "semantic": {"found_wounded": found_wounded, "found_rescue_center": found_rescue_ctr},
            "gps": gps_position,
            "compass": np.array([compass_angle]),
            "odometer": odometer
        }
        return observation
        


    def step(self, action):
        # take an action and return the new state, reward, done and info
        old_state = self.map._drones[0].state

        # take the action

        # Extract the values from the action dictionary
        forward = action["forward"][0]
        lateral = action["lateral"][0]
        rotation = action["rotation"][0]
        grasper = action["grasper"][0]
        tmp = {"forward": forward, "lateral": lateral, "rotation": rotation, "grasper": grasper}
        command = {}
        command[self.map._drones[0].identifier] = tmp
        fake_obs, fake_msg, fake_rew, done = self.playground.step(command)

        # get the new observation
        observation = self.get_observation()

        # new state
        new_state = update_state(old_state, observation["semantic"]["found_wounded"], observation["semantic"]["found_rescue_center"], self.map._drones[0].base.grasper.grasped_entities)
        self.map._drones[0].state = new_state

        

        collision, collision_angles, far_angles, min_dist, max_dist = process_lidar_sensor(self.map._drones[0])

        # reward computation
        reward = 0
        if collision:
            reward -= 1

        if old_state == Activity.SEARCHING_WOUNDED:
            if observation["semantic"]["found_wounded"] == 1: #the drone succedded to find the wounded person
                reward += 5
        elif old_state == Activity.GRASPING_WOUNDED:
            if self.map._drones[0].state == Activity.SEARCHING_RESCUE_CENTER : # the drone succedded to grasp the wounded person
                reward += 0.5
        elif old_state == Activity.SEARCHING_RESCUE_CENTER:
            if observation["semantic"]["found_rescue_center"] == 1: # the drone succedded to find the rescue center
                reward += 10
        else:
            reward += 0

        info = None

        return observation, reward, done, info
    
    

    def render(self, mode='human'):
        # render the environment
        pass

    def close(self):
        # close the environment
        pass

