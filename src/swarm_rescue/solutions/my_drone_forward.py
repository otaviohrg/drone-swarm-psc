""" Simple drone controller which will just move forward. 
    It will be used as a placeholder to instantiate the map necessary for the training in drone_env.py"""

import math
import numpy as np
import random
from typing import Optional


# SPG tools
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.utils import normalize_angle, circular_mean

from solutions.state_machine import Activity
from solutions.sensors.process_semantic_sensor import process_semantic_sensor

from stable_baselines3 import PPO, A2C, DQN

class MyDroneForward(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)
        self.state = Activity.SEARCHING_WOUNDED
    
    def define_message_for_all(self):
        pass

    def control(self):
        """
        The Drone will move forward
        """
        command_straight = {"forward": 1.0,
                            "lateral": 0.0,
                            "rotation": 0.0,
                            "grasper": 0}
        

        return command_straight
    
