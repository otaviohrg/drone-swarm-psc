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

class MyDroneModel(DroneAbstract):
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
        """ command_straight = {"forward": 1.0,
                            "lateral": 0.0,
                            "rotation": 0.0,
                            "grasper": 0}
        
        # Print the sensors data to check the values
        a, b, _ = process_semantic_sensor(self)
        print(int(a))
        print(type(b)) """

        map_shape = (800, 500)
        # Load the model
        model = A2C.load("a2c_saved", print_system_info=True)

        # get the observation from the drone
        lidar_values = self.lidar_values()
        lidar_values = np.clip(lidar_values, 0, 300)

        gps_position = self.gps_values()
        gps_position = np.array([np.clip(gps_position[0],-map_shape[0]/2, map_shape[0]/2), np.clip(gps_position[1],-map_shape[1]/2, map_shape[1]/2)])

        compass_angle = np.array([self.compass_values()])
        compass_angle = np.clip(compass_angle, -math.pi, math.pi)

        odometer = self.odometer_values()
        odometer = np.array([np.clip(odometer[0],0, 2*np.sqrt(map_shape[0]**2 + map_shape[1]**2)), np.clip(odometer[1],-math.pi,math.pi), np.clip(odometer[2],-math.pi,math.pi)])

        found_wounded, found_rescue_ctr, _ = process_semantic_sensor(self)

        observation = {
            "lidar": np.float32(lidar_values),
            "semantic_wounded": int(found_wounded),
            "semantic_rescue_center": int(found_rescue_ctr),
            "gps": np.float32(gps_position),
            "compass": np.float32(compass_angle),
            "odometer": np.float32(odometer)
        }

        action , _ = model.predict(observation, deterministic=True)

        command_straight = {"forward": action[0],
                            "lateral": action[1],
                            "rotation": action[2]}

        if self.state == Activity.SEARCHING_WOUNDED:
            command_straight["grasper"] = 0
        else:
            command_straight["grasper"] = 1


        return command_straight
    
