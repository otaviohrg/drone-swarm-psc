from typing import Optional
from enum import Enum

import random
import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle

from solutions.process_semantic_sensor import process_semantic_sensor
from solutions.process_gps import get_gps_values
from solutions.kalman_filter import KalmanFilter
from solutions.state_machine import Activity, update_state
from solutions.mqtt_utilities import MyDroneMQTT

class MyDroneEval(DroneAbstract):

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)
        
        # The state is initialized to searching wounded person
        self.state = Activity.SEARCHING_WOUNDED

        #Connect to MQTT broker
        self.mqtt = MyDroneMQTT()

        # Initialize Kalman filter parameters
        initial_state = np.array([0,0,0,0])  # x, y, vx, vy (drone is initially at rest)
        initial_covariance = np.eye(4)  # Identity matrix
        process_noise = np.eye(4) * 0.01  # Process noise covariance matrix
        measurement_noise = np.eye(2) * 0.1  # Measurement noise covariance matrix

        self.kalman_filter = KalmanFilter(initial_state, initial_covariance, process_noise, measurement_noise)

    def define_message_for_all(self):
        x = self.kalman_filter.state[0]
        y = self.kalman_filter.state[1]
        self.mqtt.publish(f"{self.mqtt.client_id} {x} {y}")

    def get_dynamic_state(self):
        x, y, vx, vy = self.kalman_filter.state.flatten()
        return x, y, vx, vy
    
    def control(self):

        x, y, vx, vy = self.get_dynamic_state()

        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0}
        
        found_wounded, found_rescue_center, command_semantic = process_semantic_sensor(self)

        self.state = update_state(self.state, found_wounded, found_rescue_center, self.base.grasper.grasped_entities)

        if self.state is Activity.SEARCHING_WOUNDED:
            command["forward"] = random.uniform(0, 1)
            command["lateral"] = random.uniform(0, 1)

        elif self.state is Activity.GRASPING_WOUNDED:
            command = command_semantic
            command["grasper"] = 1

        elif self.state is Activity.SEARCHING_RESCUE_CENTER:
            command["forward"] = random.uniform(0, 1)
            command["lateral"] = random.uniform(0, 1)
            command["grasper"] = 1

        elif self.state is Activity.DROPPING_AT_RESCUE_CENTER:
            command = command_semantic
            command["grasper"] = 1

        x_measured, y_measured = get_gps_values(self)
        effects = np.array([command["forward"], command["lateral"]])

        self.kalman_filter.drone_update(x_measured, y_measured, effects)

        #print(f"x = {x} \t y = {y} \t vx = {vx} \t vy = {vy} \t fwr = {command['forward']} \t lat = {command['lateral']}")
        #self.send_message(x, y)

        return command