from typing import Optional
from enum import Enum

import random
import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle

from solutions.sensors.process_semantic_sensor import process_semantic_sensor
from solutions.sensors.process_lidar_sensor import process_lidar_sensor, command_lidar
from solutions.sensors.process_gps import get_gps_values
from solutions.utilities.mqtt_utilities import MyDroneMQTT
from solutions.kalman_filter.kalman_filter import KalmanFilter
from solutions.state_machine import Activity, update_state

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

        #Initialise connection to MQTT broker
        self.mqtt = MyDroneMQTT()

        self.kalman_filter = None

    def define_message_for_all(self):
        pass
        #x = self.kalman_filter.state[0]
        #y = self.kalman_filter.state[1]
    
    def control(self):

        if(self.kalman_filter is None):
            # Initialize Kalman filter parameters
            initial_gps = self.gps_values()
            initial_compass = self.compass_values()
            initial_state = np.array([initial_gps[0], initial_gps[1], initial_compass, 0, 0, 0, 0, 0])  # x=y=vx=vy=0 (drone is initially at rest)
            initial_covariance = np.eye(8)  # Identity matrix
            measurement_noise = np.eye(5)*0.01

            self.kalman_filter = KalmanFilter(initial_state, initial_covariance, measurement_noise, self)


        #Retrieve dynamics information (x,y,vx,vy) from Kalman filter
        x, y, theta, vx, vy, vtheta, ax, ay  = self.kalman_filter.state.flatten()
        
        found_wounded, found_rescue_center, command_semantic = process_semantic_sensor(self)

        collided, collision_angles, far_angles, min_dist, max_dist = process_lidar_sensor(self)

        self.state = update_state(self.state, found_wounded, found_rescue_center, self.base.grasper.grasped_entities)

        if self.state is Activity.SEARCHING_WOUNDED:
            command = command_lidar(far_angles, vx, vy)

        elif self.state is Activity.GRASPING_WOUNDED:
            command = command_semantic
            command["grasper"] = 1

        elif self.state is Activity.SEARCHING_RESCUE_CENTER:
            command = command_lidar(far_angles, vx, vy)
            command["grasper"] = 1

        elif self.state is Activity.DROPPING_AT_RESCUE_CENTER:
            command = command_semantic
            command["grasper"] = 1

        self.kalman_filter.drone_update(command)

        x_measured, y_measured = get_gps_values(self)
        print(f"GPS ({x_measured}, {y_measured}) KALMAN ({x}, {y})")

        return command