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
from solutions.state_machine import Activity, update_state
from solutions.utilities.odometer_prediction import Odometer_prediction

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

        self.odometer_prediction = Odometer_prediction()

    def define_message_for_all(self):
        pass
    
    def control(self):

        if(self.odometer_prediction.initial_x is None):
            self.odometer_prediction.initial_x, self.odometer_prediction.initial_y = get_gps_values(self)

        command_placeholder = {"forward": 0.5, "lateral": 0.5, "rotation": 0.5}
        
        found_wounded, found_rescue_center, command_semantic = process_semantic_sensor(self)

        collided, collision_angles, far_angles, min_dist, max_dist = process_lidar_sensor(self)

        (dist , alpha , theta) = self.odometer_values()
        self.odometer_prediction.update(dist , alpha , theta)

        x = self.odometer_prediction.initial_x + self.odometer_prediction.integ_bruit_dx()
        y = self.odometer_prediction.initial_y + self.odometer_prediction.integ_bruit_dy()

        self.state = update_state(self.state, found_wounded, found_rescue_center, self.base.grasper.grasped_entities)

        if self.state is Activity.SEARCHING_WOUNDED:
            command = command_placeholder

        elif self.state is Activity.GRASPING_WOUNDED:
            command = command_semantic
            command["grasper"] = 1

        elif self.state is Activity.SEARCHING_RESCUE_CENTER:
            command = command_placeholder
            command["grasper"] = 1

        elif self.state is Activity.DROPPING_AT_RESCUE_CENTER:
            command = command_semantic
            command["grasper"] = 1

        x_measured, y_measured = get_gps_values(self)
        print(f"GPS ({x_measured}, {y_measured}) SPLINE ({x}, {y})")

        return command