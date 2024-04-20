from typing import Optional
from enum import Enum

import random
import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle

from solutions.sensors.process_semantic_sensor import process_semantic_sensor
from solutions.sensors.process_lidar_sensor import process_lidar_sensor
from solutions.decision.move_to_point import move_to_point
from solutions.decision.random_walks import ballistic, levy_flight
from solutions.utilities.mqtt_utilities import MyDroneMQTT
from solutions.utilities.graph_utilities import MyDroneGraph
from solutions.localization.localization import MyDroneLocalization
from solutions.state_machine import Activity, update_state

class MyDroneEval(DroneAbstract, MyDroneMQTT):

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)
        
        MyDroneMQTT.__init__(self)  # Initialise connection to MQTT broker
        
        # The state is initialized to searching wounded person
        self.state = Activity.SEARCHING_WOUNDED

        # Initialise Graph used as map
        self.graph = MyDroneGraph()

        # Initialise model for localization
        self.localization = MyDroneLocalization()

    def define_message_for_all(self):
        if(len(self.graph.latest_edges) > 50): #share edges
            self.mqtt.publish(np.array2string(self.graph.latest_edges))
            self.graph.latest_edges = np.array([]) #reset latest_edges

            #UNCOMMENT LINE BELOW TO USE INTERNAL COMM INSTEAD OF MQTT!!
            #return np.array2string(self.graph.latest_edges) 

    def on_message(self, client, userdata, msg):
        msg_content = msg.payload.decode()
        edges = np.fromstring(msg, dtype=object, sep=',') # transform string to numpy array
        self.graph.add_edges(edges) # add edges to the graph
    
    def control(self):
        
        found_wounded, found_rescue_center, command_semantic = process_semantic_sensor(self)

        collided, collision_angles, far_angles, min_dist, max_dist = process_lidar_sensor(self)

        x, y = self.localization.get_gps_values(self)
        angle = self.localization.get_compass_values(self)

        self.state = update_state(self.state, found_wounded, found_rescue_center, self.base.grasper.grasped_entities)

        if self.state is Activity.SEARCHING_WOUNDED:
            command = 

        elif self.state is Activity.GRASPING_WOUNDED:
            command = command_semantic
            command["grasper"] = 1

        elif self.state is Activity.SEARCHING_RESCUE_CENTER:
            command = 
            command["grasper"] = 1

        elif self.state is Activity.DROPPING_AT_RESCUE_CENTER:
            command = command_semantic
            command["grasper"] = 1

        return command