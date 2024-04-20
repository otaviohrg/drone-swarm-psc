from typing import Optional
from enum import Enum

import math
import random
import numpy as np
from scipy.spatial import distance

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

        self.counterStraight = 0
        self.angleStopTurning = random.uniform(-math.pi, math.pi)
        self.distStopStraight = random.uniform(10, 50)
        self.isTurning = False

        self.initial_flag = True
        self.initial_x = None
        self.initial_y = None
        self.builtReturnPath = False
        self.returnPath = []
        self.myVote = random.randint(0, 1000)
        self.voteInProgress = False
        self.voteResult = True
        self.voteRefreshCountdown = 0

    def define_message_for_all(self):
        if(len(self.graph.latest_edges) > 50): #share edges
            self.publish(np.array2string(self.graph.latest_edges), "EDGES")
            self.graph.latest_edges = np.array([]) #reset latest_edges

            #UNCOMMENT LINE BELOW TO USE INTERNAL COMM INSTEAD OF MQTT!!
            #return np.array2string(self.graph.latest_edges) 

    def vote(self):
        self.voteResult = True
        self.voteInProgress = True
        self.voteRefreshCountdown = 7
        self.publish(self.myVote, "VOTE")

    def on_message(self, client, userdata, msg):
        msg_content = msg.payload.decode()
        if(msg.topic == "EDGES"):
            edges = np.fromstring(msg, dtype=object, sep=',') # transform string to numpy array
            self.graph.add_edges(edges) # add edges to the graph
        else: #vote for person
            if(int(msg_content) < self.myVote): #I lose grasp vote :(
                self.voteResult = False
    
    def control(self):

        command_straight = {"forward": 1.0,
                            "lateral": 0.0,
                            "rotation": 0.0,
                            "grasper": 0}

        command_turn = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 1.0,
                        "grasper": 0}
        
        found_wounded, found_rescue_center, command_semantic = process_semantic_sensor(self)

        collided, collision_angles, far_angles, min_dist, max_dist = process_lidar_sensor(self)

        x, y = self.localization.get_gps_values(self)
        angle = self.localization.get_compass_values(self)

        prev_position, position, weight = self.localization.get_edge_info()
        if not (prev_position is None):
            self.graph.add_edge(prev_position[0], prev_position[1], position[0], position[1], weight, True)

        if self.initial_flag:
            self.initial_x = x
            self.initial_y = y
            self.initial_flag = False

        self.state = update_state(self.state, found_wounded, found_rescue_center, self.base.grasper.grasped_entities, self.voteResult)

        self.voteRefreshCountdown = max(0, self.voteRefreshCountdown - 1)
        if(self.voteRefreshCountdown == 0):
            self.voteInProgress = False
            self.voteResult = True

        if self.state is Activity.SEARCHING_WOUNDED:
            self.builtReturnPath = False
            self.counterStraight += 1

            if collided and not self.isTurning and self.counterStraight > self.distStopStraight:
                x_target, y_target = self.graph.rrt_steering(x, y, angle, far_angles)
                self.isTurning = True
                self.angleStopTurning = math.atan2( y_target-y, x_target-x )
                self.distStopStraight = distance.euclidean((x,y), (x_target, y_target))

            diff_angle = normalize_angle(self.angleStopTurning - angle)
            if self.isTurning and abs(diff_angle) < 0.2:
                self.isTurning = False
                self.counterStraight = 0

            if self.isTurning:
                return command_turn
            else:
                return command_straight

        elif self.state is Activity.GRASPING_WOUNDED:
            self.builtReturnPath = False
            command = command_semantic
            command["grasper"] = 1

        elif self.state is Activity.SEARCHING_RESCUE_CENTER:
            if self.builtReturnPath is False:
                self.returnPath = self.graph.shortest_path((x,y), (self.initial_x, self.initial_y))
                self.builtReturnPath = True
                #self.graph.plot_graph()
            while distance.euclidean([x,y], self.returnPath[0]) < 1.5 and len(self.returnPath) > 1:
                self.returnPath.pop(0)
            command = move_to_point(self.returnPath[0][0], self.returnPath[0][1], angle, x, y)
            command["grasper"] = 0

        elif self.state is Activity.DROPPING_AT_RESCUE_CENTER:
            self.builtReturnPath = False
            command = command_semantic
            command["grasper"] = 1

        self.localization.update_historic_commands(command)
        return command