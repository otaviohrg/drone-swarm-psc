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
from solutions.localization.kalman_filter import KalmanFilter
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

        #Initialise Kalman filter parameters
        initial_gps = self.gps_values()
        initial_compass = self.compass_values()
        initial_state = np.array([initial_gps[0], initial_gps[1], initial_compass, 0, 0, 0, 0, 0])  # x=y=vx=vy=0 (drone is initially at rest)
        initial_covariance = np.eye(8)  # Identity matrix
        noisenogps = np.eye(5)*0.01
        a,b=1,1
        noisewtgps = np.diag([a]*3+[b]*5)
        self.kalmanFilter = KalmanFilter(initial_state, initial_covariance, noisenogps,noisewtgps, self)

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
        self.voteRefreshCountdown = 45

    def define_message_for_all(self):
        #if(len(self.graph.latest_edges) > 10): #share edges
        #    print(f"I SHARED MY EDGES {self.myVote}")
        #    self.publish(np.array2string(self.graph.latest_edges), "EDGES")
        #    self.graph.latest_edges = np.array([]) #reset latest_edges
        pass
            #UNCOMMENT LINE BELOW TO USE INTERNAL COMM INSTEAD OF MQTT!!
            #return np.array2string(self.graph.latest_edges) 

    def vote(self):
        self.voteResult = True
        self.voteInProgress = True
        self.voteRefreshCountdown = 100
        self.publish(self.myVote, "VOTE")
        print(f"I VOTED!! {self.myVote}")

    def on_message(self, client, userdata, msg):
        msg_content = msg.payload.decode()
        if(msg.topic == "EDGES"):
            print(f"I RECEIVED EDGES {self.myVote}")
            edges = np.fromstring(msg, dtype=object, sep=',') # transform string to numpy array
            self.graph.add_edges(edges) # add edges to the graph
        else: #vote for person
            if(int(msg_content) < self.myVote): #I lose grasp vote :(
                print(f"I JUST LOST A VOTE!!!{self.myVote}")
                self.voteResult = False
            else:
                print(f"I AM STILL WINNING!! {self.myVote}")
    
    def control(self):

        command_right = {"forward": 0.7,
                            "lateral": 0.0,
                            "rotation": 0.5,
                            "grasper": 0}

        command_left = {"forward": 0.7,
                        "lateral": 0.0,
                        "rotation": -0.5,
                        "grasper": 0}
        
        nearby_drone, found_wounded, found_rescue_center, command_semantic = process_semantic_sensor(self)

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

        if self.state is Activity.SEARCHING_WOUNDED:
            self.builtReturnPath = False
            self.counterStraight += 1

            x_target, y_target = self.graph.rrt_steering(x, y, angle, far_angles)

            command = move_to_point(x_target, y_target, angle, x, y)
        


        elif self.state is Activity.GRASPING_WOUNDED:
            self.builtReturnPath = False
            if(nearby_drone):
                self.vote()
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
            command["grasper"] = self.voteResult and (not self.voteInProgress)

        elif self.state is Activity.DROPPING_AT_RESCUE_CENTER:
            self.builtReturnPath = False
            command = command_semantic
            command["grasper"] = 1

        self.localization.update_historic_commands(command)
        return command