# pylint: disable=import-error
import math
import random
import numpy as np
from typing import Optional
from enum import Enum

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.utils import normalize_angle, circular_mean

from solutions.RRT import RRT
from scipy.spatial import distance

#to compute best and worst angles in sensor output
import heapq

class MyDroneEval(DroneAbstract):

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)
        
        # The state is initialized to searching wounded person
        self.state = self.Activity.SEARCHING_WOUNDED

        # values used by the control function
        self.gps_x = self.measured_gps_position()[0]
        self.gps_y = self.measured_gps_position()[1]
        self.compass_angle = self.measured_compass_angle()
        self.historic_gps = []
        self.historic_angle = []

        self.RRT = RRT()
        self.currentPoint = (self.gps_x, self.gps_y)
        self.RRT.addNode(self.currentPoint, None) #add initial node (root) to tree
        self.previousPoint = None

        self.builtWayBack = False
        self.path = []

        # parameters for controlling steps
        self.counterStraight = 0
        self.angleStopTurning = 0
        self.distStopStraight = 0
        self.isTurningLeft = False
        self.isTurningRight = False

    def move_to_point(self, destx, desty):#assumes gps/compass are available
        command = {"forward": 0.5,
                    "lateral": 0.0,
                    "rotation": 0.0,
                    "grasper": 0.0}
        #CHOOSE BEST DIRECTION 3 possibilities: rotation=1, rotation=0 or rotation=-1

        #rotation = 1 => 0,2 rad per step => takes 32 steps to do the whole circle at rotation = 1
        rotation_discretization_parameter = 64
        rotation_speed = 32/rotation_discretization_parameter

        dest = (destx, desty)
        candidates = [-1, 0, 1]
        criteria = []
        for i in candidates:
            direction_angle = self.compass_angle + 2*i*math.pi/rotation_discretization_parameter
            point_in_direction = (self.gps_x + math.cos(direction_angle), self.gps_y + math.sin(direction_angle))
            criteria.append(distance.euclidean(point_in_direction, dest))

        i_chosen = candidates[np.argmin(criteria)]
        command["rotation"] = i_chosen*rotation_speed

        return command
        
    def get_gps_values(self):
        '''Return GPS values if available, else use historic_gps to predict position'''
        scaling_factor = 60
        if self.gps_is_disabled():
            last_position = self.historic_gps[-1]
            last_angle = self.historic_angle[-1]
            if self._is_turning():
                return last_position[0], last_position[1]
            else:
                return last_position[0] + math.cos(last_angle)/scaling_factor, last_position[1] + math.sin(last_angle)/scaling_factor
        else:
            return self.measured_gps_position()[0]/scaling_factor, self.measured_gps_position()[1]/scaling_factor

    def update_gps_values(self):
        self.gps_x, self.gps_y = self.get_gps_values()
        self.historic_gps.append((self.gps_x, self.gps_y))
        if len(self.historic_gps) > 300:
            self.historic_gps.pop(0)

    def get_compass_values(self):
        if self.compass_is_disabled():
            last_angle = self.historic_angle[-1]
            if self._is_turning():
                return last_angle + 0.2
            else:
                return last_angle
        else:
            return self.measured_compass_angle()

    def update_compass_values(self):
        self.compass_angle = self.get_compass_values()
        self.historic_angle.append(self.compass_angle)
        if len(self.historic_angle) > 300:
            self.historic_angle.pop(0)


    def _is_turning(self):
        return self.isTurningLeft or self.isTurningRight

    def process_lidar_sensor(self):
        """
        Returns True if the drone collided an obstacle
        """

        lidar_sensor = self.lidar()
        values = lidar_sensor.get_sensor_values().tolist()
        ray_angles = lidar_sensor.ray_angles
        size = lidar_sensor.resolution

        collision_angles = 0
        far_angles = 0
        min_dist = 0
        max_dist = 0
        if size != 0:

            # collision_angles: angles with smallest distances
            collision_angle_indexes = [values.index(i) for i in heapq.nsmallest(10, values)]
            collision_angles = ray_angles[collision_angle_indexes]
            min_dist = min(values)

            far_angle_indexes = []
            for i in range(size):
                if(values[i] >= 120): # far_angles: angles with big enough distances
                    far_angle_indexes.append(i)
            far_angles = ray_angles[far_angle_indexes]
            max_dist = max(values)

        if self.lidar_values() is None:
            return False, 0

        collided = False
        dist = min(self.lidar_values())

        if dist < 100:
            collided = True

        return collided, collision_angles, far_angles, min_dist, max_dist

    def can_see_rescue_center(self):
        detection_semantic = self.semantic_values()
        for data in detection_semantic:
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    return True
        return False
    
    def process_semantic_sensor(self):
        """
        According to his state in the state machine, the Drone will move towards a wound person or the rescue center
        """
        command = {"forward": 0.5,
                   "lateral": 0.0,
                   "rotation": 0.0}
        angular_vel_controller_max = 1.0

        detection_semantic = self.semantic_values()
        best_angle = 0

        found_wounded = False
        if (self.state is self.Activity.SEARCHING_WOUNDED
            or self.state is self.Activity.GRASPING_WOUNDED) \
                and detection_semantic is not None:
            scores = []
            for data in detection_semantic:
                # If the wounded person detected is held by nobody
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                    found_wounded = True
                    v = (data.angle * data.angle) + \
                        (data.distance * data.distance / 10 ** 5)
                    scores.append((v, data.angle, data.distance))

            # Select the best one among wounded persons detected
            best_score = 10000
            for score in scores:
                if score[0] < best_score:
                    best_score = score[0]
                    best_angle = score[1]

        found_rescue_center = False
        is_near = False
        angles_list = []
        if (self.state is self.Activity.SEARCHING_RESCUE_CENTER
            or self.state is self.Activity.DROPPING_AT_RESCUE_CENTER) \
                and detection_semantic:
            for data in detection_semantic:
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    found_rescue_center = True
                    angles_list.append(data.angle)
                    is_near = (data.distance < 50)

            if found_rescue_center:
                best_angle = circular_mean(np.array(angles_list))

        if found_rescue_center or found_wounded:
            # simple P controller
            # The robot will turn until best_angle is 0
            kp = 2.0
            a = kp * best_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a * angular_vel_controller_max

            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.2

        if found_rescue_center and is_near:
            command["forward"] = 0
            command["rotation"] = random.uniform(0.5, 1)

        return found_wounded, found_rescue_center, command

    def control(self):
        '''Returns command to control drone at each step of the simulation'''

        #############
        # TRANSITIONS OF THE STATE MACHINE
        #############

        found_wounded, found_rescue_center, command_semantic = self.process_semantic_sensor()

        if self.state is self.Activity.SEARCHING_WOUNDED and found_wounded:
            self.state = self.Activity.GRASPING_WOUNDED

        elif self.state is self.Activity.GRASPING_WOUNDED and self.base.grasper.grasped_entities:
            self.state = self.Activity.SEARCHING_RESCUE_CENTER

        elif self.state is self.Activity.GRASPING_WOUNDED and not found_wounded:
            self.state = self.Activity.SEARCHING_WOUNDED

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER and found_rescue_center:
            self.state = self.Activity.DROPPING_AT_RESCUE_CENTER

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not self.base.grasper.grasped_entities:
            self.state = self.Activity.SEARCHING_WOUNDED

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not found_rescue_center:
            self.state = self.Activity.SEARCHING_RESCUE_CENTER

        #print("state: {}, can_grasp: {}, grasped entities: {}".format(self.state.name, self.base.grasper.can_grasp, self.base.grasper.grasped_entities))

        ##########
        # COMMANDS FOR EACH STATE
        ##########

        command_straight = {"forward": 1.0,
                            "lateral": 0.0,
                            "rotation": 0.0,
                            "grasper": 0}
        command_turn_left = {"forward": 0.0,
                            "lateral": 0.0,
                            "rotation": 1.0,
                            "grasper": 0}
        command_turn_right = {"forward": 0.0,
                            "lateral": 0.0,
                            "rotation": -1.0,
                            "grasper": 0}

        collided, collision_angles, far_angles, min_dist, max_dist = self.process_lidar_sensor()

        self.update_gps_values()
        self.update_compass_values()

        self.previousPoint = self.currentPoint
        self.currentPoint = (round(self.gps_x), round(self.gps_y))

        #print(f"Let's check {self.currentPoint}")
        
        if(not self.RRT.alreadyVisited(self.currentPoint)): #add node to Tree and edge to parent
            print(f"Added to tree {self.currentPoint} parent {self.previousPoint}")
            previousNode = self.RRT.getNodeIndex(self.previousPoint)
            self.RRT.addNode(self.currentPoint, self.RRT.nodes[previousNode])
            self.RRT.addEdge(self.previousPoint, self.currentPoint)
        
        if(self.can_see_rescue_center()):
            self.RRT.rescueCenterNode = self.RRT.getNodeIndex(self.currentPoint)

        def explore():
            '''move towards a random point in collision free space'''

            u_rand, angle = self.RRT.steering(self.gps_x, self.gps_y, self.compass_angle, far_angles)
            return self.move_to_point(u_rand[0], u_rand[1])      
        
        def way_back():

            if(self.RRT.rescueCenterNode is None):
                return explore()
            else:
                if(not self.builtWayBack): #build path back to rescue center
                    node_u = self.RRT.nodes[self.RRT.getNodeIndex(self.currentPoint)]
                    node_v = self.RRT.nodes[self.RRT.rescueCenterNode]
                    self.path = self.RRT.build_path(node_u, node_v)
                    self.builtWayBack = True

                #we move towards path[0] (next point in path)
                if(len(self.path) > 0):
                    if(self.currentPoint == (self.path[0].x, self.path[0].y)):
                        self.path.pop(0)
                    return self.move_to_point(self.path[0].x, self.path[0].y)
                
                return command_straight


        if self.state is self.Activity.SEARCHING_WOUNDED:
            self.builtWayBack = False
            command = explore()
            command["grasper"] = 0

        elif self.state is self.Activity.GRASPING_WOUNDED:
            command = command_semantic
            command["grasper"] = 1

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            command = way_back() 
            command["grasper"] = 1

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            command = command_semantic
            command["grasper"] = 1

        return command
