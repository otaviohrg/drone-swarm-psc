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
            collision_angle_indexes = [values.index(i) for i in heapq.nsmallest(15, values)]
            collision_angles = ray_angles[collision_angle_indexes]
            min_dist = min(values)
            # far_angles: angles with biggest distances
            far_angle_indexes = [values.index(i) for i in heapq.nlargest(15, values)]
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

        self.gps_x = self.measured_gps_position()[0]
        self.gps_y = self.measured_gps_position()[1]
        self.compass_angle = self.measured_compass_angle()

        self.previousPoint = self.currentPoint
        self.currentPoint = (round(self.gps_x), round(self.gps_y))
        
        if(not self.RRT.alreadyVisited(self.currentPoint)): #add node to Tree and edge to parent
            previousNode = self.RRT.getNodeIndex(self.previousPoint)
            self.RRT.addNode(self.currentPoint, self.RRT.nodes[previousNode])
            self.RRT.addEdge(self.previousPoint, self.currentPoint)
        
        if(self.can_see_rescue_center()):
            self.RRT.rescueCenterNode = self.RRT.getNodeIndex(self.currentPoint)

        def explore():
            '''move towards a random point in collision free space'''

            self.counterStraight += 1

            if collided and (not self._is_turning()) and (self.counterStraight > self.distStopStraight):
                #calculate next step
                u_rand, angle = self.RRT.steering(self.gps_x, self.gps_y, self.compass_angle, far_angles)
                self.angleStopTurning = angle
                if(self.angleStopTurning < 0):
                    self.isTurningLeft = True
                    self.isTurningRight = False
                else:
                    self.isTurningLeft = False
                    self.isTurningRight = True
                    self.counterStraight = 0
                    self.distStopStraight = round(distance.euclidean((self.gps_x, self.gps_y), u_rand))

            #continue executing step
            measured_angle = 0
            if self.measured_compass_angle() is not None:
                measured_angle = self.measured_compass_angle()

            diff_angle = normalize_angle(self.angleStopTurning - measured_angle)
            if self._is_turning() and abs(diff_angle) < 0.2:
                self.isTurningLeft = False
                self.isTurningRight = False

            if self._is_turning():
                return command_turn_left if self.isTurningLeft else command_turn_right
            else:
                return command_straight
        
        def way_back():

            if(self.RRT.rescueCenterNode is None):
                return explore()
            else:
                if(not self.builtWayBack): #build path back to rescue center
                    node_u = self.RRT.nodes[self.RRT.getNodeIndex(self.currentPoint)]
                    node_v = self.RRT.nodes[self.RRT.rescueCenterNode]
                    path = self.RRT.build_path(node_u, node_v)

                #we move towards path[0] (next point in path)
                if(self.currentPoint == (path[0].x, path[0].y)):
                    path.pop(0)
                return self.move_to_point(path[0].x, path[0].y)


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
