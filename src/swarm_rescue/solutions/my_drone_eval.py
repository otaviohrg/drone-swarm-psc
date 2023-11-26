import math
import random
import numpy as np
from typing import Optional
from enum import Enum

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.utils import normalize_angle, circular_mean

from scipy.stats import norm, poisson, levy
from scipy.spatial import distance

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
        self.planningPhase = True
        self.turnCounter = 0
        self.turnArg = 0
        self.walkCounter = 0
        self.compass_angle = 0
        self.gps_x = 0
        self.gps_y = 0
        self.visitedPoints = []

    def _is_turning(self):
            return self.isTurningLeft or self.isTurningRight

    def process_lidar_sensor(self):
        """
        Returns True if the drone collided an obstacle
        """

        lidar_sensor = self.lidar()
        values = lidar_sensor.get_sensor_values()
        ray_angles = lidar_sensor.ray_angles
        size = lidar_sensor.resolution

        collision_angle = 0
        far_angle = 0
        min_dist = 0
        max_dist = 0
        if size != 0:
            # collision_angle: angle with smallest distance
            collision_angle = ray_angles[np.argmin(values)]
            min_dist = min(values)
            # far_angle: angle with biggest distance
            far_angle = ray_angles[np.argmax(values)]
            max_dist = max(values)

        if self.lidar_values() is None:
            return False, 0

        collided = False
        dist = min(self.lidar_values())

        if dist < 40:
            collided = True

        return collided, collision_angle, far_angle, min_dist, max_dist

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

        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}

        found_wounded, found_rescue_center, command_semantic = self.process_semantic_sensor()
        collided, collision_angle, far_angle, min_dist, max_dist = self.process_lidar_sensor()

        #############
        # TRANSITIONS OF THE STATE MACHINE
        #############

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
        # Searching randomly, but when a rescue center or wounded person is detected, we use a special command
        ##########

        def explore():

            command = {"forward": 0.0,
                    "lateral": 0.0,
                    "rotation": 0.0,
                    "grasper": 0.0}

            '''strategy: Hilbert curves!
                the drone path is a sequence of steps
                in each step, the drone follows a Hilbert curve
                we use a randomized rotating constant sized frame to choose exploration direction
            '''
            if(self.gps_is_disabled()): #random strategy
                if(self.planningPhase): #plan next step
                    #small rotations when cornered, big rotation in open areas
                    rotation_parameter = norm.rvs()*((max_dist - min_dist)/(max_dist + 0.001))
                    self.turnArg = 0.5 if rotation_parameter > 0 else -0.5
                    self.turnCounter = min(30, poisson.rvs(math.floor(30*rotation_parameter**2), size =1))
                    #small walk when cornered, big walk in open areas
                    walk_parameter = norm.rvs()*((max_dist - min_dist)/(max_dist + 0.001))
                    self.walkCounter = min(30, 3*poisson.rvs(math.floor(10*walk_parameter**2), size = 1))
                    self.planningPhase = False

                self.turnCounter -= 1
                if(self.turnCounter >= 0):#rotate
                    command["rotation"] = self.turnArg
                    return command
                
                if(collided): #avoid obstacle
                    if(far_angle > collision_angle):#add left rotation
                        command["rotation"] = 0.5
                    else: #add right rotation
                        command["rotation"] = -0.5

                self.walkCounter -= 1
                if(self.walkCounter >= 0):#rotate
                    command["forward"] = 1.0

                if(self.walkCounter <= 0):
                    self.planningPhase = True

                return command
            
            else: #we can use gps and compass data

                self.compass_angle = self.measured_compass_angle()
                self.gps_x = self.measured_gps_position()[0]
                self.gps_y = self.measured_gps_position()[1]

                self.visitedPoints.append( (self.gps_x, self.gps_y) )

                if(self.planningPhase): #plan next step
                    #compute heuristic for each rotation
                    weights = []
                    for i in range(31):
                        direction_angle = self.compass_angle + i*math.pi/15
                        point_in_direction = (self.gps_x + 10*math.cos(direction_angle), self.gps_y + 10*math.sin(direction_angle))
                        rotation_heuristic = 0
                        for visited in self.visitedPoints:
                            rotation_heuristic += distance.euclidean(point_in_direction, visited)
                        weights.append(rotation_heuristic)
                    randomChoice = random.choices(np.arange(31), weights, k=1)
                    self.turnCounter = randomChoice[0]
                    self.turnArg = 0.5 if (random.uniform(-1, 1) > 0 ) else -0.5
                    #small walk when cornered, big walk in open areas
                    walk_parameter = norm.rvs()*((max_dist - min_dist)/(max_dist + 0.001))
                    self.walkCounter = min(30, 3*poisson.rvs(math.floor(10*walk_parameter**2), size = 1))
                    self.planningPhase = False

                self.turnCounter -= 1
                if(self.turnCounter >= 0):#rotate
                    command["rotation"] = self.turnArg
                    return command
                
                if(collided): #avoid obstacle
                    if(far_angle > collision_angle):#add left rotation
                        command["rotation"] = 0.5
                    else: #add right rotation
                        command["rotation"] = -0.5

                self.walkCounter -= 1
                if(self.walkCounter >= 0):#rotate
                    command["forward"] = 1.0

                if(self.walkCounter <= 0):
                    self.planningPhase = True

                return command
                

        if self.state is self.Activity.SEARCHING_WOUNDED:
            command = explore()
            command["grasper"] = 0

        elif self.state is self.Activity.GRASPING_WOUNDED:
            command = command_semantic
            command["grasper"] = 1

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            command = explore()
            command["grasper"] = 1

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            command = command_semantic
            command["grasper"] = 1

        return command
