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

from scipy.stats import norm, poisson, levy
from scipy.spatial import distance

#to compute best and worst angles in sensor output
import heapq

#necessary to run: pip install hilbertcurve (added to requirements.txt)
from hilbertcurve.hilbertcurve import HilbertCurve

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
        self.gps_x = self.measured_gps_position()[0]/60 + 8
        self.gps_y = self.measured_gps_position()[1]/60 + 8
        self.compass_angle = self.measured_compass_angle()
        self.historic_gps = []
        
        #HILBERT CURVE INIT
        # Iteration of Hilbert's curve
        self.hilbert_iter = 4
        # Dimension of Hilber's curve
        self.hilbert_dim = 2
        # Number of points in Hilbert's curve
        self.hilbert_size = 2 ** (self.hilbert_iter * self.hilbert_dim)
        self.hilbert_sideSize = 2 ** (self.hilbert_iter * self.hilbert_dim/2)
        # Area covered by the Hibert's curve
        self.hilbert_xmin = 0
        self.hilbert_ymin = 0
        self.hilbert_xmax = 10 * self.hilbert_sideSize
        self.hilbert_ymax = 10 * self.hilbert_sideSize
        # Grid size
        self.hilbert_xgrid = 1
        self.hilbert_ygrid = 1
        # Bounding grid
        self.hilbert_xminGrid = self.hilbert_xmin - self.hilbert_xgrid/2
        self.hilbert_yminGrid = self.hilbert_ymin - self.hilbert_ygrid/2
        self.hilbert_xmaxGrid = self.hilbert_xmax + self.hilbert_xgrid/2
        self.hilbert_ymaxGrid = self.hilbert_ymax - self.hilbert_ygrid/2
        # Creating Hilbert's curve
        self.hilbert_curve = HilbertCurve(self.hilbert_iter, self.hilbert_dim)
        self.hilbert_distances = list(range(self.hilbert_size))
        self.hilbert_points = self.hilbert_curve.points_from_distances(self.hilbert_distances)
        self.startWayBack = False
        self.hilbert_visited_flags = [0 for i in range(self.hilbert_size)]
        # Enumerating points on Hilbert curve
        self.hilbert_xList = np.array([self.hilbert_points[i][0] for i in range(self.hilbert_size)])
        self.hilbert_yList = np.array([self.hilbert_points[i][1] for i in range(self.hilbert_size)])

        #parameters for random steps
        self.counterStraight = 0
        self.angleStopTurning = 0
        self.counterStopStraight = 0
        self.isTurningLeft = False
        self.isTurningRight = False


    def get_point_index(self, xl, yl):
        """Calculates the index of closest point on the hilbert's curve"""
        given_point = (xl, yl)
        distances = []
        for i in range(self.hilbert_size):
            distance_i = distance.euclidean(given_point, self.hilbert_points[i])
            distances.append(distance_i)
        return np.argmin(distances)

    def get_adjacent_nodes(self, i):
        """Outputs the adjacent nodes of a given node"""
        x_i = self.hilbert_xList[i]
        y_i = self.hilbert_yList[i]

        adjacent_points = {
            "p1": [x_i + self.hilbert_xgrid, y_i],
            "p2": [x_i, y_i + self.hilbert_ygrid],
            "p3": [x_i - self.hilbert_xgrid, y_i],
            "p4": [x_i, y_i - self.hilbert_ygrid],
        }

        adjacent_nodes  = []
        for point in adjacent_points:
            if adjacent_points[point][0] in self.hilbert_xList and adjacent_points[point][1] in self.hilbert_yList:
                adjacent_nodes.append(self.get_point_index(adjacent_points[point][0], adjacent_points[point][1]))
        return adjacent_nodes
    
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
            collision_angle_indexes = [values.index(i) for i in heapq.nsmallest(10, values)]
            collision_angles = ray_angles[collision_angle_indexes]
            min_dist = min(values)
            # far_angles: angles with biggest distances
            far_angle_indexes = [values.index(i) for i in heapq.nlargest(10, values)]
            far_angles = ray_angles[far_angle_indexes]
            max_dist = max(values)

        if self.lidar_values() is None:
            return False, 0

        collided = False
        dist = min(self.lidar_values())

        if dist < 100:
            collided = True

        return collided, collision_angles, far_angles, min_dist, max_dist

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
        collided, collision_angles, far_angles, min_dist, max_dist = self.process_lidar_sensor()

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
        # Searching, but when a rescue center or wounded person is detected, we use a special command
        ##########

        def explore():

            self.gps_x = self.measured_gps_position()[0]/60 + 8
            self.gps_y = self.measured_gps_position()[1]/60 + 8
            self.historic_gps.append((self.gps_x, self.gps_y))
            if(len(self.historic_gps) > 100):
                self.historic_gps.pop(0)
            self.compass_angle = self.measured_compass_angle()

            #we essentially do DFS, vertices are the points on the Hilbert curve
            closest_hilbert_point_index = self.get_point_index(self.gps_x, self.gps_y)

            adjacent_node_indexes  = self.get_adjacent_nodes(closest_hilbert_point_index)
            for candidate_node in adjacent_node_indexes:
                if(self.hilbert_visited_flags[candidate_node] < 3):

                    #reset random step parameters
                    self.isTurningLeft = False
                    self.isTurningRight = False
                    self.counterStraight = 1
                    self.counterStopStraigt = 0

                    self.hilbert_visited_flags[candidate_node] += 1
                    command = self.move_to_point(self.hilbert_xList[candidate_node], self.hilbert_yList[candidate_node])
                    return command

            #if we got here => candidate nodes are all visited already => random walk to find other vertices
            
            command_straight = {"forward": 1.0,
                            "rotation": 0.0}
            command_turn_left = {"forward": 0.0,
                            "rotation": 1.0}
            command_turn_right = {"forward": 0.0,
                            "rotation": -1.0}

            self.counterStraight += 1

            if not self._is_turning() and self.counterStraight > self.counterStopStraight:
                #pick far_angle with best heuristic
                weights = []
                for angle in far_angles:
                    direction_angle = self.compass_angle + angle
                    point_in_direction = (self.gps_x + 20*math.cos(direction_angle), self.gps_y + 20*math.sin(direction_angle))
                    rotation_heuristic = 1e9
                    for visited in self.historic_gps:
                        rotation_heuristic = min(rotation_heuristic, distance.euclidean(point_in_direction, visited))
                    weights.append(rotation_heuristic)
                self.angleStopTurning = self.compass_angle + far_angles[np.argmax(weights)]

                diff_angle = normalize_angle(self.angleStopTurning - self.measured_compass_angle())
                if diff_angle > 0:
                    self.isTurningLeft = True
                else:
                    self.isTurningRight = True

            diff_angle = normalize_angle(self.angleStopTurning - self.measured_compass_angle())
            if self._is_turning() and abs(diff_angle) < 0.2:
                self.isTurningLeft = False
                self.isTurningRight = False
                self.counterStraight = 0
                self.counterStopStraight = 20

            if self.isTurningLeft:
                return command_turn_left
            elif self.isTurningRight:
                return command_turn_right
            else:
                return command_straight

        if self.state is self.Activity.SEARCHING_WOUNDED:
            command = explore()
            command["grasper"] = 0

        elif self.state is self.Activity.GRASPING_WOUNDED:
            command = command_semantic
            command["grasper"] = 1

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            if(self.startWayBack is False):#reset Hilbert curve and explore again
                self.startWayBack = True
                self.hilbert_visited_flags = [0 for i in range(self.hilbert_size)]
            command = explore() 
            command["grasper"] = 1

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            command = command_semantic
            command["grasper"] = 1

        return command
