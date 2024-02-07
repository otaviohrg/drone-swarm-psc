# pylint: disable=import-error
import math
import random
import numpy as np
from typing import Optional
from enum import Enum
from copy import deepcopy
import heapq

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.utils import normalize_angle, circular_mean, sign

from solutions.BezierCurve import BezierCurve
from solutions.RRT import RRT
from solutions.Localization import Localization
from scipy.spatial import distance

class MyDroneEval(DroneAbstract):

    def define_message_for_all(self):
        msg_data = (self.identifier,
                    (self.measured_gps_position(), self.measured_compass_angle()))
        #Maybe we can add state specific information to the message ??
        if self.state is self.Activity.SEARCHING_WOUNDED:
            pass
        elif self.state is self.Activity.GRASPING_WOUNDED:
            pass
        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            pass
        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            pass
        return msg_data


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
        self.historic_size = 100
        self.historic_gps = []
        self.historic_angle = []
        self.historic_commands = []

        self.localization = Localization()

        self.RRT = RRT()
        self.currentPoint = (self.gps_x, self.gps_y)
        self.RRT.addNode(self.currentPoint, None) #add initial node (root) to tree
        self.previousPoint = None

        self.builtWayBack = False
        self.scaling_factor = 55
        self.path = []
        self.path_current_index = 0

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
        '''Return GPS values if available, else use no gps strategy'''
        if(self.gps_is_disabled() and len(self.historic_gps) > 1):
            last_command = self.historic_commands[-1]
            last_position = self.historic_gps[-1]
            last_angle = self.historic_angle[-1]
            predicted_x, predicted_y = self.localization.predict_position(last_position, last_command, last_angle)
            
            #gps_measurement_x = self.measured_gps_position()[0]
            #gps_measurement_y = self.measured_gps_position()[1]
            #print(f"PREDICTION ({predicted_x:.2f}, {predicted_y:.2f})  REAL ({gps_measurement_x:.2f}, {gps_measurement_y:.2f})")

            self.historic_gps.append((predicted_x, predicted_y))
            if len(self.historic_gps) > self.historic_size:
                self.historic_gps.pop(0)

            discrete_gps_x = predicted_x/self.scaling_factor
            discrete_gps_y = predicted_y/self.scaling_factor

            return discrete_gps_x, discrete_gps_y
        else:
            gps_measurement_x = self.measured_gps_position()[0]
            gps_measurement_y = self.measured_gps_position()[1]

            self.historic_gps.append((gps_measurement_x, gps_measurement_y))
            if len(self.historic_gps) > self.historic_size:
                self.historic_gps.pop(0)
    
            return self.measured_gps_position()[0]/self.scaling_factor, self.measured_gps_position()[1]/self.scaling_factor

    def update_gps_values(self):
        self.gps_x, self.gps_y = self.get_gps_values()

    def get_compass_values(self):
        if self.compass_is_disabled():
            last_command = self.historic_commands[-1]
            last_angle = self.historic_angle[-1]
            if last_command["rotation"] > 0:
                return last_angle + 0.2
            elif last_command["rotation"] < 0:
                return last_angle - 0.2
            else:
                return last_angle
        else:
            return self.measured_compass_angle()

    def update_compass_values(self):
        self.compass_angle = self.get_compass_values()
        self.historic_angle.append(self.compass_angle)
        if len(self.historic_angle) > self.historic_size:
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

            min_dist = min(values)
            max_dist = max(values)

            # collision_angles: angles with smallest distances
            collision_angle_indexes = [values.index(i) for i in heapq.nsmallest(10, values)]
            collision_angles = ray_angles[collision_angle_indexes]

            #far_angle_indexes = [values.index(i) for i in heapq.nlargest(20, values)]
            far_angle_indexes = []
            for i in range(size):
                if(values[i] >= 0.7*max_dist): # far_angles: angles with big enough distances
                    far_angle_indexes.append(i)
            far_angles = ray_angles[far_angle_indexes]

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
    
    def process_communication_sensor(self):
        ''' Use drone communication to construct swarm coordination command '''
        found_drone = False
        command_comm = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0}

        if self.communicator:
            received_messages = self.communicator.received_messages
            nearest_drone_coordinate1 = (
                self.measured_gps_position(), self.measured_compass_angle())
            nearest_drone_coordinate2 = deepcopy(nearest_drone_coordinate1)
            (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
            (nearest_position2, nearest_angle2) = nearest_drone_coordinate2

            min_dist1 = 10000
            min_dist2 = 10000
            diff_angle = 0

            # Search the two nearest drones around
            for msg in received_messages:
                message = msg[1]
                coordinate = message[1]
                (other_position, other_angle) = coordinate

                dx = other_position[0] - self.measured_gps_position()[0]
                dy = other_position[1] - self.measured_gps_position()[1]
                distance = math.sqrt(dx ** 2 + dy ** 2)

                # if another drone is near
                if distance < min_dist1:
                    min_dist2 = min_dist1
                    min_dist1 = distance
                    nearest_drone_coordinate2 = nearest_drone_coordinate1
                    nearest_drone_coordinate1 = coordinate
                    found_drone = True
                elif distance < min_dist2 and distance != min_dist1:
                    min_dist2 = distance
                    nearest_drone_coordinate2 = coordinate

            if not found_drone:
                return found_drone, command_comm

            # If we found at least 2 drones
            if found_drone and len(received_messages) >= 2:
                (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
                (nearest_position2, nearest_angle2) = nearest_drone_coordinate2
                diff_angle1 = normalize_angle(
                    nearest_angle1 - self.measured_compass_angle())
                diff_angle2 = normalize_angle(
                    nearest_angle2 - self.measured_compass_angle())
                # The mean of 2 angles can be seen as the angle of a vector, which
                # is the sum of the two unit vectors formed by the 2 angles.
                diff_angle = math.atan2(0.5 * math.sin(diff_angle1) + 0.5 * math.sin(diff_angle2),
                                        0.5 * math.cos(diff_angle1) + 0.5 * math.cos(diff_angle2))

            # If we found only 1 drone
            elif found_drone and len(received_messages) == 1:
                (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
                diff_angle1 = normalize_angle(
                    nearest_angle1 - self.measured_compass_angle())
                diff_angle = diff_angle1

            # if you are far away, you get closer
            # heading < 0: at left
            # heading > 0: at right
            # base.angular_vel_controller : -1:left, 1:right
            # we are trying to align : diff_angle -> 0
            command_comm["rotation"] = sign(diff_angle)

            # Desired distance between drones
            desired_dist = 60

            d1x = nearest_position1[0] - self.measured_gps_position()[0]
            d1y = nearest_position1[1] - self.measured_gps_position()[1]
            distance1 = math.sqrt(d1x ** 2 + d1y ** 2)

            d1 = distance1 - desired_dist
            # We use a logistic function. -1 < intensity1(d1) < 1 and  intensity1(0) = 0
            # intensity1(d1) approaches 1 (resp. -1) as d1 approaches +inf (resp. -inf)
            intensity1 = 2 / (1 + math.exp(-d1 / (desired_dist * 0.5))) - 1

            direction1 = math.atan2(d1y, d1x)
            heading1 = normalize_angle(direction1 - self.measured_compass_angle())

            # The drone will slide in the direction of heading
            longi1 = intensity1 * math.cos(heading1)
            lat1 = intensity1 * math.sin(heading1)

            # If we found only 1 drone
            if found_drone and len(received_messages) == 1:
                command_comm["forward"] = longi1
                command_comm["lateral"] = lat1

            # If we found at least 2 drones
            elif found_drone and len(received_messages) >= 2:
                d2x = nearest_position2[0] - self.measured_gps_position()[0]
                d2y = nearest_position2[1] - self.measured_gps_position()[1]
                distance2 = math.sqrt(d2x ** 2 + d2y ** 2)

                d2 = distance2 - desired_dist
                intensity2 = 2 / (1 + math.exp(-d2 / (desired_dist * 0.5))) - 1

                direction2 = math.atan2(d2y, d2x)
                heading2 = normalize_angle(direction2 - self.measured_compass_angle())

                longi2 = intensity2 * math.cos(heading2)
                lat2 = intensity2 * math.sin(heading2)

                command_comm["forward"] = 0.5 * (longi1 + longi2)
                command_comm["lateral"] = 0.5 * (lat1 + lat2)

        return found_drone, command_comm

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

        collided, collision_angles, far_angles, min_dist, max_dist = self.process_lidar_sensor()

        self.update_gps_values()
        self.update_compass_values()

        self.previousPoint = self.currentPoint
        self.currentPoint = (round(self.gps_x), round(self.gps_y))
        
        if(not self.RRT.alreadyVisited(self.currentPoint)): #add node to Tree and edge to parent
            #print(f"Added to tree {self.currentPoint} parent {self.previousPoint}")
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
                    
                    tree_path, tree_path_points = self.RRT.build_path(node_u, node_v)
                    bezier_curve = BezierCurve(tree_path_points)
                    bezier_path = bezier_curve.generate_curve_points(len(tree_path))
                    #self.path = np.add(np.add(tree_path_points, tree_path_points), bezier_path)/3
                    self.path = bezier_path
                    self.path_current_index = 0
                    bezier_curve.output_curve_image()
                    self.builtWayBack = True

                #we move towards next point in path
                next_point_in_path = (self.path[self.path_current_index][0], self.path[self.path_current_index][1])
                while(distance.euclidean(self.currentPoint, next_point_in_path) < 0.5 and self.path_current_index + 1 < len(self.path)):
                    self.path_current_index += 1
                    next_point_in_path = (self.path[self.path_current_index][0], self.path[self.path_current_index][1])

                #print(f"AT {self.currentPoint} MOVE TO {next_point_in_path} DISTANCE IS {distance.euclidean(self.currentPoint, self.path[self.path_current_index])}")
                return self.move_to_point(next_point_in_path[0], next_point_in_path[1])


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


        found_drone, command_comm = self.process_communication_sensor()
        alpha = 0.5 #empirical alpha value
        alpha_rot = 0.2 if collided else 0.5

        if(found_drone): #use communication info
            command["forward"] = alpha * command_comm["forward"] + (1 - alpha) * command["forward"]
            command["lateral"] = alpha * command_comm["lateral"] + (1 - alpha) * command["lateral"]
            command["rotation"] = alpha_rot * command_comm["forward"] + (1 - alpha_rot) * command["forward"]

        #random command to collect data for regression model
        command["forward"] = random.uniform(0, 1)
        command["lateral"] = random.uniform(0, 1)
        command["rotation"] = random.uniform(-1, 1)
        command["grasper"] = 0
        
        self.historic_commands.append(command)

        #Print to collect data for regression model
        if(len(self.historic_gps) > 2 and len(self.historic_commands) > 1 and len(self.historic_angle) > 1):
            print(f"{self.historic_commands[-1]} {self.historic_angle[-1]} {self.historic_gps[-1][0] - self.historic_gps[-2][0]} {self.historic_gps[-1][1] - self.historic_gps[-2][1]}")

        return command

