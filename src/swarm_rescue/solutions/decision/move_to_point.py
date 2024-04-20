import numpy as np
from scipy.spatial import distance
import math

def move_to_point(destx, desty, compass_angle, gps_x, gps_y):
    '''
    At each step of the movement, we try multiple different rotations and pick the one that minimises distance to target
    '''
    command = {"forward": 0.65,
                "lateral": 0.0,
                "rotation": 0.0,
                "grasper": 0.0}
    candidates = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]

    #rotation = 1 => 0,2 rad per step => takes 32 steps to do the whole circle at rotation = 1
    rotation_discretization_parameter = 64
    rotation_speed = 32/rotation_discretization_parameter

    dest = (destx, desty)
    criteria = []
    for i in candidates:
        direction_angle = compass_angle + 2*i*math.pi/rotation_discretization_parameter
        point_in_direction = (gps_x + math.cos(direction_angle), gps_y + math.sin(direction_angle))
        criteria.append(distance.euclidean(point_in_direction, dest))

    i_chosen = candidates[np.argmin(criteria)]
    command["rotation"] = i_chosen*rotation_speed

    return command