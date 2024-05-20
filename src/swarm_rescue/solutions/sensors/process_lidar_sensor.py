import heapq
import random
import numpy as np

def process_lidar_sensor(drone):
        """
        Returns flag indicating collision and lists of "good" and "bad" angles
        """

        lidar_sensor = drone.lidar()
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
            collision_angle_indexes = [values.index(i) for i in heapq.nsmallest(15, values)]
            collision_angles = ray_angles[collision_angle_indexes]

            # far_angles: angles with largest distances
            far_angle_indexes = [values.index(i) for i in heapq.nlargest(15, values)]
            far_angles = ray_angles[far_angle_indexes]

        if drone.lidar_values() is None:
            return False, 0

        collided = False
        if min_dist < 110:
            collided = True

        return collided, collision_angles, far_angles, min_dist, max_dist
