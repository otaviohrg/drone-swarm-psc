#Custom environment using gym and stable-baselines3
#The environment represents a drone or a set of drones exploring a 2D map. 
#The drones can move in all the space:
#State: - drone_state (int): the state of the drone
#       - map (np.array): the map of the environment
#       - lidar_sensor (np.array): the lidar sensor of the drone for walls detection, can have noise
#           This sensor has some characteristics:
#           - fov (field of view): 360 degrees
#           - max range (maximum range of the sensor): 300 pixels
#           - resolution, number of rays evenly spaced across the field of view: 181
#       - semantic_sensor (np.array): the semantic sensor of the drone for wounded persons, rescue center, other drones detection
#          This sensor has some characteristics:
#           - fov (field of view): 360 degrees
#           - max range (maximum range of the sensor): 200 pixels
#           - resolution, number of rays evenly spaced across the field of view: 35
#           - Each ray can return the following values:
#               - data.distance(float): distance of the nearest object detected
#               - data.angle(float): angle of the ray in radians   
#               - data.entity_type(TypeEntity): type of the detected object: TypeEntity.WOUNDED_PERSON, TypeEntity.RESCUE_CENTER, TypeEntity.DRONE , TypeEntity.WALL, TypeEntity.OTHER
#               - data.grasped(boolean): is the object grasped by a drone or an agent ? 
#           WARNING: the semantic sensor must not detect walls, just the lidar can detect them
#       - gps_sensor (np.array[2]): the gps sensor of the drone, not always available in certain zones (no-gps zones), can have noise
#       - compass_sensor (float): the compass sensor of the drone, can have noise
#       - odometer_sensor (np.array[3]): the odometer sensor of the drone relative to previous position, can have noise:
#           - dist_travelled (float): the distance travelled by the drone
#           - alpha (float): the angle between the drone and the x axis
#           - theta (float): the angle between the drone and the y axis
#       - communication_system (np.array): the communication system of the drone to communicate with other drones
#       - x (float): the x coordinate of the drone
#       - y (float): the y coordinate of the drone
#Actions: - forward_controller(float between -1 and 1): move the drone forward, This is a force apply to your drone in the longitudinal way
#         - lateral_controller(float between -1 and 1): move the drone laterally, This is a force apply to your drone in the lateral way
#         - angular_velocity_controller(float between -1 and 1): move the drone angularly, This is the speed of rotation.
#         - grasper(boolean): To grasp a wounded person
#The goal is to explore the map and find the wounded persons in some positions and bring them to the rescue center.
#The environment is a 2D grid with the following elements:
#    - 0: empty cell
#    - 1: wall
#    - 2: rescue center
#    - 3: wounded person
#    - 4: drone(s)
#    - 5: Kill zone
#    - 6: No-gps zone
#    - 7: No-communication zone

#The drone can be in either 4 states:
# 1 - searching_wounded: the drone is searching for wounded persons
# 2 - grasping_wounded: the drone is grasping a wounded person
# 3 - searching_rescue_center: the drone is bringing a wounded person to the rescue center
# 4 - dropping_wounded: the drone is dropping a wounded person to the rescue center
# Then it returns to state 1

#For a start, we will consider a single drone with a 2d map with just one wounded person, the rescue center and some walls.
#No communication, no kill zone, no no-gps zone.
#The reward is:
#    - -1 for each move
#    - +100 for each wounded person brought to the rescue zone
#    - -10 for each drone hitting a wall


import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class DroneEnv(gym.Env):
    '''Custom Environment that follows gym interface'''
    metadata = {'render.modes': ['human']}
    # Define constants for clearer code
    FORWARD = 0
    LATERAL = 1
    ROTATION = 2

    SEM_RESOLUTION = 35
    LIDAR_RESOLUTION = 181
    MAX_RANGE_SEMANTIC = 200
    MAX_RANGE_LIDAR = 300
    # Define the action space and the observation space
    # They must be gym.spaces objects
    def __init__(self, map_shape = (10,10)):
        super(DroneEnv, self).__init__()
        # Actions: forward, lateral, angular velocity and grasper which is a boolean
        self.action_space = spaces.Dict({
            "forward_controller": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "lateral_controller": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "angular_velocity_controller": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "grasper": spaces.Discrete(2, shape=(1,))
        })
        # Observations: drone state, map, lidar sensor, semantic sensor, gps sensor, compass sensor, odometer sensor(dist_travelled, alpha, theta), x, y
        #drone state: Enum of { 1, 2, 3, 4}
        
        self.observation_space = spaces.Dict({
            "drone_state": spaces.Discrete(4),
            "map": spaces.Dict({
                "shape": spaces.Box(low=[0,0], high=[map_shape[0],map_shape[1]], shape=(2,), dtype=np.float32),
                "rescue_center": spaces.Box(low=[-map_shape[0]/2,-map_shape[1]/2,0,0], high=[map_shape[0]/2,map_shape[1]/2,map_shape[0],map_shape[1]], shape=(4,), dtype=np.float32),
                "walls_pos":spaces.Dict({}),
                "wounded_persons":spaces.Dict({}),
                "box_pos":spaces.Dict({}),
                "drone_pos":spaces.Dict({})
            }),
            "semantic_sensor":spaces.Dict({
                #"fov": spaces.Box(low=-math.PI, high=math.PI, shape=(1,), dtype=np.float32),
                #"max_range": spaces.Box(low=0, high=200, shape=(1,), dtype=np.float32),
                #"resolution": spaces.Box(low=0, high=35, shape=(1,), dtype=np.int),
            
                "distance": spaces.Box(low=0, high=200, shape=(35,), dtype=np.float32),
                "angle": spaces.Box(low=-math.PI, high=math.PI, shape=(35,), dtype=np.float32),
                "entity_type": spaces.Discrete(5, shape=(35,)),
                "grasped": spaces.Discrete(2, shape=(35,))
                
            }),
            "lidar_sensor": spaces.Box(low=0, high=300, shape=(181,), dtype=np.float32),
            "compass_sensor": spaces.Box(low=-math.PI, high=math.PI, shape=(1,), dtype=np.float32),
            "gps_sensor": spaces.Box(low=[-map_shape[0]/2, -map_shape[1]/2], high=[map_shape[0]/2,map_shape[1]/2], shape=(2,), dtype=np.float32),
            "compass_sensor": spaces.Box(low=0, high=1, shape=(2,), dtype=np.int),
            "odometer_sensor": spaces.Box(low=[0,-math.PI, -math.PI], high=[2*np.sqrt(map_shape[0]**2 + map_shape[1]**2), math.PI, math.PI], shape=(3,), dtype=np.float32),
            "x": spaces.Box(-map_shape[0]/2, map_shape[0]/2, shape=(1,), dtype=np.float32),
            "y": spaces.Box(-map_shape[1]/2, map_shape[1]/2, shape=(1,), dtype=np.float32)
        })

    def initialiaze_map(self, map_shape, walls, rescue_center, wounded_persons, box_pos, drone_pos):
        '''Initializes the map of the environment with the following elements:
            - map_shape(np.array[2]): the size of the map: [width, height]
            - rescue_center(np.array[4]: position of the rescue center: [x_upleft, y_upleft, width, height])
            - walls(np.array[4]): list of walls positions: each element is a [x_start, y_start, x_end, y_end]
            - wounded_persons(np.array[n_wounded_persons, 2]): list of wounded persons positions: each element is a [x, y]
            - box_pos(np.array[n_boxes,4]): list of boxes positions: each element is a [x_upleft, y_upleft, width, height]
            - drone_pos(np.array[n_drones,2]): list of drones positions: each element is a [x, y]
            returns map: a dictionary
        '''

        map={}
        map["shape"] = map_shape
        map["rescue_center"] = rescue_center
        map["walls_pos"] = walls
        map["wounded_persons"] = wounded_persons
        map["box_pos"] = box_pos
        map["drone_pos"] = drone_pos
        return map
    
    def reset(self, map_shape, walls, rescue_center, wounded_persons, box_pos, drone_pos):
        """Initializes the environment and returns the initial state"""
        # Initialize the state with the drone in the searching_wounded state
        self.state = 1
        # We must return a valid observation
        
        self.observation_space["map"]= self.initialiaze_map(map_shape, walls, rescue_center, wounded_persons, box_pos, drone_pos)
        self.observation_space["drone_state"] = 1 #searching_wounded
        self.observation_space["semantic_sensor"].sample()
        self.observation_space["lidar_sensor"].sample()
        self.observation_space["compass_sensor"].sample()
        self.observation_space["gps_sensor"].sample()
        self.observation_space["odometer_sensor"].sample()
        self.observation_space["x"].sample()
        self.observation_space["y"].sample()
        return self.observation_space
    
    def step(self, action):
        """Update the state and return the next observation"""
        # Execute one time step within the environment
        # Update the state
        self.state += 1
        # We must return a valid observation
        return self.observation_space.sample(), 0, False, {}
    

