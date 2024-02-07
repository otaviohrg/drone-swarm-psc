import random
import math
from typing import List, Type

from spg.playground import Playground
from spg.utils.definitions import CollisionTypes

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.rescue_center import RescueCenter, wounded_rescue_center_collision
from spg_overlay.entities.sensor_disablers import ZoneType, NoGpsZone, srdisabler_disables_device
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.reporting.evaluation import ZonesConfig
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.entities.normal_wall import NormalWall,NormalBox

from .config1 import Config1 as Cf, Corridor as Cr


def createMap():
    pass


class GenMap(MapAbstract):

    def __init__(self, zones_config: ZonesConfig = ()):
        super().__init__(zones_config)
        self._time_step_limit = 2000
        self._real_time_limit = 120
        config=Cr()
        self.build(config)
    def build(self,config):
        self._size_area = config.size_area

        self._rescue_center = RescueCenter(size=(80, 130))
        self._rescue_center_pos = (config.rescue, 0)

        print("GPS :",config.gps)
        self._no_gps_zone = NoGpsZone(size=config.gps[0])
        self._no_gps_zone_pos = (config.gps[1], 0)

        self._wounded_persons_pos = config.wounded_pos
        self._number_wounded_persons = len(self._wounded_persons_pos)
        self._wounded_persons: List[WoundedPerson] = []


        self._number_drones = len(config.drones_pos)

        self._drones_pos = []
        for dronepos in config.drones_pos:
            orient = random.uniform(-math.pi, math.pi)
            self._drones_pos.append((dronepos,orient))
        self._drones: List[DroneAbstract] = []
        self.walls=config.walls
        self.boxes = config.boxes
        print(self.walls)





    def construct_playground(self, drone_type: Type[DroneAbstract]) -> Playground:
        playground = ClosedPlayground(size=self._size_area)

        # RESCUE CENTER
        playground.add_interaction(CollisionTypes.GEM,
                                   CollisionTypes.ACTIVABLE_BY_GEM,
                                   wounded_rescue_center_collision)
        playground.add(self._rescue_center, self._rescue_center_pos)
        for pos in self.walls:
            wall = NormalWall(pos_start=pos[0],
                      pos_end=pos[1])
            playground.add(wall, wall.wall_coordinates)
            print(wall.wall_coordinates)
        for pos in self.boxes:
            print("box of corner :",pos[0],"and size : ",pos[1])
            box= NormalBox(up_left_point=pos[0],
                    width=pos[1][0], height=pos[1][1])
            playground.add(box, box.wall_coordinates)
            
        self._explored_map.initialize_walls(playground)

        # DISABLER ZONES
        playground.add_interaction(CollisionTypes.DISABLER,
                                   CollisionTypes.DEVICE,
                                   srdisabler_disables_device)

        if ZoneType.NO_GPS_ZONE in self._zones_config:
            playground.add(self._no_gps_zone, self._no_gps_zone_pos)

        # POSITIONS OF THE WOUNDED PERSONS
        for i in range(self._number_wounded_persons):
            wounded_person = WoundedPerson(rescue_center=self._rescue_center)
            self._wounded_persons.append(wounded_person)
            pos = (self._wounded_persons_pos[i], 0)
            playground.add(wounded_person, pos)

        # POSITIONS OF THE DRONES
        misc_data = MiscData(size_area=self._size_area,
                             number_drones=self._number_drones)
        for i in range(self._number_drones):
            drone = drone_type(identifier=i, misc_data=misc_data)
            self._drones.append(drone)
            playground.add(drone, self._drones_pos[i])

        return playground
