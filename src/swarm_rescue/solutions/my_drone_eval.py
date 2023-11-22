from solutions.my_drone_random import MyDroneRandom
from typing import Optional
import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle

class MyDroneEval(MyDroneRandom):
    
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super(MyDroneRandom,self).__init__( identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)
        
        self.straight = {"forward": 1.0,
                            "lateral": 0.0,
                            "rotation": 0.0,
                            "grasper": 0}
        self.right = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 1.0,
                        "grasper": 0}
        self.rightTrans = {"forward": 0.0,
                    "lateral": 1,
                    "rotation": 0,
                    "grasper": 0}
        
        self.left = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 1.0,
                        "grasper": 0}
        self.strategy = "rw"
        self.rightWall=0
    def findWall(self):
        dist = min(self.lidar_values())
        if dist < 250:
            self.rightWall=1
            print("Seting up")
            return 
        return self.straight
    def RWSetup(self):
        dist = min(self.lidar_values())
        theta=np.argmin(self.lidar_values())
        #print(theta)
        if 48>theta>42:
            if 125<self.lidar_values()[45]:
                return self.rightTrans
            else:
                self.rightWall=2
                print("Following")
                return
        return self.right
    def RWFollow(self):
        dist = self.lidar_values()[90]
        if dist<100:
            self.rightWall=1
            print("Seting up")
        track = self.straight.copy()
        if 125>self.lidar_values()[45]>75:
            track["lateral"]=-(self.lidar_values()[45]-75)/50
        if 125<self.lidar_values()[45]:
            self.rightWall=3
            print("Turning")
            return
        if self.lidar_values()[45]<50:
            track["lateral"]=(50-self.lidar_values()[45])/50
        return track
    def RWTurn(self):
        if min(self.lidar_values()[45:90])<35:
            print("End of turn")
            self.rightWall=1
            print("Seting up")
            return
        mvt=self.rightTrans.copy()
        mvt["lateral"]=-0.5
        mvt["rotation"]=-0.25
        return mvt
    def RW(self):
        if self.rightWall==0:
            return self.findWall()
        if self.rightWall==1:
            return self.RWSetup()
        if self.rightWall==2:
            return self.RWFollow()
        if self.rightWall==3:
            return self.RWTurn()
        return
        
    def process_lidar_sensor(self):
        """
        Returns True if the drone collided an obstacle
        """
        if self.lidar_values() is None:
            return False
        collided = False
        dist = self.lidar_values()[90]

        if dist < 100:
            collided = True

        return collided

    def control(self):

        
        comm=self.right
        if self.strategy=="rw":
            comm=self.RW()
        return comm