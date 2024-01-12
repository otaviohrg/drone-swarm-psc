import random
import numpy as np

class Config1():
    def __init__(self):
        self.size_area = (800, 500)
        self.walls = [((13.0, 250.0),(13.0, -93.0)),((-227.0, 89.0),(-227.0, -250.0))]
        self.rescue=(295, 205)
        self.wounded_pos = [(-310, -180)]
        self.gps=((400, 500),(-190, 0))
        self.drones_pos = [(295, 118)]


class Corridor():
    def __init__(self):
        self.x,self.y=1000,500
        self.size_area = (self.x, self.y)
        self.walls = []
        #n=random.randrange(1,4)
        #self.border(n,self.y//6)
        #n=random.randrange(1,4)
        #self.border(n,-self.y//6)

        self.rescue=(self.x//2-40, 0)
        self.wounded_pos = []
        n=random.randrange(3,7)
        
        #self.walls = [((0,-250),(0,250))]
        self.walls,self.wound=organize((self.x,self.y))
        #percolation(self.walls,(0,self.y//4))
        #percolation(self.walls,(0,-self.y//4))
        print(self.walls)
        self.gps=((400, 500),(-190, 0))
        self.drones_pos = [(self.x//2-100, 0)]

def place_corridor(center,size,nbmpers,nbmurs,angle):
    x,y=size
    size=(x,y//2)
    if (angle// 90)%2==1:
        size=(y,x//2)
    else:
        size=(x,y//2)

    walls1,wounded1=corridor(size,nbmpers,nbmurs)
    print(walls1)
    walls1=rotate(walls1,(0,0),angle)
    print(walls1)
    walls2,wounded2=corridor(size,nbmpers,nbmurs)
    walls2=rotate(walls2,(0,0),angle+180)
    fence=[((x//2,y//2),(-x//2,y//2)),((-x//2,y//2),(-x//2,-y//2)),((-x//2,-y//2),(x//2,-y//2)),((x//2,-y//2),(x//2,y//2))]
    return translateW(walls1+walls2+fence,center),(wounded1+wounded2)
    



def corridor(size,nbmpers,nbmurs):
    (x,y)=size
    c=sorted([random.randrange(-x//2+80,x//2-80) for _ in range(nbmurs)])
    walls=[]
    for i in range(nbmurs-1):
        if c[nbmurs-i-1]-c[nbmurs-i-2]<82:
            del c[nbmurs-i-1]
    n=len(c)
    def f(i):
        return i//3
    k=-x//2
    for i in range(n+1):
        if i==n:
            walls.append(((k, f(y)),(x//2, f(y))))
            continue
        walls.append(((k, f(y)),(c[i]-40, f(y))))
        k=c[i]+40
    for i in range(n-1):
        p=random.randrange(c[i]+40,c[i+1]-40)
        walls.append(((p, f(y)),(p, y)))

    wounded=[]
    for i in range(nbmpers):
        wounded.append((random.randrange(-x//2+10,x//2-10), random.randrange(-y//2+10,y//2-10)))
    return walls,wounded

def rotate(walls,center,angle):
    cos,sin=0,0
    if (angle// 90)%4==1:
        sin=1
    elif angle// 90%4==2:
        cos=-1
    elif angle// 90%4==3:
        sin=-1
    elif angle// 90%4==0:
        cos=1
    new_walls=[]
    for w in walls:
        a=[]
        for (x,y) in w:
            x,y=x-center[0],y-center[1]
            x,y=(x*cos-y*sin),(x*sin+y*cos)
            x,y=x+center[0],y+center[1]
            a.append((x+center[0],y+center[1]))
        new_walls.append((a[0],a[1]))
    return new_walls

def translateW(walls,delta):
    new_walls=[]
    for w in walls:
        a=[]
        for (x,y) in w:
            a.append((x+delta[0],y+delta[1]))
        new_walls.append((a[0],a[1]))
    return new_walls
def organize(size,n=random.randrange(1)):
    mapX=size[0]
    mapY=size[1]

    x=mapX//2
    y=mapY//2
    rwalls,rwound=[],[]
    walls,wound=place_corridor((-x//2,y//2),(x,y),10,5,0)
    rwalls+=walls;rwound+=wound
    walls,wound=place_corridor((-x//2,-y//2),(x,y),10,5,0)
    rwalls+=walls;rwound+=wound
    walls,wound=place_corridor((x//2,0),(x,y*2),10,5,90)
    rwalls+=walls;rwound+=wound
    percolation(rwalls,(0,y//2))
    percolation(rwalls,(0,-y//2))
    return rwalls,rwound

def percolation(walls,point):
    scalaire=lambda x,y:x[0]*y[0]+x[1]*y[1]
    x,y=point
    broken=[]
    for i,w in enumerate(walls):
        (a,b),(c,d)=w
        xw,yw=x-a,y-b
        c,d=c-a,d-b
        print("c,d :", c, d)
        print("point : ",xw,yw)
        wallsize=(scalaire((c,d),(c,d)))**0.5
        ux,uy=c/wallsize,d/wallsize
        print("U : ",ux,uy)
        prj=scalaire((xw,yw),(ux,uy))
        projx,projy=prj*ux,prj*uy
        #print("\n projx , projy",projx,projy," c,d ",c,d)
        if abs(projx)>abs(c) or abs(projy)>abs(d):
            #print(" ***** *** far away \n\n\n ***")
            continue
        print("proj : ",projx,projy)
        dist=(xw-projx)**2+(yw-projy)**2
        print("dist :", dist)
        if dist<1600:
            h=(40**2-dist)**0.5
            breakpt1=(int(projx-h*ux),int(projy-h*uy))
            breakpt2=(int(projx+h*ux),int(projy+h*uy))
            print("breakpt1 : ",breakpt1)
            if scalaire(breakpt1,(ux,uy))>0:
                print("In Walls : ",((a,b),(a+breakpt1[0],b+breakpt1[1])), "replaces ",walls[i])
                walls[i]=((a,b),(a+breakpt1[0],b+breakpt1[1]))
                print("breakpt1  other ref: ",(a+breakpt1[0],b+breakpt1[1]))
                if scalaire((breakpt2[0]-c,breakpt2[1]-d),(ux,uy))<0:
                    print("\nlook\n break point,u, then dot",breakpt2,(ux,uy),scalaire(breakpt2,(ux,uy)),"\n")
                    print("Added to broken : ",((c+a,d+b),(a+breakpt2[0],b+breakpt2[1])))
                    print( "a,b,c,d :",a,b,c,d)
                    broken.append(((c+a,d+b),(a+breakpt2[0],b+breakpt2[1])))
            else:
                walls[i]=((c+a,d+b),(a+breakpt2[0],b+breakpt2[1]))
    for w in broken:
        walls.append(w)
    print(walls)




