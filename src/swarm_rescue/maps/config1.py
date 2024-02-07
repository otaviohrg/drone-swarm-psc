import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def show_boxes(box,ax):
    (x1, y1), (sizeX1, sizeY1) = box
    rect1 = patches.Rectangle((x1, y1), sizeX1, sizeY1, linewidth=1, edgecolor='r', facecolor='none')

    ax.add_patch(rect1)


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
        self.gps=((400, 500),(-190, 0))
        #n=random.randrange(1,4)
        #self.border(n,self.y//6)
        #n=random.randrange(1,4)
        #self.border(n,-self.y//6)
        self.walls,self.wound,self.boxes=organize((self.x,self.y))
        rx=self.x//2-40
        ry=0
        self.rescue=(rx, ry)
        percobox(self.boxes,(rx-20,ry-40))
        percobox(self.boxes,(rx-20,ry+40))
        n=random.randrange(3,7)
        
        #self.walls = [((0,-250),(0,250))]
        print(self.wound)
        self.wounded_pos=self.wound
        #percolation(self.walls,(0,self.y//4))
        #percolation(self.walls,(0,-self.y//4))
        print(self.walls)
        self.gps=((400, 500),(-190, 0))
        print("\n\n\nlook at Me!!!!!\n\n\n")
        self.drones_pos = []
        for i in range(3):
            self.drones_pos.append(freespace((0,-self.y//4),(self.x//4,self.y//4),self.boxes))
        print("Position :",self.drones_pos)

def openspace(size,nbmpers,nbblocs):
    boxes= []
    (x,y)=size
    for i in range(nbblocs):
        sx,sy = random.randrange(10,x//2),random.randrange(10,y//2)
        px,py=random.randrange(-x//2+40,x//2-sx-40),random.randrange(-y+sy+40,y-40)
        rd=random.random()
        if px<-x//2+40:px=-x//2
        elif py<-y+40:py=y
        elif px+sx>x//2-40:px=x//2+sx
        elif py+sy>y-40:py=-y+sy
        if rd>0.5:
            pass
            #print("px,py : ",px,py, "x,y : ",x,y)
        else:
            if rd>0.375:px=-x//2
            elif rd>0.25:px=x//2-sx
            elif rd>0.125:py=-y+sy
            else:py=y
        for b in boxes:
            if boxintersect(b,((px,py),(sx,sy))):
                break
        else:
            boxes.append(((px,py),(sx,sy)))


    wounded=[]
    for i in range(nbmpers):
        xp,yp=freespace((0,0),(x,2*y),boxes)
        for b in boxes:
            if boxintersect(b,((xp,yp),(5,5))):
                break
        else:
            wounded.append((xp, yp))
    return boxes,wounded

def place_openspace(center,size,nbmpers,nbmurs,angle):
    x,y=size
    size=(x,y//2)
    if (angle// 90)%2==1:
        size=(y,x//2)
    else:
        size=(x,y//2)

    boxes,wounded=openspace(size,nbmpers//2,nbmurs)
    print(boxes)
    
    for i in range(len(boxes)):
        for j in range(i+1,len(boxes)):
            if boxintersect(boxes[i],boxes[j]):
                print("\n Error detected \n")
    #_, ax = plt.subplots()
    #for b in boxes:
    #    show_boxes(b,ax)
    #ax.set_xlim(-x//2, x//2)
    #ax.set_ylim(-y, y)
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.title('Box Intersection')
    #plt.grid(True)
    #plt.show()
    
    #boxes=rotateW(boxes,(0,0),angle)
    wounded=rotateL(wounded,(0,0),angle+180)
    print(boxes)
    fence=[((x//2,y//2),(-x//2,y//2)),((-x//2,y//2),(-x//2,-y//2)),((-x//2,-y//2),(x//2,-y//2)),((x//2,-y//2),(x//2,y//2))]
    print(wounded)
    return translateW(fence,center),translateL(wounded,center),translateB(boxes,center)

def boxintersect(b1,b2):
    (x1, y1), (sizeX1, sizeY1) = b1
    (x2, y2), (sizeX2, sizeY2) = b2
    #print(x1 + sizeX1 +40,x2)
    #print(x2 + sizeX2 +40,x1)
    #print(y1 + sizeY1 +40,y2)
    #print(y2 + sizeY2 +40,y1)
    if x1 + sizeX1 +40<= x2 or x2 + sizeX2+40 <= x1 or y1 + sizeY1+40 <= y2 or y2 + sizeY2 +40<= y1:
        return False
    else:
        return True

def place_corridor(center,size,nbmpers,nbmurs,angle):
    x,y=size
    size=(x,y//2)
    if (angle// 90)%2==1:
        size=(y,x//2)
    else:
        size=(x,y//2)

    walls1,wounded1=corridor(size,nbmpers//2,nbmurs)
    print(walls1)
    walls1=rotateW(walls1,(0,0),angle)
    wounded1=rotateL(wounded1,(0,0),angle+180)
    print(walls1)
    walls2,wounded2=corridor(size,(nbmpers+1)//2,nbmurs)
    walls2=rotateW(walls2,(0,0),angle+180)
    wounded2=rotateL(wounded2,(0,0),angle+180)
    fence=[((x//2,y//2),(-x//2,y//2)),((-x//2,y//2),(-x//2,-y//2)),((-x//2,-y//2),(x//2,-y//2)),((x//2,-y//2),(x//2,y//2))]
    
    print(wounded1+wounded2)
    return translateW(walls1+walls2+fence,center),translateL(wounded1+wounded2,center),[]
    



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
        wounded.append((random.randrange(-x//2,x//2), random.randrange(-y,y)))
    print(wounded)
    return walls,wounded


def rotate(obj,center,angle):
    #print("my argument is :",obj)
    cos,sin=0,0
    if (angle// 90)%4==1:
        sin=1
    elif angle// 90%4==2:
        cos=-1
    elif angle// 90%4==3:
        sin=-1
    elif angle// 90%4==0:
        cos=1
    x,y=obj[0]-center[0],obj[1]-center[1]
    x,y=(x*cos-y*sin),(x*sin+y*cos)
    x,y=x+center[0],y+center[1]
    return (x+center[0],y+center[1])

def rotateW(walls,center,angle):
    return [(rotate(s,center,angle),rotate(e,center,angle)) for (s,e) in walls]
def rotateL(l,center,angle):
    return [rotate(o,center,angle) for o in l]

def translate(obj,delta):
    return (obj[0]+delta[0],obj[1]+delta[1])
def translateW(walls,delta):
    return [(translate(s,delta),translate(e,delta)) for (s,e) in walls]
def translateB(boxes,delta):
    return [(translate(s,delta),e) for (s,e) in boxes]
def translateL(l,delta):
    return [translate(o,delta) for o in l]

def organize(size,n=random.randrange(1)):
    mapX=size[0]
    mapY=size[1]

    x=mapX//2
    y=mapY//2
    rwalls,rwound,rboxes=[],[],[]
    walls,wound,boxes=place_openspace((-x//2,y//2),(x,y),10,3,0)
    rwalls+=walls;rwound+=wound;rboxes+=boxes
    walls,wound,boxes=place_openspace((-x//2,-y//2),(x,y),14,5,0)
    rwalls+=walls;rwound+=wound;rboxes+=boxes
    walls,wound,boxes=place_openspace((x//2,0),(x,y*2),28,5,90)
    rwalls+=walls;rwound+=wound;rboxes+=boxes
    rboxes.append(((x-60,0),(x,y)))
    percolation(rwalls,(0,y//2))
    percobox(rboxes,(0,y//2))
    percolation(rwalls,(0,-y//2))
    percobox(rboxes,(0,-y//2))
    print(rwound)

    return rwalls,rwound,rboxes

def percolation(walls,point):
    #print("Walls :",walls)
    scalaire=lambda x,y:x[0]*y[0]+x[1]*y[1]
    x,y=point
    broken=[]
    for i,w in enumerate(walls):
        (a,b),(c,d)=w
        xw,yw=x-a,y-b
        c,d=c-a,d-b
        #print("c,d :", c, d)
        #print("point : ",xw,yw)
        wallsize=(scalaire((c,d),(c,d)))**0.5
        ux,uy=c/wallsize,d/wallsize
        #print("U : ",ux,uy)
        prj=scalaire((xw,yw),(ux,uy))
        projx,projy=prj*ux,prj*uy
        #print("\n projx , projy",projx,projy," c,d ",c,d)
        if abs(projx)>abs(c) or abs(projy)>abs(d):
            #print(" ***** *** far away \n\n\n ***")
            continue
        #print("proj : ",projx,projy)
        dist=(xw-projx)**2+(yw-projy)**2
        #print("dist :", dist)
        if dist<1600:
            h=(40**2-dist)**0.5
            breakpt1=(int(projx-h*ux),int(projy-h*uy))
            breakpt2=(int(projx+h*ux),int(projy+h*uy))
            #print("breakpt1 : ",breakpt1)
            if scalaire(breakpt1,(ux,uy))>0:
                #print("In Walls : ",((a,b),(a+breakpt1[0],b+breakpt1[1])), "replaces ",walls[i])
                walls[i]=((a,b),(a+breakpt1[0],b+breakpt1[1]))
                #print("breakpt1  other ref: ",(a+breakpt1[0],b+breakpt1[1]))
                if scalaire((breakpt2[0]-c,breakpt2[1]-d),(ux,uy))<0:
                    #print("\nlook\n break point,u, then dot",breakpt2,(ux,uy),scalaire(breakpt2,(ux,uy)),"\n")
                    #print("Added to broken : ",((c+a,d+b),(a+breakpt2[0],b+breakpt2[1])))
                    #print( "a,b,c,d :",a,b,c,d)
                    broken.append(((c+a,d+b),(a+breakpt2[0],b+breakpt2[1])))
            else:
                walls[i]=((c+a,d+b),(a+breakpt2[0],b+breakpt2[1]))
    for w in broken:
        walls.append(w)
    #print(walls)


def percobox(boxes,point):
    for i,b in enumerate(boxes):
        print("Box in percolation :", b)
        print("Percolation point:", point)
        if boxintersect(b,(point,(0,0))):
            if point[0]-40-b[0][0]>0:
                boxes[i]=((b[0][0],b[0][1]),(point[0]-40-b[0][0],b[1][1]))
                print("width changed")
            elif point[1]-40-b[1][0]>0:
                boxes[i]=((b[0][0],b[0][1]),(b[1][0],point[1]-40-b[1][0]))
                print("height changed")
            else:
                del boxes[i]
                print("box destroyed")
def freespace(center,size,boxes):
    cx,cy=center
    sx,sy=size
    while 1:
        x=random.randrange(cx-sx//2,cx+sx//2)
        y=random.randrange(cy-sy//2,cy+sy//2)
        for b in boxes:
            if boxintersect(b,((x,y),(0,0))):
                continue
        break
    return (x,y)