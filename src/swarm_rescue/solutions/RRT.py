'''We model map as the following Graph G(V,E)
V = points with integer coordinates
E = all line segments connecting vertices
Note: The real edge set might be smaller than E. If a Wall blocks an edge, we say it doesn't really exist.

Goal: During exploration, we want to build a spanning tree T(V, E')
When drones find a wounded person:
    if path to rescue center exists in random tree => go to rescue center
    else continue building F until path exists

Ideally, all drones would share the same spanning tree, but I don't know if that's possible :(

                                    Exploration:
suppose drone is at node u
a random sample u_Rand is spawned in the collision-free space
STEERING -> we take a small step towards u_Rand to expand the tree in this direction
we add the steered node u_next to the tree

IMPORTANT: The first time Rescue Center is detected by a drone, node must be marked in the tree.
'''
import math
import random

class Node:
    '''Represents a node in graph G'''
    def __init__(self, parentNode, x, y):
        self.parentNode = parentNode #if initial node, we say parentNode=None
        self.x = x
        self.y = y
        self.level = 0 if (parentNode is None) else parentNode.level + 1 #level in spanning tree
        self.adjList = [] #list of adjacent nodes in Spanning Tree

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.x == other.x and self.y == other.y
        return False
    
    def addEdge(self, v):
        self.adjList.append(v)

class RRT:
    '''Data Structure for Rapidly-Exploring Random Tree (from Lavalle's book)'''
    def __init__(self):
        self.rescueCenterNode = None
        self.nodes = []
        self.pointNodeDict = {}

    def alreadyVisited(self, p):
        '''check if point p is in the Tree'''
        return not (self.pointNodeDict.get(p) is None)

    def getNodeIndex(self, p):
        '''get index of Node corresponding to point p in the Tree'''
        if(self.alreadyVisited(p)):
            return self.pointNodeDict.get(p)
        return None

    def addNode(self, p, parentNode):
        '''add point p to tree'''
        u = Node(parentNode, p[0], p[1])
        self.pointNodeDict[p] = len(self.nodes)
        self.nodes.append(u)

    def addEdge(self, p1, p2):
        '''add edge (p1, p2) to tree'''
        u = self.getNodeIndex(p1)
        v = self.getNodeIndex(p2)
        self.nodes[u].addEdge(self.nodes[v])
        self.nodes[v].addEdge(self.nodes[u])

    def steering(self, x, y, compass_angle, far_angles):
        ''' spawn random sample u_Rand to expand RRT (Exploration Phase)'''
        candidate_points = []
        for angle in far_angles: #far_angles define collision-free space
            direction_angle = compass_angle + angle
            direction_vector = (math.cos(direction_angle), math.sin(direction_angle))

            for i in range(2, 4):
                point_in_direction = (round(x + 10*(i+1)*direction_vector[0]), round(y + 10*(i+1)*direction_vector[1]))
                if(not self.alreadyVisited(point_in_direction)):
                    candidate_points.append((point_in_direction, direction_angle))

        u_rand = random.choice(candidate_points)
        return u_rand[0], u_rand[1]

    def lca(self, u, v):
        '''compute lowest common ancestor of nodes u and v'''
        if u.level < v.level:
            u, v = v, u #u is lowest node in tree
        #since we take O(|V|) to build explicit path, O(|V|) solution is fine here
        while u.level > v.level:
            u = u.parentNode
        while u != v:
            u = u.parentNode
            v = v.parentNode
        return u
    
    def build_path(self, u, v):
        '''compute path between nodes u and v'''
        lca_uv = self.lca(u, v)
        path_u = [] #we build path from v to lca_uv
        while(u != lca_uv):
            path_u.append(u)
            u = u.parentNode
        path_u.append(lca_uv)
        
        path_v = [] #we build path from v to lca_uv
        while(v != lca_uv):
            path_v.append(v)
            v = v.parentNode
        path_v = path_v[::-1] #we reverse to get path from lca_uv to v

        path_u.append(path_v)
        return path_u