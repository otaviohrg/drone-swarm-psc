import os
import matplotlib.pyplot as plt
import heapq
import numpy as np

class MyDroneGraph:
    '''
    Drone creates a map using a graph data structure (implemented with adjacency list)
    Nodes have (x,y) coordinates and a unique id
    Drones exchange edge lists which make the graph denser
    '''

    def __init__(self):
        self.adj_list = {}
        self.node_counter = 0  # used to generate node ids
        self.coordinates_to_node = {}  # convert coordinate to id
        self.node_to_coordinates = {}
        self.latest_edges = np.array([]) #used to communicate latest edges to other drones
        
    
    def add_vertex(self, x, y):
        coordinates = (x,y)
        node_id = self.node_counter
        self.node_counter += 1
        self.adj_list[node_id] = {}
        self.coordinates_to_node[coordinates] = node_id
        self.node_to_coordinates[node_id] = coordinates
        return node_id
    
    def add_edge(self, ux, uy, vx, vy, weight, must_communicate):
        '''
        If edge was communicated by another drone => must_communicate is False
        else must_communicate is True (add to lastest_edges list)
        '''
        u = self.coordinates_to_node.get((ux, uy))
        v = self.coordinates_to_node.get((vx, vy))
        if u is None: #we check if node exists in the graph
            u = self.add_vertex(ux, uy)
        if v is None:
            v = self.add_vertex(vx, vy)
        self.adj_list[u][v] = weight
        self.adj_list[v][u] = weight
        if(must_communicate):
            self.latest_edges = np.append(self.latest_edges, (ux,uy,vx,vy,weight))

    
    def add_edges(self, edges):
        for edge in edges:
            ux, uy, vx, vy, weight = edge
            self.add_edge(ux, uy, vx, vy, weight)
    
    def heuristic(self, u, v):
        return abs(u[0] - v[0]) + abs(u[1] - v[1])
    
    def shortest_path(self, start, end): #We use A* algorithm for shortest path
        pq = [(0, start)]
        visited = set()
        parent = {}
        while pq:
            cost, node = heapq.heappop(pq)
            if node == end:
                path = []
                while node is not None:
                    path.append(node)
                    node = parent.get(node)
                return list(reversed(path))  # Return the path in the correct order
            if node not in visited:
                visited.add(node)
                for neighbor, weight in self.adj_list[node].items():
                    if neighbor not in visited:
                        heapq.heappush(pq, (cost + weight + self.heuristic(self.node_to_coordinates[neighbor], self.node_to_coordinates[end]), neighbor))
                        parent[neighbor] = node
        return []  # if path does not exist, return an empty list

    def plot_graph(self, filename="graph.png"):
        fig, ax = plt.subplots()
        for u, neighbors in self.adj_list.items():
            for v, _ in neighbors.items():
                ax.plot([self.node_to_coordinates[u][0], self.node_to_coordinates[v][0]], [self.node_to_coordinates[u][1], self.node_to_coordinates[v][1]], 'k-')
        for node_id, coordinates in self.node_to_coordinates.items():
            ax.plot(coordinates[0], coordinates[1], 'bo')
            ax.text(coordinates[0], coordinates[1], str(node_id))
        ax.set_aspect('equal', 'box')
        plt.savefig(filename)
        plt.close(fig)
