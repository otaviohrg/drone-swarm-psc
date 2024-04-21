import os
import math
import matplotlib.pyplot as plt
import heapq
import numpy as np
import random


cnt = 0

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

    def alreadyVisited(self, p):
        '''check if point p is in the graph'''
        return not (self.coordinates_to_node.get(p) is None)
    
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

        # Add vertices if they do not exist
        if u is None:
            u = self.add_vertex(ux, uy)
        if v is None:
            v = self.add_vertex(vx, vy)
        
        # Check if the edge already exists
        if u is not None and v is not None:
            if v in self.adj_list[u]:
                # Edge already exists, do not add it again
                return
        
        # Add edge to the adjacency list
        self.adj_list[u][v] = weight
        self.adj_list[v][u] = weight
        
        # Add edge to latest_edges list if necessary
        if must_communicate:
            self.latest_edges = np.append(self.latest_edges, (ux, uy, vx, vy, weight))

    
    def add_edges(self, edges):
        for edge in edges:
            ux, uy, vx, vy, weight = edge
            self.add_edge(ux, uy, vx, vy, weight, False)
    
    def heuristic(self, u, v):
        return abs(u[0] - v[0]) + abs(u[1] - v[1])
    
    def shortest_path(self, start_coords, end_coords):
        #we specify start and end coordinates to build path
        #we pick nodes that are the closest possible to these coordinates => flexibility and precision

        start_node = min(self.node_to_coordinates.keys(), key=lambda node: ((self.node_to_coordinates[node][0] - start_coords[0]) ** 2 + (self.node_to_coordinates[node][1] - start_coords[1]) ** 2))
        end_node = min(self.node_to_coordinates.keys(), key=lambda node: ((self.node_to_coordinates[node][0] - end_coords[0]) ** 2 + (self.node_to_coordinates[node][1] - end_coords[1]) ** 2))

        pq = [(0, start_node)]
        visited = set()
        parent = {}
        while pq:
            cost, node = heapq.heappop(pq)
            if node == end_node:
                path = []
                while node is not None:
                    path.append(self.node_to_coordinates[node])
                    node = parent.get(node)
                return list(reversed(path))  # Return the path in the correct order
            if node not in visited:
                visited.add(node)
                for neighbor, weight in self.adj_list[node].items():
                    if neighbor not in visited:
                        heapq.heappush(pq, (cost + weight + self.heuristic(self.node_to_coordinates[neighbor], end_coords), neighbor))
                        parent[neighbor] = node
        return []  # if path does not exist, return an empty list

    def plot_graph(self):
        global cnt
        cnt += 1
        if(cnt < 300):
            pass
        cnt = 0
        filename = f'graph-{random.randint(0, 1000)}.png' #unique random filename
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

    def rrt_steering(self, x, y, angle, far_angles):
        """
        Choose the direction with the fewest neighboring points in the graph.
        """
        # Select a random subset of 30 points from the graph
        aux = list(self.node_to_coordinates.values())
        random_points = random.sample(aux, min(30, len(aux)))

        # Initialize variables to keep track of the direction with the fewest points
        min_points = float('inf')
        best_direction = None

        # Iterate over candidate directions and choose the direction with the fewest points
        for direction in far_angles:
            # Calculate the new position in the current direction
            x_candidate = x + 5*math.cos(direction + angle)
            y_candidate = y + 5*math.sin(direction + angle)

            # Count the number of points within a certain radius (e.g., 10 units) around the new position
            num_points = sum(1 for x_p, y_p in random_points if math.sqrt((x_p - x_candidate)**2 + (y_p - y_candidate)**2) <= 10)

            # Update the best direction if the current direction has fewer points
            if num_points < min_points:
                min_points = num_points
                best_direction = direction

        x_chosen = x + 5*math.cos(best_direction + angle)
        y_chosen = y + 5*math.sin(best_direction + angle)

        return x_chosen, y_chosen
