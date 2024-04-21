import numpy as np

def process_communication_sensor(self, drone):

    if self.communicator:
        received_messages = self.communicator.received_messages

        # each message contains a list of edges
        for msg in received_messages:
            edges = np.fromstring(msg, dtype=object, sep=',') # transform string to numpy array
            drone.graph.add_edges(edges) # add edges to the graph
