
import numpy as np
from math import sqrt, pi
import networkx as nx
import matplotlib.pyplot as plt

class EuclideanNetwork:
    def __init__(self, node_count=100, delta=1.0):
        '''
        Initializes the Euclidean Network with a given number of nodes and delta parameter.
        
        Parameters:
        - node_count: Number of nodes in the network.
        - delta: Exponent for determining probability of random link creation.
        '''
        self.node_count = node_count  # Number of nodes
        self.delta = delta  # Delta parameter for random link probability
        self.node_positions = None  # Store node positions
        self.network_graph = None  # Store the generated graph
        
        # Create node positions in a circular layout
        angle_step = 2 * pi / self.node_count  # Angle between consecutive nodes
        angles = np.arange(0, 2 * pi, angle_step)  # Angles for each node position
        self.node_positions = np.array([np.cos(angles), np.sin(angles)]).T  # Positions in 2D plane

        self._build_network()  # Generate the graph with edges
        

    def _build_network(self, delta=None):
        '''
        Private method to generate the graph with both adjacent and random links.
        Uses the original random link generation logic based on distance and delta.
        
        Parameters:
        - delta: Exponent for determining probability of random link creation. If None, uses self.delta.
        '''
        if delta is None:
            delta = self.delta  # Use the instance's delta if not provided
            
        self.network_graph = nx.Graph()  # Initialize an empty graph
        self.network_graph.add_nodes_from(range(self.node_count))  # Add nodes to the graph

        # Create links between adjacent nodes (forming a circular chain)
        for i in range(self.node_count - 1):
            self.network_graph.add_edge(i, i + 1)  # Connect node i with i+1
        self.network_graph.add_edge(self.node_count - 1, 0)  # Connect last node to the first (circular)

        # Normalize the distance based on the maximum distance between two adjacent nodes
        norm = self._calculate_distance(self.node_positions[0], self.node_positions[1])

        created_nodes = 0  # Counter for created random links
        while created_nodes < self.node_count:
            rand1 = np.random.randint(0, self.node_count)
            rand2 = np.random.randint(0, self.node_count)
            
            if rand1 == rand2:  # Skip if both nodes are the same
                continue

            # Calculate the Euclidean distance between the two nodes
            distance = self._calculate_distance(self.node_positions[rand1], self.node_positions[rand2])

            # Create random links based on distance and delta parameter
            if np.random.rand() < (distance / norm) ** (-delta):  # Use delta for probability
                self.network_graph.add_edge(rand1, rand2)
                created_nodes += 1



    def _calculate_distance(self, point1, point2):
        '''
        Calculates the Euclidean distance between two points.
        
        Parameters:
        - point1: First point (as a 2D coordinate).
        - point2: Second point (as a 2D coordinate).
        
        Returns:
        - Euclidean distance between point1 and point2.
        '''
        return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    
    def display_network(self):
        '''
        Plots the generated network using matplotlib.
        It shows both adjacent and random links similar to the provided code.
        '''
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Add title including the value of delta
        ax.set_title(r'$\delta$ = {}'.format(self.delta))
        
        # Get node positions
        positions = {i: self.node_positions[i] for i in range(self.node_count)}
    
        # Plot the nodes as black dots
        ax.scatter(self.node_positions[:, 0], self.node_positions[:, 1], color='black')
    
        # Draw the edges (connections between nodes)
        for i in range(self.node_count):
            for j in range(i + 1, self.node_count):
                if self.network_graph.has_edge(i, j):
                    x_values = [self.node_positions[i][0], self.node_positions[j][0]]
                    y_values = [self.node_positions[i][1], self.node_positions[j][1]]
                    ax.plot(x_values, y_values, color='black', linewidth=0.5)
    
        # Adjust the axis to remove the ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    
        plt.show()

    def get_graph(self):
        """Returns the networkx graph created"""
        return self.network_graph
        
    def display_adjacency_matrix(self, invertY=True):
        """Show the adjacency matrix with matplotlib
        
        Parameter:
        inverder: bool
            set to true to have the (0,0) point to be 
            on the lowest-left corner (instead of upper left)
        Return:
        adj_matrix: nx.adjacency_matrix
            The adjacency matrix of the graph
        """

        adj_matrix = nx.adjacency_matrix(self.network_graph)
        plt.imshow(adj_matrix.toarray(), cmap='hot_r')
        
        if invertY:
            plt.gca().invert_yaxis()

        plt.xlabel('site i')
        plt.ylabel('site j')
        plt.show()

        return adj_matrix

# Example of how to use the class:
if __name__ == "__main__":
    network = EuclideanNetwork(node_count=50, delta=1.5)
    network.display_network()
