import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def create_2d_social_network(grid_size, long_range_prob=1):
    G = nx.Graph()
    num_nodes = grid_size ** 2  
    G.add_nodes_from(range(num_nodes))

    def node_index(row, col):
        return row * grid_size + col

    def euclidean_distance(node1, node2):
        row1, col1 = divmod(node1, grid_size)
        row2, col2 = divmod(node2, grid_size)
        return np.sqrt((row1 - row2)**2 + (col1 - col2)**2)

    for row in range(grid_size):
        for col in range(grid_size):
            node = node_index(row, col)
            if col < grid_size - 1:
                right_neighbor = node_index(row, col + 1)
                distance = euclidean_distance(node, right_neighbor)
                weight = 1 / (distance + 1e-6)  # Avoid division by zero
                G.add_edge(node, right_neighbor, weight=weight)
            if row < grid_size - 1:
                bottom_neighbor = node_index(row + 1, col)
                distance = euclidean_distance(node, bottom_neighbor)
                weight = 1 / (distance + 1e-6)  # Avoid division by zero
                G.add_edge(node, bottom_neighbor, weight=weight)

    long_range_edges = []
    for node in range(num_nodes):
        if np.random.rand() < long_range_prob:
            target_node = np.random.randint(num_nodes)
            while target_node == node or target_node in [t[1] for t in long_range_edges]:
                target_node = np.random.randint(num_nodes)

            distance = euclidean_distance(node, target_node)
            weight = 1 / (distance + 1e-6)
            G.add_edge(node, target_node, weight=weight)
            long_range_edges.append((node, target_node))  
            
    return G, long_range_edges

def visualize_weighted_network(G, grid_size):
    pos = {node: (node // grid_size, node % grid_size) for node in G.nodes()}
    edges = G.edges(data=True)
    
    # Extract weights and scale them for better visibility
    weights = [data['weight'] for _, _, data in edges]
    scaled_weights = [5 * w / max(weights) for w in weights]  # Normalize and scale for better visualization

    # Draw the nodes
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color="blue")
    
    # Draw the edges with scaled widths
    nx.draw_networkx_edges(G, pos, width=scaled_weights, edge_color="black")

    plt.title("Network with Weights Proportional to Distance")
    plt.show()

def initialize_opinions(num_nodes, num_positive_seeds, num_negative_seeds):
    opinions = np.zeros(num_nodes)
    positive_indices = np.random.choice(num_nodes, num_positive_seeds, replace=False)
    remaining_indices = list(set(range(num_nodes)) - set(positive_indices))
    negative_indices = np.random.choice(remaining_indices, num_negative_seeds, replace=False)
    opinions[positive_indices] = 1
    opinions[negative_indices] = -1
    return opinions, positive_indices, negative_indices

def find_ground_state_min_cut(G, positive_indices, negative_indices):
    G_extended = G.copy()
    
    super_source = 'source'
    super_sink = 'sink'
    G_extended.add_node(super_source)
    G_extended.add_node(super_sink)
    
    for node in positive_indices:
        G_extended.add_edge(super_source, node, weight=np.inf)  
    
    for node in negative_indices:
        G_extended.add_edge(super_sink, node, weight=np.inf) 
        
    cut_value, partition = nx.stoer_wagner(G_extended)
    
    reachable_from_source = partition[0] if super_source in partition[0] else partition[1]
    
    ground_state = np.zeros(len(G)) 
    for node in G.nodes():
        if node in reachable_from_source:
            ground_state[node] = 1  
        else:
            ground_state[node] = -1 
    
    return ground_state

def set_remaining_nodes_negative(opinions):
    for node in range(len(opinions)):
        if opinions[node] == 0:
            opinions[node] = -1
            
grid_size = 10
G, _ = create_2d_social_network(grid_size)
visualize_weighted_network(G, grid_size)