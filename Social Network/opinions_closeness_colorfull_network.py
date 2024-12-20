import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm

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

def visualize_weighted_network_with_opinions(G, grid_size, opinions):
    pos = {node: (node // grid_size, node % grid_size) for node in G.nodes()}
    edges = G.edges(data=True)

    # Extract weights and scale them for better visibility
    weights = [data['weight'] for _, _, data in edges]
    max_weight = max(weights)
    min_weight = min(weights)
    normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]

    # Generate edge colors based on normalized weights (from black to blue)
    edge_colors = [cm.Blues(weight) for weight in normalized_weights]  # Using the "Blues" colormap

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color nodes based on opinions
    node_colors = ['red' if opinion == 1 else 'blue' if opinion == -1 else 'black' for opinion in opinions]

    # Draw the network nodes with different colors based on opinions
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, ax=ax)

    # Draw the edges with color based on the weight
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=2, ax=ax)

    # Add color bar (legend) for edge weights
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
    sm.set_array([])  # Empty array is just for the color bar
    cbar = plt.colorbar(sm, ax=ax)  # Use the Axes object `ax` for the color bar
    cbar.set_label("Edge Weight")

    # Display title
    ax.set_title("Network with Opinions and Edge Weights")
    ax.axis("off")  # Turn off the axis for a cleaner visualization
    plt.show()

def initialize_opinions(num_nodes, num_positive_seeds, num_negative_seeds):
    opinions = np.zeros(num_nodes)
    positive_indices = np.random.choice(num_nodes, num_positive_seeds, replace=False)
    remaining_indices = list(set(range(num_nodes)) - set(positive_indices))
    negative_indices = np.random.choice(remaining_indices, num_negative_seeds, replace=False)
    opinions[positive_indices] = 1
    opinions[negative_indices] = -1
    return opinions, positive_indices, negative_indices

# Example usage
grid_size = 10
num_positive_seeds = 10
num_negative_seeds = 20

# Create the social network
G, _ = create_2d_social_network(grid_size)

# Initialize opinions
opinions, positive_indices, negative_indices = initialize_opinions(grid_size ** 2, num_positive_seeds, num_negative_seeds)

# Visualize the network with colorful edges and nodes based on opinions
visualize_weighted_network_with_opinions(G, grid_size, opinions)
