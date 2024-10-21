import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def visualize_network(G, long_range_edges, grid_size, opinions, title="Network Visualization"):
    pos = {(node): (node // grid_size, node % grid_size) for node in G.nodes()} 

    nx.draw_networkx_edges(G, pos, edgelist=[edge for edge in G.edges if edge not in long_range_edges], edge_color='black', width=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=[edge for edge in long_range_edges if opinions[edge[0]] != opinions[edge[1]]], edge_color='green', width=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=long_range_edges, edge_color='black', width=0.5)

    positive_indices = np.where(opinions == 1)[0]
    negative_indices = np.where(opinions == -1)[0]

    nx.draw_networkx_nodes(G, pos, nodelist=positive_indices, node_size=25, node_color='red')
    nx.draw_networkx_nodes(G, pos, nodelist=negative_indices, node_size=25, node_color='blue')

    other_nodes = set(G.nodes()) - set(positive_indices) - set(negative_indices)
    nx.draw_networkx_nodes(G, pos, nodelist=list(other_nodes), node_size=25, node_color='black')

    plt.title(title)
    plt.xlim(-1, grid_size)
    plt.ylim(-1, grid_size)
    plt.show()

def plot_positive_counts(positive_counts, positive_counts2):
    plt.figure(figsize=(10, 6))
    plt.plot(positive_counts, marker='o', label='Greedy algorithm')
    plt.plot(range(len(positive_counts2)), positive_counts2, marker='o', color='red', label='Degree-based algorithm')
    plt.title('Evolution of Positive Nodes Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Positive Nodes')
    plt.xticks(range(len(positive_counts2))) 
    plt.legend()
    plt.grid(True)
    plt.show()
