import numpy as np

def select_nodes_by_weight(G, opinions, num_nodes_to_transform):
    potential_nodes = []
    for node in G.nodes():
        if opinions[node] == 0:  
            weight_sum = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node) if opinions[neighbor] == 1)
            potential_nodes.append((node, weight_sum))
    
    potential_nodes.sort(key=lambda x: x[1], reverse=True)
    selected_nodes = [node for node, _ in potential_nodes[:num_nodes_to_transform]]
    
    return selected_nodes

def transform_nodes(opinions, selected_nodes):
    for node in selected_nodes:
        opinions[node] = 1  

def track_positive_counts(opinions, counts):
    counts.append(np.sum(opinions == 1))

def degree_based_placement(G, opinions, extra_positive_seeds, total_iterations, num_nodes_to_transform):
    positive_counts = []
    
    for iteration in range(total_iterations):
        T = total_iterations - iteration
        
        selected_nodes = select_nodes_by_weight(G, opinions, num_nodes_to_transform)
        transform_nodes(opinions, selected_nodes)
        update_influenced_nodes(G, opinions, T)
        
        track_positive_counts(opinions, positive_counts)
    
    return positive_counts

def calc_positive_probability(G, node, opinions, T, scaling_factor=0.35):
    neighbor_opinions = np.array([opinions[neighbor] for neighbor in G.neighbors(node)])
    weights = np.array([G[node][neighbor]['weight'] for neighbor in G.neighbors(node)])
    
    influence_sum = np.sum(weights * neighbor_opinions)
    prob_positive = np.exp(influence_sum / T) / (np.exp(influence_sum / T) + np.exp(-influence_sum / T))
    
    prob_positive *= scaling_factor  # Apply scaling factor
    
    return prob_positive

def update_influenced_nodes(G, opinions, T):
    for node in range(len(opinions)):
        if opinions[node] == 0:  
            prob_positive = calc_positive_probability(G, node, opinions, T)
            if np.random.rand() < prob_positive:
                opinions[node] = 1  

