import numpy as np
from opinions_network import find_ground_state_min_cut, set_remaining_nodes_negative

def greedy_placement_with_T(G, opinions, positive_indices, negative_indices, total_iterations=100):
    fixed_nodes = set(positive_indices).union(set(negative_indices))
    positive_counts = []  
    
    for iteration in range(total_iterations):
        T = total_iterations - iteration  
        
        best_node = None
        max_positive_count = -1
        
        if T == 0:
            ground_state = find_ground_state_min_cut(G, positive_indices, negative_indices)
            opinions = ground_state  
            set_remaining_nodes_negative(opinions)
            remaining_neutral_nodes = np.where(opinions == 0)[0]  
            negative_indices = np.append(negative_indices, remaining_neutral_nodes)  
            break 
        
        for node in range(len(opinions)):
            if node in fixed_nodes:
                continue
      
            prob_positive = calc_positive_probability(G, node, opinions, T)
            
            if np.random.rand() < prob_positive:
                original_opinion = opinions[node]
                opinions[node] = 1 
                ground_state = find_ground_state_min_cut(G, positive_indices, negative_indices)
                positive_count = np.sum(ground_state == 1)
                
                if positive_count > max_positive_count:
                    max_positive_count = positive_count
                    best_node = node
                
                opinions[node] = original_opinion
        
        if best_node is not None:
            fixed_nodes.add(best_node)
            opinions[best_node] = 1  
            positive_indices = np.append(positive_indices, best_node)  
            
        update_influenced_nodes(G, opinions, T)
        
        total_positive_count = np.sum(opinions == 1)
        positive_counts.append(total_positive_count)

    return opinions, fixed_nodes, positive_counts, negative_indices  

def calc_positive_probability(G, node, opinions, T, scaling_factor=0.45):
    neighbor_opinions = np.array([opinions[neighbor] for neighbor in G.neighbors(node)])
    weights = np.array([G[node][neighbor]['weight'] for neighbor in G.neighbors(node)])
    
    influence_sum = np.sum(weights * neighbor_opinions)
    prob_positive = np.exp(influence_sum / T) / (np.exp(influence_sum / T) + np.exp(-influence_sum / T))
    prob_positive *= scaling_factor
    
    return prob_positive

def update_influenced_nodes(G, opinions, T):
    for node in range(len(opinions)):
        if opinions[node] == 0:  
            prob_positive = calc_positive_probability(G, node, opinions, T)
            if np.random.rand() < prob_positive:
                opinions[node] = 1  

