from opinions_network import create_2d_social_network, initialize_opinions, set_remaining_nodes_negative
from greedy_placement import greedy_placement_with_T
from degree_based_placement import degree_based_placement
from plotting import visualize_network, plot_positive_counts

'''
    Parameters
'''
grid_size = 10  
num_positive_seeds = 10
num_negative_seeds = 20
extra_positive_seeds = 10
num_nodes_to_transform = 1 
total_iterations = 10  

# The opinions network
social_network, long_range_edges = create_2d_social_network(grid_size)

opinions, positive_indices, negative_indices = initialize_opinions(
    grid_size ** 2, num_positive_seeds, num_negative_seeds)

# Plot before 
visualize_network(social_network, long_range_edges, grid_size, opinions, title="Before Placement")

# Greedy placement
final_opinions_greedy, final_fixed_nodes_greedy, positive_counts_greedy, negative_indices_greedy = greedy_placement_with_T(
    social_network, opinions.copy(), positive_indices.copy(), negative_indices.copy(), total_iterations)

# Degree-based placement
positive_counts_degree = degree_based_placement(social_network, opinions.copy(), extra_positive_seeds, total_iterations, num_nodes_to_transform)

set_remaining_nodes_negative(final_opinions_greedy)

# Plotting
visualize_network(social_network, long_range_edges, grid_size, final_opinions_greedy, title="After Greedy Placement")
plot_positive_counts(positive_counts_greedy, positive_counts_degree)
