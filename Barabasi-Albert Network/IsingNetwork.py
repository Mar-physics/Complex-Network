import numpy as np
import random
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt

class IsingSystem:
    
    def __init__(self, graph, coupling_constant=1.0, num_iterations=10000, temp_range=np.arange(0, 10, 0.1), 
                 is_symmetric=True, external_field_strength=0.0, initial_spin_probability=1):
        
        self.model_name = "IsingSystem"
        self.num_nodes = graph.number_of_nodes()
        self.graph_structure = graph
        self.coupling_constant = coupling_constant
        self.num_iterations = num_iterations
        self.initial_spin_probability = initial_spin_probability
        self.temp_range = temp_range
        self.is_symmetric = is_symmetric
        self.external_field_strength = external_field_strength
        self.simulation_results = []
        self.neighbor_list = {node: list(self.graph_structure.neighbors(node)) for node in self.graph_structure.nodes()}
        
    def initialize_spins(self, spin_probability):
        self.spins = np.random.choice([-1, 1], self.num_nodes, p=[1 - spin_probability, spin_probability])

    def update_coupling_constant(self, coupling_constant):
        self.coupling_constant = coupling_constant

    def update_num_iterations(self, num_iterations):
        self.num_iterations = num_iterations

    def update_initial_spin_probability(self, spin_probability):
        if np.abs(spin_probability) > 1:
            raise ValueError("Initial spin probability should be between 0 and 1")
        self.initial_spin_probability = spin_probability

    def update_external_field_strength(self, external_field_strength):
        self.external_field_strength = external_field_strength
    
    def _calculate_magnetization(self):
        return np.sum(self.spins)
    
    def _calculate_energy(self):
        energy = 0.0
        for node in range(self.num_nodes):
            neighbor_spin_sum = np.sum(self.spins[self.neighbor_list[node]])
            energy += self.spins[node] * neighbor_spin_sum
        return -0.5 * self.coupling_constant * energy - self.external_field_strength * self._calculate_magnetization()
    
    def _flip_spin(self, spin):
        return -spin if self.is_symmetric else 1 if spin == 0 else 0

    def _metropolis_step(self, temperature):
        beta = 1 / temperature
        random_node = np.random.randint(0, self.num_nodes)  # Randomly select a node
        current_spin = self.spins[random_node]               # Get the spin of this node
        neighbor_spin_sum = np.sum(self.spins[self.neighbor_list[random_node]])  # Sum of neighboring spins

        new_spin = self._flip_spin(current_spin)
        delta_energy = (self.coupling_constant * neighbor_spin_sum - self.external_field_strength) * (current_spin - new_spin)  # Energy change
        
        if delta_energy < 0 or random.random() < np.exp(-delta_energy * beta):  # Accept the transition
            self.spins[random_node] = new_spin

    def run_simulation(self, temperature, iterations=None):
        """Run the model simulation at a specified temperature using the Metropolis algorithm."""
        if iterations is None:
            iterations = self.num_iterations

        results = {}   # Dictionary to hold results
        magnetizations = np.zeros(iterations)
        energies = np.zeros(iterations)
    
        self.initialize_spins(self.initial_spin_probability)  # Initialize spin configuration    
    
        for _ in tqdm(range(int(iterations / 2))):
            self._metropolis_step(temperature)
        for i in tqdm(range(int(iterations / 2))):
            self._metropolis_step(temperature)
            magnetizations[i] = self._calculate_magnetization()
            energies[i] = self._calculate_energy()

        normalized_magnetization = magnetizations / self.num_nodes
        results['magnetization_per_spin'] = normalized_magnetization.mean()
        results['energy'] = energies.mean()
    
        m4 = normalized_magnetization**4
        m2 = normalized_magnetization**2
        results['binder_cumulant'] = 1 - m4.mean() / (3 * (m2.mean()**2))
    
        M2 = magnetizations**2  # using k_B = 1
        results['susceptibility_per_spin'] = self.num_nodes / (1 * temperature) * (M2.mean() - normalized_magnetization.mean()**2)
    
        E2 = energies**2
        results['specific_heat_per_spin'] = self.num_nodes / ((1 * temperature)**2) * (E2.mean() - energies.mean()**2)

        return results
    
    def iterate_over_temperatures(self, temp_range=None, iterations=None, num_simulations=1, use_parallel=True, verbosity=0):
        """Perform simulations across a range of temperatures and compute results."""
        if temp_range is None:
            temp_range = self.temp_range
        else:
            self.temp_range = temp_range
        
        if iterations is None:
            iterations = self.num_iterations
        else:
            self.num_iterations = iterations

        def compute_means(temp_index):
            results_to_average = np.zeros(num_simulations, dtype=dict)
            
            for sim_index in tqdm(range(num_simulations), leave=False):
                results_to_average[sim_index] = self.run_simulation(temp_range[temp_index], iterations)
            
            keys = ['magnetization_per_spin', 'energy', 'binder_cumulant',
                    'susceptibility_per_spin', 'specific_heat_per_spin']
            averages = {}
            for key in keys:
                values = [result[key] for result in results_to_average]
                averages[key] = np.mean(values)
            return averages
        
        if use_parallel:
            num_cores = multiprocessing.cpu_count()
            self.simulation_results = Parallel(n_jobs=num_cores, verbose=verbosity)(delayed(compute_means)(i) for i in range(len(temp_range)))
        else:
            self.simulation_results = np.zeros(len(temp_range), dtype=dict)    
            
            for i in tqdm(range(len(temp_range))):
                self.simulation_results[i] = compute_means(i)

        return self.simulation_results
    
    def retrieve_data(self,quantity):
        """Return the data of given quantity (or quantities)
        
        Parameter:
        quantity: str
            The quantity that need to be returned.
        
        Return:
        data: np.array
            Array containing the quantity wrt self.temperature_range
        """

        if self.simulation_results is None:
            self.iterate_over_temperatures()

        data = np.array([item[quantity] for item in self.simulation_results])
        return data
    
    def get_temperature_range(self):
        return self.temp_range
    
    def plot(self, quantities=None, ylabels=None):
        """
        Plot specified quantities using matplotlib.
        
        Parameters:
        ----------
        quantities: str or list of str
            The quantities to plot. Can be a single string or a list of strings.
            If not specified, all default quantities are plotted.
        
        ylabels: str or list of str
            Labels for the y-axis of each plot. 
            If not specified, the names of the quantities are used as labels.
        """
        
        # If no quantities are specified, default to these common physical properties
        if quantities is None:
            quantities = ['magnetization_per_spin', 'energy', 'binder_cumulant',
                          'susceptibility_per_spin', 'specific_heat_per_spin']
        elif isinstance(quantities, str):
            # Convert to list if only a single string is provided
            quantities = [quantities]

        # Use quantity names as default y-axis labels if none are provided
        if ylabels is None:
            ylabels = quantities
        elif isinstance(ylabels, str):
            ylabels = [ylabels]

        # Plot each quantity against the temperature, with its corresponding y-axis label
        for quantity, ylabel in zip(quantities, ylabels):
            data = np.array([item[quantity] for item in self.simulation_results])
            plt.figure()
            plt.scatter(self.temp_range, data)
            plt.xlabel('Temperature')
            plt.ylabel(ylabel)
            plt.title(f'{ylabel} vs Temperature')
            plt.show()