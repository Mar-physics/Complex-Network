from EuclideanNetwork import EuclideanNetwork
from IsingNetwork import IsingSystem
import numpy as np
from tqdm import tqdm

# Definisci l'intervallo di N per includere anche 250
N_range = np.arange(50, 251, 50)  # Include N = 50, 100, 150, 200, 250
U = []  # Lista per memorizzare i dati

# Ciclo per generare e raccogliere i dati
for N in tqdm(N_range):
    g = EuclideanNetwork(N, 0.6)
    model = IsingSystem(g.get_graph(), temp_range=np.arange(0, 5, 0.1))

    model.iterate_over_temperatures(num_simulations=10)
    U.append(model.retrieve_data('binder_cumulant'))

# Salva tutti gli array in un unico file .npz
np.savez('binder_cumulant_data.npz', 
         arr_0=U[0], 
         arr_1=U[1], 
         arr_2=U[2], 
         arr_3=U[3],
         arr_4=U[4])

# Carica i dati dal file .npz
data = np.load('binder_cumulant_data.npz')

# Accedi ai singoli array
binder_cumulant_N50 = data['arr_0']
binder_cumulant_N100 = data['arr_1']
binder_cumulant_N150 = data['arr_2']
binder_cumulant_N200 = data['arr_3']
binder_cumulant_N250 = data['arr_4']


# Esempio di plottaggio
import matplotlib.pyplot as plt

temperatures = np.arange(0, 5, 0.1)

plt.figure(figsize=(10, 6))
plt.plot(temperatures, binder_cumulant_N50, label='N = 50')
plt.plot(temperatures, binder_cumulant_N100, label='N = 100')
plt.plot(temperatures, binder_cumulant_N150, label='N = 150')
plt.plot(temperatures, binder_cumulant_N200, label='N = 200')
plt.plot(temperatures, binder_cumulant_N250, label='N = 250')

plt.title('δ = 0.6')
plt.xlabel('Temperature')
plt.ylabel('Binder Cumulant')
plt.legend()
plt.show()

Tc_range = [2.2, 2.25, 2.3, 2.1, 2.0, 1.8, 1.65, 1.4, 1.2]
v_range = [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3]

# Plotta i risultati scalati
for v in v_range:
    # Calcola gli assi x per diversi valori di N
    xAxis50 = (temperatures - Tc_range[2]) * (50 ** (1/v))
    xAxis100 = (temperatures- Tc_range[2]) * (100 ** (1/v))
    xAxis150 = (temperatures - Tc_range[2]) * (150 ** (1/v))
    xAxis200 = (temperatures - Tc_range[2]) * (200 ** (1/v))
    xAxis250 = (temperatures - Tc_range[2]) * (250 ** (1/v))

    xAxis_s = [xAxis50, xAxis100, xAxis150, xAxis200, xAxis250]

    # Plottaggio
    plt.figure()
    plt.scatter(xAxis50, binder_cumulant_N50, label='N = 50')
    plt.scatter(xAxis100, binder_cumulant_N100, label='N = 100')
    plt.scatter(xAxis150, binder_cumulant_N150, label='N = 150')
    plt.scatter(xAxis200, binder_cumulant_N200, label='N = 200')
    plt.scatter(xAxis250, binder_cumulant_N250, label='N = 250')

    plt.xlim(-20, 20)
    plt.legend()
    plt.title(r'$\bar{\nu} = $' + str(v))
    plt.xlabel(r'$(T-T_c)N^{1/\bar{\nu}}$')
    plt.ylabel('Binder cumulant')
    plt.show()



