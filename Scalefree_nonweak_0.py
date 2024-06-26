import numpy as np
import networkx as nx
import scipy
import Evolution

sample = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.5, 1]
for i in sample:
    for j in range(100):
        scalefree_network = nx.barabasi_albert_graph(400, 2)
        struc = []
        for row_index in range(400):
            start_index = nx.adjacency_matrix(scalefree_network).indptr[row_index]
            end_index = nx.adjacency_matrix(scalefree_network).indptr[row_index + 1]
            non_zero_indices = nx.adjacency_matrix(scalefree_network).indices[start_index:end_index]
            struc.append(non_zero_indices)
        population = np.random.randint(0,4,400)
        DB_ragular = Evolution.Evolution(population, nx.adjacency_matrix(struc).indices.reshape(400,3), 0.1, np.array([3,-1,4,0,0,0]), i)
        DB_ragular.DB_evolve(200000)
        with open('n_scalefree_w='+str(i)'.txt', 'ab') as f:
            np.savetxt(f, np.average(DB_ragular.trace, axis=0).reshape(1,4))
