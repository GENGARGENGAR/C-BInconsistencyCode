import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy
import Evolution3

sample = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
for i in sample:
    states = np.zeros(6).reshape(1,6)   
    for j in range(100): 
        regular_network = nx.random_regular_graph(3, 400)
        population = np.random.randint(0,6,400)
        struc = []
        for row_index in range(400):
            start_index = nx.adjacency_matrix(regular_network).indptr[row_index]
            end_index = nx.adjacency_matrix(regular_network).indptr[row_index + 1]
            non_zero_indices = nx.adjacency_matrix(regular_network).indices[start_index:end_index]
            struc.append(non_zero_indices)
        DB_regular = Evolution3.Evolution(population, struc, 0.1, np.array([3,-1,4,0,i,0.8]), 0.002)
        DB_regular.DB_evolve(1000000)
        with open('data/opinion=3_'+'e1='+str(i)+'.txt', 'ab') as f:
            np.savetxt(f, np.average(DB_regular.trace, axis=0).reshape(1,6))