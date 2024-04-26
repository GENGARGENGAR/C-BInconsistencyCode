import Evolution
import numpy as np
import networkx as nx

sample = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
for i in sample:
    states = np.zeros(4).reshape(1,4)
    for j in range(20):
        struc = nx.random_regular_graph(3, 400)
        population = np.random.randint(0,4,400)
        DB_regular = Evolution.Evolution(population, nx.adjacency_matrix(struc).indices.reshape(400,3), 0.1, np.array([3,-1,4,0,i,0.8]), 0.01)
        DB_regular.DB_evolve(DB_regular,200000)
        with open('0.1'+str(i)+'.txt', 'ab') as f:
            np.savetxt(f, np.average(star.trace, axis=0).reshape(1,4))