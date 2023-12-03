import Evolution
import numpy as np
import networkx as nx

sample = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
for i in sample:
    states = np.zeros(4).reshape(1,4)
    for j in range(20):
        struc = nx.random_regular_graph(3, 400)
        population = np.random.randint(0,4,400)
        DB_ragular = Evolution(population, nx.adjacency_matrix(struc).indices.reshape(400,3), 0.3, np.array([3,-1,4,0,i,0.8]), 0.01)
        Evolution.DB_evolve(DB_ragular,200000)
        print(np.average(DB_ragular.trace, axis=0))
        states = np.append(states, np.average(DB_ragular.trace, axis=0).reshape(1,4), axis=0)
    with open('0.3'+str(i)+'.txt', 'ab') as f:
        np.savetxt(f, np.delete(states, 0, axis=0))