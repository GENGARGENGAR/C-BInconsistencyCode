import Evolution_5_opinions
import numpy as np
import networkx as nx

sample = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
for i in sample:
    states = np.zeros(10).reshape(1,10)
    for j in range(20):
        struc = nx.random_regular_graph(3, 400)
        population = np.random.randint(0,10,400)
        DB_ragular = Evolution_5_opinions(population, nx.adjacency_matrix(struc).indices.reshape(400,3), 0.1, np.array([3,-1,4,0,i,0.8]), 0.01)
        Evolution_5_opinions.DB_evolve(DB_ragular,1000000)
        print(np.average(DB_ragular.trace, axis=0))
        states = np.append(states, np.average(DB_ragular.trace, axis=0).reshape(1,10), axis=0)
    with open('opinion=5_'+'e1='+str(i)+'.txt', 'ab') as f:
        np.savetxt(f, np.delete(states, 0, axis=0))