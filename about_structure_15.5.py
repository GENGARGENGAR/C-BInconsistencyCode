import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy
import Evolution

states = np.zeros(4).reshape(1,4)
struc = [np.array([1]),np.array([0,2,3,4]),np.array([1,5]),np.array([1,5]),np.array([1,5]),np.array([2,3,4])]
sample = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6])
for i in sample:
    states = np.zeros(4).reshape(1,4)
    for j in range(200):
        population = np.random.randint(0,4,6)
        star = Evolution.Evolution(population, struc, 0.01, np.array([11,-1,12,0,i,0.8]), 0.01)
        star.DB_evolve(1000000)
        with open('threshold=18'+'e1='+str(i)+'.txt', 'ab') as f:
            np.savetxt(f, np.average(star.trace, axis=0).reshape(1,4))