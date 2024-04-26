import numpy as np
import networkx as nx
import scipy
import Evolution


population = np.random.randint(0,4,34)
karate = np.loadtxt('karate_data.txt', dtype=int)
struc_raw = np.empty((34,34))
for i in range(34):
    for j in range(34):
        struc_raw[i][j]=0
for i in range(76):
    struc_raw[karate[i][0]-1][karate[i][1]-1]=1
    struc_raw[karate[i][1]-1][karate[i][0]-1]=1
struc = []
for i in range(34):
    struc.append(np.empty(0, dtype=int))
for i in range(34):
    for j in range(34):
        if struc_raw[i][j]==1:
            struc[i]=np.append(struc[i],j)
sample = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.5, 1]
for i in sample:
    for j in range(100):
        DB_karate = Evolution.Evolution(population, struc, 0.1, np.array([6,-1,7,0,0.2,0.6]), i)
        DB_karate.DB_evolve(200000)
        with open('s_karate_w'+str(i)+'.txt', 'ab') as f:
            np.savetxt(f, np.average(DB_karate.trace, axis=0).reshape(1,4))