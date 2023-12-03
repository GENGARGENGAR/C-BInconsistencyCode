import Evolution
import numpy as np

states = np.zeros(4).reshape(1,4)
struc = np.array([[0,1,0,0,0,0],[1,0,1,1,0,0],[0,1,0,1,0,0],[0,1,1,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0]])
sample = np.array([1.4, 1.6])
for i in sample:
    states = np.zeros(4).reshape(1,4)
    for j in range(50):
        population = np.random.randint(0,4,6)
        star = Evolution(population, struc, 0.01, np.array([11,-1,12,0,i,0.8]), 0.01)
        Evolution.DB_evolve(star,1000000)
        print(np.average(star.trace, axis=0))
        states = np.append(states, np.average(star.trace, axis=0).reshape(1,4), axis=0)
    with open('threshold=18'+'e1='+str(i)+'.txt', 'ab') as f:
        np.savetxt(f, np.delete(states, 0, axis=0))