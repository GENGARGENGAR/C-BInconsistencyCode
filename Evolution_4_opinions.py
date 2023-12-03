import numpy as np

class Evolution_4_opinions:
    def __init__(self, population: np.ndarray, struc: np.ndarray, mutation: float, payoffs: np.ndarray, intensity: float):
        R = payoffs[0]
        S = payoffs[1]
        T = payoffs[2]
        P = payoffs[3]
        epsilon_1 = payoffs[4]
        epsilon_2 = payoffs[5]
        self.pop_size = len(struc)
        self.population = population
        self.population_structure = struc
        self.mutation_rate = mutation
        self.payoff_matrix = np.array([[R, R, R, R, S, S, S, S], [R-epsilon_1/3, R-epsilon_1/3, R-epsilon_1/3, R-epsilon_1/3, S-epsilon_1/3, S-epsilon_1/3, S-epsilon_1/3, S-epsilon_1/3], [R-2*epsilon_1/3, R-2*epsilon_1/3, R-2*epsilon_1/3, R-2*epsilon_1/3, S-2*epsilon_1/3, S-2*epsilon_1/3, S-2*epsilon_1/3, S-2*epsilon_1/3], [R-epsilon_1, R-epsilon_1, R-epsilon_1, R-epsilon_1, S-epsilon_1, S-epsilon_1, S-epsilon_1, S-epsilon_1], [T-epsilon_2, T-epsilon_2, T-epsilon_2, T-epsilon_2, P-epsilon_2, P-epsilon_2, P-epsilon_2, P-epsilon_2],[T-2*epsilon_2/3, T-2*epsilon_2/3, T-2*epsilon_2/3, T-2*epsilon_2/3, P-2*epsilon_2/3, P-2*epsilon_2/3, P-2*epsilon_2/3, P-2*epsilon_2/3], [T-epsilon_2/3, T-epsilon_2/3, T-epsilon_2/3, T-epsilon_2/3, P-epsilon_2/3, P-epsilon_2/3, P-epsilon_2/3, P-epsilon_2/3], [T, T, T, T, P, P, P, P]])
        self.selection_intensity = intensity
        self.fitness = np.zeros(self.pop_size)
        self.evolution_state = np.array([np.count_nonzero(self.population==0), np.count_nonzero(self.population==1), np.count_nonzero(self.population==2), np.count_nonzero(self.population==3), np.count_nonzero(self.population==4), np.count_nonzero(self.population==5), np.count_nonzero(self.population==6), np.count_nonzero(self.population==7)]).reshape(1,8)/self.pop_size
        self.trace = np.empty(0).reshape(0,8)
        self.average_state = np.zeros(shape=(1,8)).reshape(1,8)
        self.cal_fitness()

    def cal_fitness(self):
        fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            for j in self.population_structure[i]:
                fitness[i] += self.payoff_matrix[self.population[i]][self.population[j]]
        self.fitness = 1-self.selection_intensity*np.ones(self.pop_size)+self.selection_intensity*fitness

    def DB_evolve(self, step: int):
        self.trace = self.evolution_state
        for i in range(step):
            death_individual = np.random.randint(self.pop_size)
            rand = np.random.rand()
            if rand < self.mutation_rate:
                rand_mutation = np.random.randint(8)
                self.population[death_individual] = rand_mutation
            else:
                neighbers = self.population_structure[death_individual]
                neigh_fitness = self.fitness[neighbers]
                probs = neigh_fitness / neigh_fitness.sum(-1, keepdims=True)
                selected_neigh = np.random.choice(neighbers, size=1, p=probs, replace=False)
                self.population[death_individual] = self.population[selected_neigh]
            self.cal_fitness()
            self.evolution_state = np.array([np.count_nonzero(self.population==0), np.count_nonzero(self.population==1), np.count_nonzero(self.population==2), np.count_nonzero(self.population==3), np.count_nonzero(self.population==4), np.count_nonzero(self.population==5), np.count_nonzero(self.population==6), np.count_nonzero(self.population==7)]).reshape(1,8)/self.pop_size
            self.trace = np.append(self.trace, self.evolution_state, axis=0)