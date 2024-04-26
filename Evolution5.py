#!/usr/bin/env python
# coding: utf-8

import numpy as np

class Evolution:
    def __init__(self, population: np.ndarray, struc, mutation: float, payoffs: np.ndarray, intensity: float):
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
        self.payoff_matrix = np.array([[R, R, R, R, R, S, S, S, S, S], [R-epsilon_1/4, R-epsilon_1/4, R-epsilon_1/4, R-epsilon_1/4, R-epsilon_1/4, S-epsilon_1/4, S-epsilon_1/4, S-epsilon_1/4, S-epsilon_1/4, S-epsilon_1/4], [R-2*epsilon_1/4, R-2*epsilon_1/4, R-2*epsilon_1/4, R-2*epsilon_1/4, R-2*epsilon_1/4, S-2*epsilon_1/4, S-2*epsilon_1/4, S-2*epsilon_1/4, S-2*epsilon_1/4, S-2*epsilon_1/4], [R-3*epsilon_1/4, R-3*epsilon_1/4, R-3*epsilon_1/4, R-3*epsilon_1/4, R-3*epsilon_1/4, S-3*epsilon_1/4, S-3*epsilon_1/4, S-3*epsilon_1/4, S-3*epsilon_1/4, S-3*epsilon_1/4], [R-epsilon_1, R-epsilon_1, R-epsilon_1, R-epsilon_1, R-epsilon_1, S-epsilon_1, S-epsilon_1, S-epsilon_1, S-epsilon_1, S-epsilon_1], [T-epsilon_2, T-epsilon_2, T-epsilon_2, T-epsilon_2, T-epsilon_2, P-epsilon_2, P-epsilon_2, P-epsilon_2, P-epsilon_2, P-epsilon_2], [T-3*epsilon_2/4, T-3*epsilon_2/4, T-3*epsilon_2/4, T-3*epsilon_2/4, T-3*epsilon_2/4, P-3*epsilon_2/4, P-3*epsilon_2/4, P-3*epsilon_2/4, P-3*epsilon_2/4, P-3*epsilon_2/4], [T-2*epsilon_2/4, T-2*epsilon_2/4, T-2*epsilon_2/4, T-2*epsilon_2/4, T-2*epsilon_2/4, P-2*epsilon_2/4, P-2*epsilon_2/4, P-2*epsilon_2/4, P-2*epsilon_2/4, P-2*epsilon_2/4], [T-epsilon_2/4, T-epsilon_2/4, T-epsilon_2/4, T-epsilon_2/4, T-epsilon_2/4, P-epsilon_2/4, P-epsilon_2/4, P-epsilon_2/4, P-epsilon_2/4, P-epsilon_2/4],  [T, T, T, T, T, P, P, P, P, P]])
        self.selection_intensity = intensity
        self.fitness = np.zeros(self.pop_size)
        self.evolution_state = np.array([np.count_nonzero(self.population==i) for i in range(10)]).reshape(1,10)/self.pop_size
        self.trace = np.empty(0).reshape(0,10)
        self.average_state = np.zeros(shape=(1,10)).reshape(1,10)
        self.save_name = save_name

    def cal_fitness(self):
        payoff = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            payoff[i] = self.payoff_matrix[self.population[i]][self.population[self.population_structure[i]]].sum()
        self.fitness = 1-self.selection_intensity*np.ones(self.pop_size)+self.selection_intensity*payoff
        
    def revise_fitness(self, replaced):
        self.fitness[replaced] = 0
        self.fitness[replaced] = self.payoff_matrix[self.population[replaced]][self.population[self.population_structure[replaced]]].sum()
        self.fitness[replaced] = (1-self.selection_intensity*np.ones(self.pop_size)+self.selection_intensity*self.fitness)[replaced]
        for neighbers in self.population_structure[replaced]:
            self.fitness[neighbers] = 0
            self.fitness[neighbers] = self.payoff_matrix[self.population[neighbers]][self.population[self.population_structure[neighbers]]].sum()
            self.fitness[neighbers] = (1-self.selection_intensity*np.ones(self.pop_size)+self.selection_intensity*self.fitness)[neighbers]

    def death(self):
        death_individual = np.random.randint(self.pop_size)
        return death_individual
    
    def birth(self, death_individual):
        neighbers = self.population_structure[death_individual]
        neigh_fitness = self.fitness[neighbers]
        if neigh_fitness.sum(-1, keepdims=True)==np.zeros(1):
            selected_neigh = neighbers[0]
        else:
            probs = neigh_fitness / neigh_fitness.sum(-1, keepdims=True)
            selected_neigh = np.random.choice(neighbers, size=1, p=probs, replace=False)
        self.population[death_individual] = self.population[selected_neigh]

    def mutation(self, death_individual):
        rand = np.random.rand()
        if rand < self.mutation_rate:
            rand_mutation = np.random.randint(10)
            self.population[death_individual] = rand_mutation
        else:
            self.birth(death_individual)
                
    def DB_evolve(self, step: int):
        self.trace = self.evolution_state
        self.cal_fitness
        for i in range(step):
            death_individual = self.death()
            self.mutation(death_individual)
            self.revise_fitness(death_individual)
            self.evolution_state = np.array([np.count_nonzero(self.population==i) for i in range(10)]).reshape(1,10)/self.pop_size
            self.trace = np.append(self.trace, self.evolution_state, axis=0)