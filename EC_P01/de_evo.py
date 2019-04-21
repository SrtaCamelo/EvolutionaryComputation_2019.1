#Projeto 01, Evolutinary Computing
#Differencial Evolution Code
#Srta Camelo

import fitness as ft
import numpy as np

"""
REFERENCIA: https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/#
"""
#----------- Evolutive Programming -----------------
def de(population,x_train, y_train, x_test, y_test):
    fitness_all = ft.calculate_pop_ft(population, x_train, y_train, x_test, y_test)
    weight_dimention = (2005,2)
    mut_constant = 0.5
    limiar_crossover = 0.6
    best_idx = 0
    for i in range(len(population)):  #Iterate over all cromossomes
        indices = [indices for indices in range(len(population)) if indices != i]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + mut_constant * (b - c), -1, 1)
        cross_points = np.random.uniform(low = 0, high= 1,size = weight_dimention) < limiar_crossover
        trial = np.where(cross_points, mutant, population[i])
        trial_ft = ft.fitness(trial,x_train, y_train, x_test, y_test)
        if trial_ft > fitness_all[i]:
            fitness_all[i] = trial_ft
            population[i] = trial_ft
            if trial_ft > fitness_all[best_idx]:
                best_idx = i
                #best_weight = trial
    return fitness_all[best_idx]


"""
pop_size = (5,3)
population = np.random.uniform(low=-1.0, high=1.0, size=pop_size)
#print("population")
#print(population)
accu = ep(population,0,0,0,0)
print(accu)
"""