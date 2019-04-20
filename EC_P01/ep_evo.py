#Projeto 01, Evolutinary Computing
#Differencial Evolution Code
#Srta Camelo

import fitness as ft
import numpy as np

"""
REFERENCIA: https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/#
"""
#----------- Evolutive Programming -----------------
def ep(population,x_train, y_train, x_test, y_test):
    # fitness_all = ft.calculate_pop_ft(population, x_train, y_train, x_test, y_test)
    num_generations = 3
    mut_constant = 0.5
    limiar_crossover = 0.6
    for i in range(len(population)):  #Iterate over all cromossomes
        indices = [indices for indices in range(len(population)) if indices != i]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + mut_constant * (b - c), -1, 1)
        cross_points = np.random.rand(len(population[0])) < limiar_crossover

        print(cross_points)
        trial = np.where(cross_points, mutant, population[i])
        print(mutant)
        print(population[i])
        print(trial)

pop_size = (5,3)
population = np.random.uniform(low=-1.0, high=1.0, size=pop_size)
#print("population")
#print(population)
ep(population,0,0,0,0)