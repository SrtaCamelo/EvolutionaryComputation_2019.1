#Projeto 01, Evolutinary Computing
#Differencial Evolution Code
#Srta Camelo

import fitness as ft
import numpy as np
import random

"""
REFERENCIA: http://www.cleveralgorithms.com/nature-inspired/evolution/evolutionary_programming.html#
"""
#----------- Evolutive Programming -----------------
"""
Mutates a random gene from the offspring
changing it to a random value between -1 and 1
"""
def mutation(hemafrodite):

    gene = np.random.random_integers(0, len(hemafrodite)-1)
    random_value = np.random.uniform(-1.0, 1.0, 1)

    new_mutation = hemafrodite
    new_mutation[gene] = random_value
    return new_mutation

def selectBest(union,populationSize,wins_idx):
    union = np.array(union)
    best = union[wins_idx]
    best = best[0:populationSize]

    return best
"""
Function compares two cromossomes to check which has these has best fitness, returns the cromossome and the fitness
Param: 
"""
def best(best_child,best_child_ft,best_solution,best_solution_ft):
    if best_child_ft > best_solution_ft:
        best_ft = best_child_ft
        best = best_child
    else:
        best_ft = best_solution_ft
        best = best_solution
    return best, best_ft

def ep(population,x_train, y_train, x_test, y_test):
    num_genertions = 3
    all_fitness = ft.calculate_pop_ft(population, x_train, y_train, x_test, y_test)
    max_fitness_idx = np.where(all_fitness == np.max(all_fitness))
    max_fitness_idx = max_fitness_idx[0][0]

    best_solution = population[max_fitness_idx]
    best_solution_ft = all_fitness[max_fitness_idx]
    boutsize = 3
    si_wins = 0

    for i in range(num_genertions):
        children = []
        for parent in population:
            child = mutation(parent)
            children.append(child)

        children_ft = ft.calculate_pop_ft(children, x_train, y_train, x_test, y_test)
        max_fitness_idx = np.where(all_fitness == np.max(all_fitness))
        max_fitness_idx = max_fitness_idx[0][0]

        best_child = population[max_fitness_idx]
        best_child_ft = all_fitness[max_fitness_idx]

        best_solution, best_solution_ft = best(best_child,best_child_ft,best_solution,best_solution_ft)

        union = np.concatenate((population,children))
        union = union.tolist()
        fitness = np.concatenate((all_fitness,children_ft))
        fitness = fitness.tolist()
        wins_idx = []
        for si in union:
            idx_si = union.index(si)
            for l in range(boutsize):
                sj = random.choice(union)
                idx_sj = union.index(sj)
                if(fitness[idx_si] > fitness[idx_sj]):
                    si_wins += 1
                else:
                    actual_idx_sj = idx_sj
            if si_wins > 1:
                wins_idx.append(idx_si)
            else:
                wins_idx.append(actual_idx_sj)
        population = selectBest(union,len(population),wins_idx)
    return best_solution_ft

"""
pop_size = (5,3)
population = np.random.uniform(low=-1.0, high=1.0, size=pop_size)
#print("population")
#print(population)
accu = ep(population,0,0,0,0)
print(accu)
"""