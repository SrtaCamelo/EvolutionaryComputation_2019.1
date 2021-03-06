#Projeto 01, Evolutinary Computing
#Differencial Evolution Code
#Srta Camelo

import fitness as ft
import numpy as np
import random
import math

"""
REFERENCIA: http://www.cleveralgorithms.com/nature-inspired/evolution/evolutionary_programming.html#
"Fast Evolutionary Programming"
"""
#----------- Evolutive Programming -----------------
"""
Mutates a random gene from the offspring
changing it to a random value between -1 and 1
"""
def mutation2(hemafrodite):

    gene = np.random.random_integers(0, len(hemafrodite)-1)
    random_value = np.random.uniform(0, 1.0, 1)

    new_mutation = hemafrodite
    new_mutation[gene] = random_value
    return new_mutation

def mutation(hemafrodite,mut):
    new_mutation = []

    r1 = (math.sqrt(((2*len(hemafrodite)))))**-1
    r2 = (math.sqrt(math.sqrt(((4*len(hemafrodite))))))**-1
    #print(r1, r2)
    for gene in hemafrodite:
        o = mut * np.exp((r1*random.gauss(0, 1))+ (r2*random.gauss(0, 1)))
        new_gene = gene + (o * random.gauss(0, 1))
        if new_gene > 0:
            gene = new_gene
        new_mutation.append(gene)
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
def best(best_child,best_child_ft,best_child_model,best_solution,best_solution_ft,best_solution_model):
    if best_child_ft > best_solution_ft:
        best_ft = best_child_ft
        best = best_child
        best_model = best_child_model
    else:
        best_ft = best_solution_ft
        best = best_solution
        best_model = best_solution_model
    return best, best_ft,best_model

def ep(population,x_train, y_train,x_validate, y_validate, x_test, y_test):
    mut = 0.3
    num_genertions = 50
    all_fitness,all_models = ft.calculate_pop_ft(population, x_train, y_train, x_validate, y_validate)
    max_fitness_idx = np.where(all_fitness == np.max(all_fitness))
    max_fitness_idx = max_fitness_idx[0][0]

    best_solution = population[max_fitness_idx]
    best_solution_ft = all_fitness[max_fitness_idx]
    best_solution_model = all_models[max_fitness_idx]
    boutsize = 3
    si_wins = 0

    for i in range(num_genertions):
        children = []
        for parent in population:
            child = mutation(parent,mut)
            children.append(child)


        children_ft, children_models = ft.calculate_pop_ft(children, x_train, y_train, x_validate, y_validate)
        max_fitness_idx = np.where(all_fitness == np.max(all_fitness))
        max_fitness_idx = max_fitness_idx[0][0]

        best_child = population[max_fitness_idx]
        best_child_ft = all_fitness[max_fitness_idx]
        best_child_model = all_fitness[max_fitness_idx]

        best_solution, best_solution_ft, best_solution_model = best(best_child,best_child_ft,best_child_model,best_solution,best_solution_ft,best_solution_model)

        union = np.concatenate((population,children))
        union = union.tolist()
        fitness = np.concatenate((all_fitness,children_ft))
        fitness = fitness.tolist()
        #models = np.concatenate(all_models,children_models)
        #models = models.tolist()
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
    best_accu = ft.fitness_best(best_solution,best_solution_model,x_test,y_test)
    #print(best_accu)
    return best_accu

"""
pop_size = (5,3)
population = np.random.uniform(low=-1.0, high=1.0, size=pop_size)
#print("population")
#print(population)
accu = ep(population,0,0,0,0)
print(accu)
"""