#Projeto 01, Evolutinary Computing
#Genetic Algoritm Code
#Srta Camelo

#---------- Imports ----------
import fitness as ft
import numpy as np
import math
import random

"""
REFERENCIA: Computational Methods of Feature Selection

"""
#----------- Genetic Algoritm -----------------
"""
Efetuates Crossover between mother and father.
Parameters: 
    parents: mother and father
    num: how many genes to get from father

"""
def crossover(father,mother,cross):
    crossing = np.random.uniform(low=0, high=1.0, size=(len(father)))
    crossing = crossing > cross
    new_born = []
    for i in range(len(father)):
        if crossing[i]:
            new_born.append(father[i])
        else:
            new_born.append(mother[i])

    return new_born

"""
Mutates a random gene from the offspring
changing it to a random value between -1 and 1
"""
def mutation(hemafrodite,mut):
    new_mutation = []

    r1 = (math.sqrt(((2 * len(hemafrodite))))) ** -1
    r2 = (math.sqrt(math.sqrt(((4*len(hemafrodite))))))**-1
    #print(r1, r2)
    for gene in hemafrodite:
        o = mut * np.exp((r1*random.gauss(0, 1))+ (r2*random.gauss(0, 1)))
        new_gene = gene + (o * random.gauss(0, 1))
        if new_gene > 0:
            gene = new_gene
        new_mutation.append(gene)
    return new_mutation

"""
Roleta de seleção, seleciona os pais de acordo com a sua probabilidade (ftness /fitness de todos
Parametros: fitness -> Lista com todos os fitness de todos os integrantes da população
            k -> Quantos individuos devem ser selecionados
            nPop -> Tamanho da população
"""
def roulette_wheel_selection(fitness, k, nPop):
    total_fit = float(sum(fitness))
    relative_fitness = [f/total_fit for f in fitness]
    prob = [sum(relative_fitness[:i+1])
            for i in range(len(relative_fitness))]

    chosen = []
    for n in range(k):
        r = np.random.uniform(low=0, high=1.0)
        for i in range(nPop):
            if r <= prob[i]:
                chosen.append(i)
                break
    return chosen

def mating_pool(population, all_fitness,number_parents):
    parents = []
    for i in range(number_parents):
        max_fitness_idx = np.where(all_fitness == np.max(all_fitness))
        #print(all_fitness)
        max_fitness_idx = max_fitness_idx[0][0]

        parents.append(population[max_fitness_idx])

        all_fitness[max_fitness_idx] = -99999999999

    return parents


def ga(population,x_train, y_train,x_validate, y_validate, x_test, y_test):
    mut = 0.3
    cross = 0.2
    best = ft.find_best(population, x_train, y_train, x_validate, y_validate)
    #treshold = 0.5
    new_population = []
    #parents_mated = []
    num_generations = 50
    for k in range(num_generations):
        fitness_all, models_all = ft.calculate_pop_ft(population,x_train, y_train,x_validate, y_validate)

        for j in range(int((len(population)/2))):
            parents =roulette_wheel_selection(fitness_all,2,len(population))
            parents = (population[parents[0]],population[parents[1]])
            offspring = crossover(parents[0],parents[1],cross)
            offspring2 = crossover(parents[0],parents[1],cross)
            offspring = mutation(offspring,mut)
            offspring2 = mutation(offspring2,mut)

            new_population.append(offspring)
            new_population.append(offspring2)

            #parents_mated.append(parents[0])
            #parents_mated.append(parents[1])

        #new_population = np.concatenate((new_population,parents_mated[:int((len(population)/2))]))
        population = new_population
        gen_best = ft.find_best(population, x_train, y_train, x_validate, y_validate)
        if gen_best[1][0] > best[1][0]:
            best = gen_best

        #------- Clean aux Lists------
        #parents_mated = []
        new_population = []

    #best = ft.find_best(population,x_train, y_train, x_test, y_test)
    best = ft.fitness_best(best[0],best[1][1],x_test,y_test)

    return best


