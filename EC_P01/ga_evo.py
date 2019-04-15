#Projeto 01, Evolutinary Computing
#Genetic Algoritm Code
#Srta Camelo

#---------- Imports ----------
import fitness as ft
import numpy as np

#----------- Genetic Algoritm -----------------
"""
Efetuates Crossover between mother and father.
Parameters: 
    parents: mother and father
    num: how many genes to get from father

"""
def crossover(parents,num):
    father, mother = parents[0], parents[1]
    new_born = father[0:num]
    new_born = np.concatenate((new_born, mother[num: len(mother)]))
    return new_born
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

def mating_pool(population, all_fitness,number_parents):
    parents = []
    for i in range(number_parents):
        max_fitness_idx = np.where(all_fitness == np.max(all_fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents.append(population[max_fitness_idx])

        all_fitness[max_fitness_idx] = -99999999999

    return parents


def ga(population,x_train, y_train, x_test, y_test):

    new_population = []
    parents_mated = []
    num_generations = 5
    for i in range(num_generations):
        fitness_all = ft.calculate_pop_ft(population,x_train, y_train, x_test, y_test)
        for j in range(int((len(population)/2))):
            parents = mating_pool(population,fitness_all,2)
            offspring = crossover(parents,3)
            offspring = mutation(offspring)

            new_population.append(offspring)

            parents_mated.append(parents[0])
            parents_mated.append(parents[1])

        new_population = np.concatenate((new_population,parents_mated[:int((len(population)/2))]))
        population = new_population

        #------- Clean aux Lists------
        parents_mated = []
        new_population = []

    best = ft.find_best(population,x_train, y_train, x_test, y_test)

    return best


