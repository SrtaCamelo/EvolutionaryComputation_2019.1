#Projeto 01, Evolutinary Computing
#Evolutionary Strategy Code Adaptated - Cauchy (Normal Crossover)
#Srta Camelo


import fitness as ft
import numpy as np
import random
import math
#from operator import itemgetter

"""
REFERENCIA: Estratégias Evolutivas Aplicadas à Resoluçãode Otimização Multimoda
"""

#--------Evolutionary Strategy------------

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

    r1 = (math.sqrt(((2*len(hemafrodite)))))**-1
    r2 = (math.sqrt(math.sqrt(((4*len(hemafrodite))))))**-1
    #print(r1, r2)
    for gene in hemafrodite:
        #print(np.random.standard_cauchy(1))
        o = mut * np.exp((r1*random.gauss(0, 1))+ (r2*random.gauss(0, 1)))
        new_gene = gene + (o * np.random.standard_cauchy(1))

        if new_gene > 0:
            gene = new_gene

        gene = float(gene)
        new_mutation.append(gene)
    return new_mutation

def fetch_n_better(decendants, ft_decendants, n):
    sorted_pop = [x for _, x in sorted(zip(ft_decendants, decendants),reverse=True,key=lambda x: x[0])]
    fitness_sorted = [y for y, x in sorted(zip(ft_decendants, decendants),reverse=True,key=lambda x: x[0])]
    new_population = sorted_pop[0:n]
    best = (sorted_pop[0],(fitness_sorted[0]))
    return new_population, best

def es(population,x_train, y_train,x_validate, y_validate, x_test, y_test):
    mut = 0.3
    cross = 0.2
    #fitness_all = ft.calculate_pop_ft(population, x_train, y_train, x_test, y_test)

    best_fit = ft.find_best(population,x_train, y_train, x_validate, y_validate)
    num_generations = 50
    decendants = []

    for i in range(num_generations):
        number_decendants = 0
        decendants = []
        while number_decendants < (len(population)+5):
            #print(population)
            father = np.random.random_integers(0,len(population)-1)
            mother = np.random.random_integers(0,len(population)-1)
            mother2 = np.random.random_integers(0,len(population)-1)


            father = population[father]
            mother = population[mother]
            mother2 = population[mother2]

            monft = ft.fitness(mother, x_train, y_train, x_validate, y_validate)
            monft2 = ft.fitness(mother2, x_train, y_train, x_validate, y_validate)

            if monft[0] < monft2[0]:
                mother = mother2

            crossed = crossover(father,mother,cross)
            mutated = mutation(crossed,mut)
            #print(crossed)
            decendants.append(mutated)

            number_decendants += 1
        ft_decendants,model_decendants = ft.calculate_pop_ft(decendants, x_train, y_train, x_validate, y_validate)

        new_population,ft_new_pop,model_new = ft.fetch_n_better(decendants,ft_decendants,model_decendants,len(population))
        ft_best = (new_population[0],(ft_new_pop[0],model_new[0]))
        population = new_population
        if ft_best[1][0] > best_fit[1][0]:
            best_fit = ft_best
    ee_accu = ft.fitness_best(best_fit[0], best_fit[1][1],x_test,y_test)
    return ee_accu

"""
#Test
pop_size = (5,3)
population = np.random.uniform(low=0, high=1.0, size=pop_size)
#print("population")
#print(population)
acc = es(population,0,0,0,0,0,0)
print(acc)
"""