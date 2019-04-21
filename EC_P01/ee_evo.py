#Projeto 01, Evolutinary Computing
#Evolutionary Strategy Code
#Srta Camelo


import fitness as ft
import numpy as np
#from operator import itemgetter

"""
REFERENCIA: 
"""

#--------Evolutionary Strategy------------

"""
Efetuates Crossover between mother and father.
Parameters: 
    parents: mother and father
    num: how many genes to get from father

"""
def crossover(father,mother,num):
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
    #print(random_value)

    new_mutation = hemafrodite
    new_mutation[gene] = random_value
    return new_mutation

def fetch_n_better(decendants, ft_decendants, n):
    sorted_pop = []
    sorted_pop= [x for _, x in sorted(zip(ft_decendants, decendants),reverse=True,key=lambda x: x[0])]
    new_population = sorted_pop[0:n]
    return new_population

def es(population,x_train, y_train, x_test, y_test):
    #fitness_all = ft.calculate_pop_ft(population, x_train, y_train, x_test, y_test)
    num_generations = 3
    decendants = []
    for i in range(num_generations):
        number_decendants = 0

        while number_decendants < (len(population)+5):
            #print(population)
            father = np.random.random_integers(0,len(population)-1)
            mother = np.random.random_integers(0,len(population)-1)

            father = population[father]
            mother = population[mother]


            crossed = crossover(father,mother,1)
            mutated = mutation(crossed)
            #print(crossed)
            decendants.append(mutated)

            number_decendants += 1

        ft_decendants = ft.calculate_pop_ft(decendants, x_train, y_train, x_test, y_test)
        new_population = fetch_n_better(decendants,ft_decendants,len(population))
        population = new_population
    ee_accu = ft.fitness(population[0],x_train, y_train, x_test, y_test)
    return ee_accu
"""
#Test
pop_size = (5,3)
population = np.random.uniform(low=-1.0, high=1.0, size=pop_size)
#print("population")
#print(population)
es(population,0,0,0,0)
"""