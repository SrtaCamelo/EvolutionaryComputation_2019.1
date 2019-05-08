#Projeto 01, Evolutinary Computing
#Evolutionary Strategy Code Adaptated - Cauchy + Crossover Half Parents
#Srta Camelo


import fitness as ft
import numpy as np
import random
import math
#from operator import itemgetter

"""
REFERENCIA: Estratégias Evolutivas Aplicadas à Resoluçãode Otimização Multimodal
"""

#--------Evolutionary Strategy - Cauchy------------

def concatenate(bool1, bool2):
    bool = []
    for i in bool1:
        bool.append(i)
    for i in bool2:
        bool.append(i)
    return bool
"""
Efetuates Crossover between mother and father.
Parameters: 
    parents: mother and father
Gets half of mother gene and half of father's gene, radomicaly
"""
def crossover(father,mother):
    half = int(len(father)/2)
    trues = np.ones(half)
    falses = np.zeros(half)
    bool_array = concatenate(trues,falses)
    random.shuffle(bool_array)

    new_born = []
    for i in range(len(father)):
        if bool_array[i]:
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
def probability_sum(ft_instance,ft_all):
    sum_all_ft = np.sum(ft_all)
    prob = 1 - (ft_instance/sum_all_ft)
    return prob
def es_c(population,x_train, y_train,x_validate, y_validate, x_test, y_test):
    mut = 0.3
    cross = 0.2


    best_fit = ft.find_best(population,x_train, y_train, x_validate, y_validate)
    num_generations = 50
    decendants = []

    for i in range(num_generations):
        fitness_all, models_all = ft.calculate_pop_ft(population, x_train, y_train, x_validate, y_validate)
        number_decendants = 0
        decendants = []
        while number_decendants < (len(population)+5):
            #print(population)
            check = 1
            while check:
                father_idx = np.random.random_integers(0,len(population)-1)

                father = population[father_idx]


                father_ft = fitness_all[father_idx]
                radom_gauss = random.gauss(0, 1)
                father_prob = probability_sum(father_ft,fitness_all)
                if radom_gauss < father_prob:
                    check = 0
            check2 = 1
            while check2:
                mother_idx = np.random.random_integers(0, len(population) - 1)

                mother = population[mother_idx]

                mother_ft = fitness_all[mother_idx]
                radom_gauss = random.gauss(0, 1)
                mother_prob = probability_sum(mother_ft, fitness_all)
                if radom_gauss < mother_prob:
                    check2 = 0

            crossed = crossover(father,mother,cross)
            mutated = mutation(crossed,mut)
            #print(crossed)
            decendants.append(mutated)

            number_decendants += 1
        ft_decendants,model_decendants = ft.calculate_pop_ft(decendants, x_train, y_train,x_validate,y_validate)

        new_population,ft_new_pop,model_new = ft.fetch_n_better(decendants,ft_decendants,model_decendants,len(population))
        ft_best = (new_population[0],(ft_new_pop[0],model_new[0]))
        population = new_population
        if ft_best[1][0] > best_fit[1][0]:
            best_fit = ft_best
    #
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