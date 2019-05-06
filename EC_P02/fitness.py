#Projeto 01, Evolutinary Computing
#Neural Fitness Calculation Code
#Srta Camelo

#-------------Imports---------
from operator import itemgetter

import random_forest as rf
import numpy as np


def best_model(oldmodel, newmodel):
    if newmodel[0] > oldmodel[0]:
        best = newmodel
    else:
        best = oldmodel
    return best

"""
Evolutionary Fitness function calculates accuracy as in neural Network as fitness
Parametes: 
    pesos: Weights from the evolutionary algoritm
    x_train, y_train, x_test, y_test: splited Data Set to use in neural network
Returns: Accuracy of the Neural Network using given data and given weights
"""


def fitness(features,x_train, y_train,x_validate, y_validate):
    #e1 = np.random.random()
    #e2 = np.random.random()
    #evaluation = (e1,e2)
    #print(features)
    features = np.array(features)
    features = features > 0.5
    x_train = x_train.loc[:, features]
    x_validate = x_validate.loc[:, features]

    evaluation, model = rf.rf(x_train, y_train,x_validate, y_validate)
    #print(evaluation)
    evaluation = (evaluation, model)

    return evaluation

def fitness_best(features, clf,x_test,y_test):
    features = np.array(features)
    features = features > 0.5
    x_test = x_test.loc[:, features]
    evaluation = rf.rf_best(clf,x_test,y_test)
    #print(evaluation)
    return evaluation

"""
Calculates fitness for all population (each individual)
"""
def fetch_n_better(decendants, ft_decendants,model_decendants, n):
    ft_sorted = []
    #print(ft_decendants)
    indexes = sorted(range(len(ft_decendants)), key=ft_decendants.__getitem__)
    ft_sorted = list(itemgetter(*indexes)(ft_decendants))
    sorted_pop = list(itemgetter(*indexes)(decendants))
    model_sorted = list(itemgetter(*indexes)(model_decendants))

    #ft_sorted,sorted_pop,model_sorted = zip(*sorted(zip(ft_decendants,model_decendants)))

    ft_sorted = list(ft_sorted)
    sorted_pop = list(sorted_pop)
    model_sorted = list(model_sorted)
    #print(ft_sorted)
    new_pop = sorted_pop[-n:]
    ft_new = ft_sorted[-n:]
    model_new = model_sorted[-n:]
    new_pop = new_pop[::-1]
    ft_new = ft_new[::-1]
    model_new = model_new[::-1]

    #print("NEW FT")
    #print(model_new)

    #sorted_pop = [x for _, x in sorted(zip(ft_decendants, decendants),reverse=True,key=lambda x: x[0])]
    #fitness_sorted = [y for y, x in sorted(zip(ft_decendants, decendants),reverse=True,key=lambda x: x[0])]
    #new_population = sorted_pop[0:n]
    #best = (sorted_pop[0],(fitness_sorted[0]))
    #return new_population, best
    return new_pop,ft_new,model_new

def calculate_pop_ft(population,x_train, y_train,x_validate, y_validate):

    all_fitness = []
    all_models = []
    for cromossome in population:
        ft, model = fitness(cromossome,x_train, y_train, x_validate, y_validate)
        all_fitness.append(ft)
        all_models.append(model)
    return all_fitness, all_models

def find_best(population,x_train, y_train, x_validate, y_validate):

    all_fitness, all_models = calculate_pop_ft(population, x_train, y_train, x_validate, y_validate)
    #Retornar maior acurácia
    best_pop, best_ft, best_models = fetch_n_better(population,all_fitness,all_models,len(population))
    best = (best_pop[0],(best_ft[0],best_models[0]))
    """"
    Retornar o melhor conjunto de pesos
    max_fitness_idx = np.where(all_fitness == np.max(all_fitness))
    max_fitness_idx = max_fitness_idx[0][0]
    best = population[max_fitness_idx, :]
    """
    return best