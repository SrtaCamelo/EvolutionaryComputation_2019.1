#Projeto 01, Evolutinary Computing
#Neural Fitness Calculation Code
#Srta Camelo

#-------------Imports---------
import neural_network as nn
import numpy as np

"""
Evolutionary Fitness function calculates accuracy as in neural Network as fitness
Parametes: 
    pesos: Weights from the evolutionary algoritm
    x_train, y_train, x_test, y_test: splited Data Set to use in neural network
Returns: Accuracy of the Neural Network using given data and given weights
"""


def fitness(pesos,x_train, y_train, x_test, y_test):
    model = nn.neural_network(pesos)
    evaluation = nn.use_network(model,x_train, y_train, x_test, y_test)
    """
    for i in pesos:
        evaluation += i
    #net = nn.neural_network(pesos)
    #evaluation = nn.use_network(net, x_train, y_train, x_test, y_test)
    """
    return evaluation


"""
Calculates fitness for all population (each individual)
"""


def calculate_pop_ft(population,x_train, y_train, x_test, y_test):

    all_fitness = []
    for cromossome in population:
        ft = fitness(cromossome,x_train, y_train, x_test, y_test)
        all_fitness.append(ft)
    return all_fitness

def find_best(population,x_train, y_train, x_test, y_test):

    all_fitness = calculate_pop_ft(population, x_train, y_train, x_test, y_test)
    #Retornar maior acurácia
    best = max(all_fitness)
    """"
    Retornar o melhor conjunto de pesos
    max_fitness_idx = np.where(all_fitness == np.max(all_fitness))
    max_fitness_idx = max_fitness_idx[0][0]
    best = population[max_fitness_idx, :]
    """
    return best