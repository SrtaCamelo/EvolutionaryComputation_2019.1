#Projeto 01, Evolutinary Computing
#Group Searsh Otimization Code
#Srta Camelo


#imports

import fitness as ft
import numpy as np
import math
import random

def a(v, best,c1, present, gbest):
    #print(len(present))
    #print(len(best[0]))
    #print(len(v))
    print("fez")
    v2 = []
    rand = random.uniform(0, 1)
    #v[] = v[] + c1 * rand() * (pbest[] - present[]) + c2 * rand() * (gbest[] - present[])
    #W*velocity_vector[i]) + (c1*random.random()) * (pbest_position[i] - particle_position_vector[i]) + (c2*random.random()) * (gbest_position-particle_position_vector[i])
    v = v + (c1* rand) * (best[0] - present) + (c1 * rand) * (gbest[0])
    print(len((c1 * rand) * (gbest[0])))
    for vi in v:
        if vi > 1:
            vi = 1
        elif vi <0:
            vi = 0
        v2.append(vi)
    return v2

def b(p,v):
    p2 = []
    #present[] = persent[] + v[]
    p = p + v
    for pi in p:
        if pi > 1:
            pi = 1
        elif pi < 0:
            pi = 0
        p2.append(pi)
    return p2
def pso(population,x_train, y_train, x_test, y_test,x_validate,y_validate):
    # max_velocity = 0

    c1 = 1

    w, h = 20, 1200;
    velocity = [[0 for x in range(h)] for y in range(w)]
    present_positions = [[0 for x in range(h)] for y in range(w)]

    best_positions = [[0 for x in range(h)] for y in range(w)]

    best_lfitness = [0]*20
    atual_fitness = [0]*20

    gbest = ft.find_best(population, x_train, y_train, x_validate, y_validate)
    num_generations = 50

    for k in range(num_generations):

        lbest = ft.find_best(population,x_train, y_train, x_validate, y_validate)
        if lbest[1][0] > gbest[1][0]:
            gbest = lbest
        for particleIdx in range(len(population)):
            velocity[particleIdx] = a(velocity[particleIdx], lbest, c1,present_positions[particleIdx], gbest)
            present_positions[particleIdx] = b(present_positions[particleIdx],velocity[particleIdx])
            """

            for dimentionIdx in range(1200):
                print(velocity[particleIdx][dimentionIdx])
                velocity[particleIdx][dimentionIdx] = a(velocity[particleIdx][dimentionIdx], gbest,c1,present_positions[particleIdx][dimentionIdx])
                present_positions[particleIdx][dimentionIdx] = b(present_positions[particleIdx][dimentionIdx], velocity[particleIdx][dimentionIdx])
            """
    best_accu = ft.fitness_best(gbest[0], gbest[1], x_test, y_test)
    # print(best_accu)
    return best_accu



