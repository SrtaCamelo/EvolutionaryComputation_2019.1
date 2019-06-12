#Projeto 03, Evolutinary Computing
#Bacterial Foraging Optimization Algorithm
#Srta Camelo

#-----------------imports-----------------
import numpy as np
import fitness as ft
import random as rd
import math
import numpy as np

def interaction(cell, population, dattack, wattack, wrepell, hrepell,s,p):
    first_sum = 0
    for i in range(s):
        inner_loop = 0
        for m in range(p):
            #DUVIDA: OTHER SERIA OUTRA BACTERIA ALEATORIA?????????
            other = rd.uniform(0, 1)
            diff = cell[m] - other
            diff = diff**2
            inner_loop += diff
        inner_loop = -wattack * inner_loop
        inner_loop = -dattack * math.exp(inner_loop)
        first_sum += inner_loop

    second_sum = 0
    for i in range(s):

        inner_loop = 0
        for m in range(p):
            other = rd.uniform(0, 1)
            diff = cell[m] - other
            diff = diff**2
            inner_loop += diff
        inner_loop = -wrepell * inner_loop
        inner_loop = hrepell * math.exp(inner_loop)
        second_sum += inner_loop
    gcell = first_sum + second_sum

    return gcell


def chemotaxixAndSwim(population,CellSize,PopSize,Ns,StepSize,x_train, y_train,x_validate,y_validate):
    dattack = 0.1
    wattack = 0.2
    hrepellant = 0.1
    wrepellant = 10
    cellHealth_all = []
    for m  in range(PopSize):
        cellft = ft.fitness(population[m],x_train, y_train,x_validate,y_validate )
        cellft = cellft[0]
        cellft = cellft + interaction(population[m],population,dattack,wattack,wrepellant,hrepellant,20,1200)
        cellHealth = cellft
        cell2 = population[m].copy()
        for i in range(Ns):
            #DUVIDA CRIAR UMA CELULA VAZIA OU COPIAR DA CELULA MAE? SetepSize array ou escalar?
            random_step = np.random.choice([1, -1], CellSize)
            random_step = np.array(random_step) * StepSize
            cell2 = np.add(np.array(cell2),random_step).tolist()
            cell2ft = ft.fitness(cell2,x_train, y_train,x_validate,y_validate )
            cell2ft = cell2ft[0]
            cell2ft += interaction(cell2,population,dattack,wattack,wrepellant,hrepellant,20,1200)

            if cell2ft < cellft:
                i = Ns
                break
            else:
                population[m] = cell2
                cellHealth = cellHealth + cell2ft
        cellHealth_all.append(cellHealth)
    return population, cellHealth_all



"""
Ned = Number of elimination-dispersal steps
Nre = is the number of reproduction steps
Nc = is the number of chemotaxis steps
Ns = is the number of swim steps for a given cell
"""


def bfo(population,x_train, y_train, x_test, y_test,x_validate,y_validate):
    #print("BFO")
    Celln = 20
    Ned = 10
    Nre = 10
    Nc = 20
    Ns = 10
    Ped = 0.25
    StepSize = 0.1

    best_fit = ft.find_best(population, x_train, y_train, x_validate, y_validate)
    for l in range(Ned):
        for k in range(Nre):
            for j in range(Nc):
                population, cellHealth_all = chemotaxixAndSwim(population,1200,20,Ns,StepSize,x_train, y_train,x_validate,y_validate)
                local_best = ft.find_best(population, x_train, y_train, x_validate, y_validate)
                if local_best[1][0] > best_fit[1][0]:
                    best_fit = local_best
            #DUVIDA DEVERIA USAR O HEALTH AO INVES DO FITNESS AQUI?
            all_fitness, all_models = ft.calculate_pop_ft(population,x_train, y_train,x_validate, y_validate)
            new_pop, ft_new, model_new = ft.fetch_n_better(population, all_fitness,all_models, 10)
        #DIVIDA COMO GARANTIR QUE NO FINAL A POPULAÇÃO VAI TER 20 BACTERIAS???
        for cell in new_pop:
            if(rd.uniform(0, 1) >= Ped):
                cell = np.random.uniform(low=0, high=1.0, size= 1200)
                new_pop.append(cell)
        population = new_pop
    bfo_accu = ft.fitness_best(best_fit[0], best_fit[1][1], x_test, y_test)

    return bfo_accu
"""
#Tessst
#Test
pop_size = (5,3)
population = np.random.uniform(low=0, high=1.0, size=pop_size)
#print("population")
#print(population)
acc = bfo(population,0,0,0,0,0,0)
#print(acc)
"""