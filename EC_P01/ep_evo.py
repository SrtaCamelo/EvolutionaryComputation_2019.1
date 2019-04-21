#Projeto 01, Evolutinary Computing
#Differencial Evolution Code
#Srta Camelo

import fitness as ft
import numpy as np

"""
REFERENCIA: https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/#
"""
#----------- Evolutive Programming -----------------
def ep(population,x_train, y_train, x_test, y_test):
    something = 0

pop_size = (5,3)
population = np.random.uniform(low=-1.0, high=1.0, size=pop_size)
#print("population")
#print(population)
ep(population,0,0,0,0)