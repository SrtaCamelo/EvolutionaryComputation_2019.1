#----------------Imports--------------------
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy import stats
from itertools import compress

#----------------Modules ---------------------
import ga_evo as ga
import ee_evo as ee
import de_evo as de
import ep_evo as ep
import ee_cauchy2 as eec
#import neural_network as nn
import random_forest as rf

#-------- Extructures----------------
ga_accu = [0.63,
0.68,
0.63,
0.57,
0.65,
0.61,
0.67,
0.68,
0.65,
0.62,
0.68]
ee_accu  = [0.76,
0.75,
0.85,
0.65,
0.58,
0.49,
0.78,
0.58,
0.8,
0.73,
0.8]
ep_accu = [0.69,
0.46,
0.63,
0.67,
0.71,
0.64,
0.69,
0.67,
0.61,
0.49,
0.59,]
de_accu = [0.65,
0.74,
0.71,
0.69,
0.72,
0.81,
0.73,
0.68,
0.75,
0.80,
0.81]
ee_caucht_accu = [0.63,
0.68,
0.63,
0.57,
0.65,
0.61,
0.67,
0.68,
0.65,
0.62,
0.68]
ee_cauchy2_accu = [0.63,
0.68,
0.63,
0.57,
0.65,
0.61,
0.67,
0.68,
0.65,
0.62,
0.68]
normal_accu = [0.74,
0.56,
0.7,
0.62,
0.66,
0.67,
0.53,
0.48,
0.65,
0.46,
0.7]

gso_accu = [
0.79,
0.7,
0.65,
0.78,
0.8,
0.74,
0.79,
0.73,
0.81,
0.79,
0.8,
0.69,
0.77,
0.7,
0.75,
0.62,
0.78]
pso_accu = [0.79,
0.7,
0.65,
0.78,
0.8,
0.74,
0.68,
0.65,
0.62,
0.68]

def scores(acuracy_list):
    #scaler = StandardScaler()
    series = pd.Series(acuracy_list)
    media = series.mean()
    dp = series.std()
    mediana = series.median()
    return media,dp,mediana

#---------------Gosset's hipotesis calculatiom---
def tstudant_hipotesis(a,b):
    t2, p2 = stats.ttest_ind(a, b)
    return t2,p2
#--------------Log Generator (Could have been saved in a txt file, but naaaah)--------------
def showHipotesis():
    print("Normal e GA")
    t2, p2 = tstudant_hipotesis(normal_accu,ga_accu)
    print("T-value:", t2, "P-value:", p2)

    print("Normal e EP")
    t2, p2 = tstudant_hipotesis(normal_accu, ep_accu)
    print("T-value:", t2, "P-value:", p2)

    print("Normal e EE")
    t2, p2 = tstudant_hipotesis(normal_accu, ee_accu)
    print("T-value:", t2, "P-value:", p2)

    print("Normal e DE")
    t2, p2 = tstudant_hipotesis(normal_accu, de_accu)
    print("T-value:", t2, "P-value:", p2)

    print("Normal e EE_Cauchy")
    t2, p2 = tstudant_hipotesis(normal_accu, ee_caucht_accu)
    print("T-value:", t2, "P-value:", p2)

    print("Normal e EE_Cauchy2")
    t2, p2 = tstudant_hipotesis(normal_accu, ee_cauchy2_accu)
    print("T-value:", t2, "P-value:", p2)

    print("Normal e GSO")
    t2, p2 = tstudant_hipotesis(normal_accu, gso_accu)
    print("T-value:", t2, "P-value:", p2)

    print("Normal e PSO")
    t2, p2 = tstudant_hipotesis(normal_accu, pso_accu)
    print("T-value:", t2, "P-value:", p2)

    print("GA e EP")
    t2,p2 = tstudant_hipotesis(ga_accu,ep_accu)
    print("T-value:",t2,"P-value:",p2)

    print("GA e EE")
    t2, p2 = tstudant_hipotesis(ga_accu, ee_accu)
    print("T-value:", t2, "P-value:", p2)

    print("GA e DE")
    t2, p2 = tstudant_hipotesis(ga_accu, de_accu)
    print("T-value:", t2, "P-value:", p2)


    print("GA e ee_cauchy")
    t2, p2 = tstudant_hipotesis(ga_accu, ee_caucht_accu)
    print("T-value:", t2, "P-value:", p2)

    print("GA e ee_cauchy2")
    t2, p2 = tstudant_hipotesis(ga_accu, ee_cauchy2_accu)
    print("T-value:", t2, "P-value:", p2)

    print("GA e GSO")
    t2, p2 = tstudant_hipotesis(ga_accu, gso_accu)
    print("T-value:", t2, "P-value:", p2)

    print("GA e PSO")
    t2, p2 = tstudant_hipotesis(ga_accu, pso_accu)
    print("T-value:", t2, "P-value:", p2)

    print("EP e EE")
    t2, p2 = tstudant_hipotesis(ep_accu, ee_accu)
    print("T-value:", t2, "P-value:", p2)

    print("EP e DE")
    t2, p2 = tstudant_hipotesis(ep_accu, de_accu)
    print("T-value:", t2, "P-value:", p2)

    print("EP e ee_cauchy")
    t2, p2 = tstudant_hipotesis(ep_accu, ee_caucht_accu)
    print("T-value:", t2, "P-value:", p2)

    print("EP e ee_cauchy2")
    t2, p2 = tstudant_hipotesis(ep_accu, ee_cauchy2_accu)
    print("T-value:", t2, "P-value:", p2)

    print("EP e GSO")
    t2, p2 = tstudant_hipotesis(ep_accu, gso_accu)
    print("T-value:", t2, "P-value:", p2)

    print("EP e PSO")
    t2, p2 = tstudant_hipotesis(ep_accu, pso_accu)
    print("T-value:", t2, "P-value:", p2)

    print("EE e DE")
    t2, p2 = tstudant_hipotesis(ee_accu, de_accu)
    print("T-value:", t2, "P-value:", p2)

    print("EE e ee_cauchy")
    t2, p2 = tstudant_hipotesis(ee_accu, ee_caucht_accu)
    print("T-value:", t2, "P-value:", p2)

    print("EE e ee_cauchy2")
    t2, p2 = tstudant_hipotesis(ee_accu, ee_cauchy2_accu)
    print("T-value:", t2, "P-value:", p2)

    print("EE e GSO")
    t2, p2 = tstudant_hipotesis(ee_accu, gso_accu)
    print("T-value:", t2, "P-value:", p2)

    print("EE e PSO")
    t2, p2 = tstudant_hipotesis(ee_accu, pso_accu)
    print("T-value:", t2, "P-value:", p2)

    print("DE e ee_cauchy2")
    t2, p2 = tstudant_hipotesis(de_accu, ee_caucht_accu)
    print("T-value:", t2, "P-value:", p2)

    print("DE e ee_cauchy2")
    t2, p2 = tstudant_hipotesis(de_accu, ee_cauchy2_accu)
    print("T-value:", t2, "P-value:", p2)

    print("DE e GSO")
    t2, p2 = tstudant_hipotesis(de_accu, gso_accu)
    print("T-value:", t2, "P-value:", p2)

    print("DE e PSO")
    t2, p2 = tstudant_hipotesis(de_accu, pso_accu)
    print("T-value:", t2, "P-value:", p2)

    print("GSO e PSO")
    t2, p2 = tstudant_hipotesis(gso_accu, pso_accu)
    print("T-value:", t2, "P-value:", p2)

#---------Função Fuleca só pra gerar o Log-----
def showStatistics(acuracy_list):
    mean, dp, median = scores(acuracy_list[0])
    print("###-------Algoritm:-----###",acuracy_list[1],"\n")
    print("#---Mean: ", mean,"\n")
    print("#---DP:", dp,"\n")
    print("#---Median:", median,"\n")


#--------------These code lines above Do de SD, median and mean calculus---------------------------

List_ofLists =[(normal_accu, "Normal RF"),(ga_accu,"Genetic Algoritm"),(de_accu,"Discrete Evolution"), (ee_accu,"Evolutionary Estrategy"),(ep_accu,"Evolutionary Programming"),(ee_caucht_accu, "Cauchy EE normal"), (ee_cauchy2_accu, "Cauchy EE Crossover"), (ee_cauchy2_accu, "Cauchy EE Crossover"), (gso_accu, "GSO"), (pso_accu, "PSO")]

for acuracy_list in List_ofLists:
    #print(acuracy_list[0])
    showStatistics(acuracy_list)

#-------------This call above calls t-studant hipotesis for classifiers variations-----------------
showHipotesis()