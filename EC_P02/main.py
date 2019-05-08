#import ga_evo as ga
#import neural_network as nn

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
import ee_cauchy as eec2
import ee_cauchy2 as eec
#import neural_network as nn
import random_forest as rf

#-------- Extructures----------------
ga_accu = []
ee_accu  = []
ep_accu = []
de_accu = []
ee_caucht_accu = []
ee_cauchy2_accu = []
normal_accu = []


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
    print("GA e EP")
    t2,p2 = tstudant_hipotesis(ga_accu,ep_accu)
    print("T-value:",t2,"P-value:",p2)
    print("GA e EE")
    t2, p2 = tstudant_hipotesis(ga_accu, ee_accu)
    print("T-value:", t2, "P-value:", p2)
    print("GA e DE")
    t2, p2 = tstudant_hipotesis(ga_accu, de_accu)
    print("T-value:", t2, "P-value:", p2)
    print("EP e EE")
    t2, p2 = tstudant_hipotesis(ep_accu, ee_accu)
    print("T-value:", t2, "P-value:", p2)
    print("EP e DE")
    t2, p2 = tstudant_hipotesis(ep_accu, de_accu)
    print("T-value:", t2, "P-value:", p2)
    print("EE e DE")
    t2, p2 = tstudant_hipotesis(ee_accu, de_accu)
    print("T-value:", t2, "P-value:", p2)

#---------Função Fuleca só pra gerar o Log-----
def showStatistics(acuracy_list):
    mean, dp, median = scores(acuracy_list[0])
    print("###-------Algoritm:-----###",acuracy_list[1],"\n")
    print("#---Mean: ", mean,"\n")
    print("#---DP:", dp,"\n")
    print("#---Median:", median,"\n")

#----------Gerar população para Evolucionários-----
def generate_population():
    population = []
    return population
#--------------Chamar a Rede Neural/ Evolucionarios-------
"""
Parametros: x_train (conjunto de features de cada case para treino)
            y_train (classificação de cada case do treino)
            x_test (conjunto de features de cada case para teste)
            y_test (classificação de cada case do teste)
"""
#------------- Run all Algoritms for given dataSet------------
def call_classifiers(x_train, y_train, x_test, y_test,x_validation,y_validation):
#--------- Call Neural NetWork-------------
    #-----Generate Population------------
    pop_size = (20,1200) # 20 individuals
    population = np.random.uniform(low=0, high=1.0, size=pop_size)
    population1 = population.copy()
    population2 = population.copy()
    population3 = population.copy()
    population4 = population.copy()
    population5 = population.copy()
    population6 = population.copy()

    #------------Call all Evolutionary Algoritms for same population------

    accu_ga = ga.ga(population1,x_train, y_train, x_test, y_test,x_validation,y_validation)
    accu_ee = ee.es(population2,x_train, y_train, x_test, y_test,x_validation,y_validation)
    accu_de = de.de(population3,x_train, y_train, x_test, y_test,x_validation,y_validation)
    accu_pe = ep.ep(population4,x_train, y_train, x_test, y_test,x_validation,y_validation)
    accu_eec = eec.es(population5,x_train, y_train, x_test, y_test,x_validation,y_validation)
    accu_eec2 = eec2.es(population6,x_train, y_train, x_test, y_test,x_validation,y_validation)
    accu_normal, model = rf.rf(x_train, y_train,x_validation,y_validation)
    accu_normal = rf.rf_best(model,x_test, y_test)
    ga_accu.append(accu_ga)
    ee_accu.append(accu_ee)
    de_accu.append(accu_de)
    ep_accu.append(accu_pe)
    ee_caucht_accu.append(accu_eec)
    ee_cauchy2_accu.append(accu_eec2)
    normal_accu.append(accu_normal)


"""
Parametros: data : Pandas DataFrame
Essa função divide o dataSet randomizado em 10 folds,
repassando o conjunto de treinamento e teste pros classificadores.

"""
#--------------K-fold method (k = 10)---------------------
def experiment_call(data,n_start, n_final,key):
    for i in range(50):
        print(i)
        #Shuffle Data
        data = data.sample(frac=1)
    #---------Split Class column from features-------------
        X = data.iloc[:,n_start:n_final]
        Y = data[key]

    #----------Split Training from Test
        x_train, x_test,y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state = 0)
        x_validation, x_test, y_validation, y_test = train_test_split(x_test,y_test,test_size = 0.5, random_state = 0)
        call_classifiers(x_train, y_train, x_test, y_test,x_validation,y_validation)

#--------------Shuffle data an Split Features-------------
def prepareDataSet(datapath):
    #Shuffle
    data = pd.read_csv(datapath)
    #data = data.sample(frac=1)
    return data
#---------------------Main Code ------------------------
path = "steam_100.csv"
data = prepareDataSet(path)
experiment_call(data,2,len(data.columns)-1,"class")

#--------------These code lines above Do de SD, median and mean calculus---------------------------

List_ofLists =[(normal_accu, "Normal RF"),(ga_accu,"Genetic Algoritm"),(de_accu,"Discrete Evolution"), (ee_accu,"Evolutionary Estrategy"),(ep_accu,"Evolutionary Programming"),(ee_caucht_accu, "Cauchy EE normal"), (ee_cauchy2_accu, "Cauchy EE Crossover")]

for acuracy_list in List_ofLists:
    #print(acuracy_list[0])
    showStatistics(acuracy_list)

#-------------This call above calls t-studant hipotesis for classifiers variations-----------------
showHipotesis()
