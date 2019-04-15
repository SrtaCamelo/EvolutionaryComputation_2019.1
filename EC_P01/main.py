#import ga_evo as ga
#import neural_network as nn

#----------------Imports--------------------
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
#----------------Modules ---------------------
import ga_evo as ga
import neural_network as nn

#-------- Extructures----------------
ga_accu = []
a  = []
b = []
c = []


def scores(acuracy_list):
    #scaler = StandardScaler()
    series = pd.Series(acuracy_list)
    media = series.mean()
    dp = series.std()
    mediana = series.median()
    return media,dp,mediana

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
def call_classifiers(x_train, y_train, x_test, y_test):
#--------- Call Neural NetWork-------------
    #-----Generate Population------------
    pop_size = (10,5) # 20 individuals with 5 weights each
    population = np.random.uniform(low=-1.0, high=1.0, size=pop_size)

    #------------Call all Evolutionary Algoritms for same population------

    ga_accu  = ga.ga(population,x_train, y_train, x_test, y_test)
    print(ga_accu)
    #ga.append(ga_accu)

"""
Parametros: data : Pandas DataFrame
Essa função divide o dataSet randomizado em 10 folds,
repassando o conjunto de treinamento e teste pros classificadores.

"""
#--------------K-fold method (k = 10)---------------------
def experiment_call(data,n_start, n_final,key):
    for i in range(50):
        #Shuffle Data
        data = data.sample(frac=1)
    #---------Split Class column from features-------------
        X = data.iloc[:,n_start:n_final]
        Y = data[key]
    #----------Split Training from Test
        x_train, x_test,y_train, y_test = train_test_split(X, Y, test_size = 0.7, random_state = 0)
        call_classifiers(x_train, y_train, x_test, y_test)

#--------------Shuffle data an Split Features-------------
def prepareDataSet(datapath):
    #Shuffle
    data = pd.read_csv(datapath)
    #data = data.sample(frac=1)
    return data
#---------------------Main Code ------------------------
path = "steam.csv"
data = prepareDataSet(path)

experiment_call(data,2,len(data.columns)-1,"class")