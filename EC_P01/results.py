from scipy import stats
import pandas as pd
#-------- Extructures----------------
ga_accu = [0.7733333349227905,0.7466666674613953,0.7266666682561239,
0.8000000015894572]
ee_accu  = [0.7933333373069763,0.7266666682561239,0.7400000007947286,0.7733333349227905]
ep_accu = [0.8000000015894572,
0.7266666682561239,
0.7466666674613953,
0.7733333349227905]
de_accu = [0.820000003973643
,0.7333333325386048
,0.8266666706403096
,0.7333333325386048]

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

List_ofLists =[(ga_accu,"Genetic Algoritm"),(de_accu,"Discrete Evolution"), (ee_accu,"Evolutionary Estrategy"),(ep_accu,"Evolutionary Programming")]

for acuracy_list in List_ofLists:
    #print(acuracy_list[0])
    print(len(acuracy_list[0]))
    #showStatistics(acuracy_list)

#-------------This call above calls t-studant hipotesis for classifiers variations-----------------
showHipotesis()