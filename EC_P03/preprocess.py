#Projeto 01, Evolutinary Computing
#----Pre-processamento------

#--------imports------------
import os
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
#-----------Functions----------
"""
:parameter: path (string path)
:return: text data
This function opens the file and returns the data inside
"""
def openFile(path):
    lines = []
    f = open(path, 'r+')
    for line in f:
        lines.append(line)
    f.close()
    return lines


"""
:parameter: mainpath (string path)
:return: void
This function just copies the original files into other directory
(putting toguether all negative samples in a file and all positive 
in another). Original files were distributed into test and training.
"""
def moveAllfiles(mainpath):
    path_01 = ""
    finalpath = ""
"""
:parameter: void
:return: List of texts
This function opens all txt files and add them to a python list
"""
def indexFiles(mainpath):
    text_list = []

    files = os.listdir(mainpath)
    for file in files:
        path = mainpath + '/'+file
        text = openFile(path)
        if text != None:
            text_list.append(text)
    return text_list
"""
:parameter: textList: Python List
:returns: tokenList: Python List
Receives a list of texts and returns a tokenized text list (separates each word in a "token")
"""
def tokenizer(textList):
    tokenList = []
    for text in textList:
        tokenized = word_tokenize(text,'english')
        filtered = extractStopWords(tokenized)
        tokenList.append(filtered)
    return tokenList

def extractStopWords(data):
    stopWords = set(stopwords.words('english'))
    filteredWords = []
    for word in data:
        if word not in stopWords:
            filteredWords.append(word)
    return filteredWords
def compute_tf_idf(TextList):
    vectorizer = TfidfVectorizer(smooth_idf = True, min_df = 5,norm = 'l1')
    X = vectorizer.fit_transform(TextList)

    return X
def define_class_column(data,n):

    class_column1 = [1] * int((n/2))
    class_column0 = [0] * int((n/2))
    class_column = class_column1 + class_column0
    new_column = pd.DataFrame({'class':class_column})
    data = data.merge(new_column, left_index=True, right_index=True)
    print(data)
    return data
def pandasDataFrame(tfidf):
    data = pd.DataFrame(tfidf.toarray())
    data = define_class_column(data,len(data))
    print(len(data))
    data.to_csv("steam_100.csv", header=True, mode='a', sep=',')


#------------Main---------------
tags = ["positive","negative"]
data = []
mainpath = "positive.txt"     #Change Here to fetch other DataSet
#data = openFile(mainpath)
#print(len(data))

for tag in tags:
    path = tag+".txt"
    #print(path)
    textList = openFile(path)
    textList = textList[0:250]
    data.append(textList)

textList = data[0]+data[1]
#tokenized_list = tokenizer(textList)
#print(tokenized_list)
tfidf = compute_tf_idf(textList)
pandasDataFrame(tfidf)




