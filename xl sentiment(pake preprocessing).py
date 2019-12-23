# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:07:44 2019

@author: HP
"""
import csv
from nltk.tokenize import RegexpTokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from math import log10
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def LoadDataset(fileName):#load data
    akun = []
    data = []
    label = []
    with open(fileName, 'r') as file:
        fileCsv = csv.reader(file, delimiter=';')
        for row in fileCsv:
            if row[0] != 'akun':
                akun.append(row[0])
                data.append(row[1])
                label.append(row[2])
    return akun,data,label

def Preprocessing(data):#Preprocessing
    cleanData = []
    tokenizer = RegexpTokenizer(r'\w+')
    factory_stopwords = StopWordRemoverFactory()
    stopwords = factory_stopwords.get_stop_words()
    factory_stemmer = StemmerFactory()
    stemmer = factory_stemmer.create_stemmer()
    for i in range(len(data)):
        lowerText = data[i].lower()#Case folding
        tokenizedText = tokenizer.tokenize(lowerText)#Punctual removal and tokenization
        swRemovedText = []#Stopwords removal
        for j in range(len(tokenizedText)):
            if tokenizedText[j] not in stopwords:
                swRemovedText.append(tokenizedText[j])
        stemmedText = []
        for k in range(len(swRemovedText)):#Stemming
            stemmedText.append(stemmer.stem(swRemovedText[k]))
        cleanData.append(stemmedText)
    return cleanData

def CreateUnigram(data):#Create unigram
    unigram = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] not in unigram:
                unigram.append(data[i][j])
    return unigram

def CreateTFIDF(data,unigram):#Create tf idf
    df = {}
    for u in unigram:
        for i in range(len(data)):
            if u in data[i]:
                if u in df:
                    df[u] += 1
                else:
                    df[u] = 1
    dataTFIDF = []
    for i in range(len(data)):
        tempTFIDF = []
        for j in range(len(unigram)):
            if unigram[j] in data[i]:
                tf = 0
                for k in range(len(data[i])):
                    if unigram[j] == data[i][k]:
                        tf += 1
                idf = log10(len(data)/df[unigram[j]])
                tempTFIDF.append(idf*tf)
            else:
                tempTFIDF.append(0)
        dataTFIDF.append(tempTFIDF)
    return dataTFIDF

akun,data,label = LoadDataset("labeling fix.csv")
cleanData = Preprocessing(data)
unigram = CreateUnigram(cleanData)
dataTFIDF = CreateTFIDF(cleanData,unigram)
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
result = cross_val_score(clf,dataTFIDF,label,cv=5)#10 = 10-fold-cross-validation
print("Dengan preprocessing")
print("Akurasi tiap fold:",result)
print("Akurasi rata-rata:",sum(result)/len(result))


























