#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os, re, operator
from pprint import pprint

from src.utils import *
from src.k_nearest_neighbors import *
from src.logistic_regression import *
from src.naive_bayes import *
from src.neutral_network import *
from src.support_vector_machines import *
from src.validation import *

PATH_POSITIVE_TRUTHFUL  = 'op_spam_v1.4/positive/truthful/'
PATH_POSITIVE_DECEPTIVE = 'op_spam_v1.4/positive/deceptive/'
PATH_NEGATIVE_TRUTHFUL  = 'op_spam_v1.4/negative/truthful/'
PATH_NEGATIVE_DECEPTIVE = 'op_spam_v1.4/negative/deceptive/'
ALL_PATH = [PATH_POSITIVE_TRUTHFUL, PATH_POSITIVE_DECEPTIVE,
           PATH_NEGATIVE_TRUTHFUL, PATH_NEGATIVE_DECEPTIVE]

def wordsProcessed(path):
    dic = {}
    dic2 = {}
    for texts in os.listdir(path):
        with open(path + texts, 'r', encoding='utf-8') as stream:
            text = stream.read()
            wordSplited = []
            
            #Pre-pre processing
            wordSplited2 = [word for word in (re.sub(r'[^\w\s]+','', text.replace('\n','')).lower().split(' ')) if word != '']

            #N-Gram
            for index in range(2, 5):
                ngrams = zip(*[wordSplited2[i:] for i in range(index)])
                wordSplited += ([" ".join(ngram) for ngram in ngrams])
            
            #Creating dictonary for count n-gram words
            for word in wordSplited:
                if word in dic:
                    dic[word] += 1
                else:
                    dic[word] = 1
                    
    #Remove values < 2
    for key, value in dic.items():
        if value > 3:
            dic2[key] = value
    listWords = [k for k, v in dic2.items()]
    dic2 = sorted(dic2.items(), key=operator.itemgetter(1))
    
    return wordSplited, dic2, listWords

posTruthWordSplited, posTruthDic, listWords = wordsProcessed(PATH_POSITIVE_DECEPTIVE)
#pprint(posTruthDic)
print(len(posTruthDic))


# In[2]:


matrix = []
def generateFeatures(path, example_class):
    i = 1
    for texts in os.listdir(path):
        i+=1
        #print(i)
        aux = np.zeros(len(listWords) + 1)
        with open(path + texts, 'r', encoding='utf-8') as stream:
            #Pre-pre processing
            text = stream.read()
            allNgrams = []
            wordSplited = [word for word in (re.sub(r'[^\w\s]+','', text.replace('\n','')).lower().split(' ')) if word != '']

            #N-Gram
            for index in range(2, 5):
                ngrams = zip(*[wordSplited[i:] for i in range(index)])
                allNgrams += ([" ".join(ngram) for ngram in ngrams])
            
            for idx, word in enumerate(listWords):
                for ngram in allNgrams:
                    if word == ngram:
                        aux[idx] += 1
            aux[len(listWords)] = example_class
            matrix.append(aux)
            
generateFeatures(PATH_POSITIVE_DECEPTIVE, 1)
generateFeatures(PATH_POSITIVE_TRUTHFUL, 0)

matrix = np.array(matrix)
matrix_norm, mu, sigma = normalize(matrix[:, :-1])
Y = matrix[:, -1]

#save(np.column_stack((matrix_norm, matrix[:, -1])), listWords)
print('ACABOOOOOOO')


# In[3]:


if matrix_norm.shape[1] != len(posTruthDic):
    raise Exception('Tamanhos diferentes!')


# In[4]:


a = matrix_norm[400:720]
b = matrix_norm[0:320]
X2 = np.append(a,b, axis=0)

a = Y[400:720]
b = Y[0:320]
Y2 = np.append(a,b, axis=0)

a = matrix_norm[720:800]
b = matrix_norm[320:400]
X_val = np.append(a,b, axis=0)

a = Y[720:800]
b = Y[320:400]
Y_val = np.append(a,b, axis=0)

#custo, gamma = svm(X2, Y2, X_val, Y_val)
#print(custo, '\t', gamma)


# In[5]:


y, ind_viz = knn(X_val[81], X2, Y2, 5)
print(Y_val[81], ' ',y,' ', ind_viz, ' ', Y2[ind_viz])

