# -*- coding: utf-8 -*-
#pVitoria = sum(Y==1)/len(Y) 
#pDerrota = sum(Y==0)/len(Y)

import numpy as np
import pandas as pd

def calcularProbabilidades(X, Y):
    """
    CALCULARPROBABILIDADES Computa a probabilidade de ocorrencia de cada 
    atributo por rotulo possivel. A funcao retorna dois vetores de tamanho n
    (qtde de atributos), um para cada classe.
    
    CALCULARPROBABILIDADES(X, Y) calcula a probabilidade de ocorrencia de cada atributo em cada classe. 
    Cada vetor de saida tem dimensao (n x 1), sendo n a quantidade de atributos por amostra.
    """
    
    # Inicializa os vetores de probabilidades
    probsPos = np.zeros(X.shape[1])
    probsNeg = np.zeros(X.shape[1])

    # Seleciona os índices de cada uma das classes
    idxPos = np.where(Y == 1)
    idxNeg = np.where(Y == 0)
    
    # Calcula o número de ocorrências de valores nas colunas para cada classe
    occPos = X[idxPos].sum(axis=0) + 1
    occNeg = X[idxNeg].sum(axis=0) + 1
    
    print(X[idxPos].sum())
    print(X[idxPos].shape[1])

    # Calcula as probabilidades
    probsPos = occPos / (X[idxPos].sum() + X[idxPos].shape[1])
    probsNeg = occNeg / (X[idxNeg].sum() + X[idxNeg].shape[1])

    return probsPos, probsNeg

def classificacao(x, probsPos, probsNeg, probClassPos, probClassNeg):
    
    # Realiza o cáculo das probabilidades
    probPos = probClassPos * np.prod([ prob ** x[idx] for idx, prob in enumerate(probsPos) ])
    probNeg = probClassNeg * np.prod([ prob ** x[idx] for idx, prob in enumerate(probsNeg) ])
    
    return probPos, probNeg