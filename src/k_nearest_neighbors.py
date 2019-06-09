# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def distancia(x, X):
    """
    Calcula a distância entre a amostra x e todos as amostras da 
    base X.
    Retorna um vetor de distancias. 
    """
    return np.sqrt(np.sum(np.power((X - x), 2), axis=1))

def knn(x, X, Y, K):
    """
    KNN método dos K-vizinhos mais proximos para predizer a classe de um novo
    dado.

    KNN (x, X, Y, K) retorna o rótulo y da amostra x e os índices
        [ind_viz] dos K-vizinhos mais próximos de x em X.
 
        Parâmetros de entrada:
        -> x (1 x n): amostra a ser classificada
        -> X (m x n): base de dados de treinamento
        -> Y (m x 1): conjunto de rótulos de cada amostra de X
        -> K (1 x 1): quantidade de vizinhos mais próximos
 
        Parâmetros de saída:
        -> y (1 x 1): predição (0 ou 1) do rótulo da amostra x
        -> ind_viz (K x 1): índice das K amostras mais próximas de x
                            encontradas em X (da mais próxima para a menos
                            próxima)
    """
    
    # Inicializa a variável de retorno e algumas variáveis uteis
    y = 0 # Inicializa rótulo como sendo da classe negativa
    ind_viz = np.ones(K, dtype=int) # Inicializa índices (linhas) em X das K amostras mais 
                             # próximas de x.
        
    # Calcula a distância entre a amostra de teste x e cada amostra de X. Você
    # deverá completar essa função.
    distancias = distancia(x, X)
	
    # Ordena o vetor de distâncias, retornando um vetor de indices da amostra mais próxima à mais distante
    idx_dists = np.argsort(distancias)
    
    # Seleciona os K vizinhos mais próximos
    ind_viz = idx_dists[0:K]
	
    # Encontra a classe do novo exemplo utilizando moda
    classes, num_exs = np.unique(Y[ind_viz], return_counts=True)
    y = classes[np.argmax(num_exs)]
        
    return y, ind_viz