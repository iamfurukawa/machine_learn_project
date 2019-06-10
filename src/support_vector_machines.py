# -*- coding: utf-8 -*-

import svmutil
from svmutil import svm_read_problem
from svmutil import svm_problem
from svmutil import svm_parameter
from svmutil import svm_train
from svmutil import svm_predict
from svmutil import svm_save_model

def svm_encontrar_melhor(X, Y, Xval, Yval, C, G, K):
    """
    Retorna o melhor valor para os parâmetros custo e gamma do SVM radial.
    
    Parâmetros
    ----------
    X : matriz com os dados de treinamento
    
    y : vetor com classes de cada dados de treinamento
    
    Xval : matriz com os dados de validação
    
    yval : vetor com as classes dos dados de validação
    
    C : lista com valores para custo
    
    G : lista com valores para gamma
    
    K : inteiro indicando o kernel a ser usado
    
    Retorno
    -------
    custo, gamma : os melhores valores para os parêmetros custo e gamma.
    
     """
    
    #inicializa as variáveis que deverão ser retornadas pela função
    custo = 1000
    gamma = 1000
    
    acuracia = 0.0 #Acc
    acuracia_atual = 0.0 #Actual acc
    
    for cost in C:
        for gamm in G:
            model = svm_train(Y, X, '-c %f -t %d -g %f -q' %(cost, K, gamm))
            acuracia_atual = svm_predict(Yval, Xval, model)[1][0]
            if acuracia < acuracia_atual:
                acuracia = acuracia_atual
                custo = cost
                gamma = gamm
    
    return custo, gamma

def svm_predizer(X, Y, X_teste, Y_teste, C, G, K):
    """
    Retorna o melhor valor para os parâmetros custo e gamma do SVM radial.
    
    Parâmetros
    ----------
    X : matriz com os dados de treinamento
    
    y : vetor com classes de cada dados de treinamento
    
    X_teste : amostra
    
    Y_teste : classe da amostra
    
    C : lista com valores para custo
    
    G : lista com valores para gamma
    
    K : inteiro indicando o kernel a ser usado
    
    Retorno
    -------
    resultado : classificação do SVM.
    
    """
    model = svm_train(Y, X, '-c %f -t %d -g %f -q' %(C, K, G))
    resultado = svm_predict(Y_teste, X_teste, model)
    return resultado