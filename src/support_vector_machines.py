# -*- coding: utf-8 -*-

import svmutil
from svmutil import svm_read_problem
from svmutil import svm_problem
from svmutil import svm_parameter
from svmutil import svm_train
from svmutil import svm_predict
from svmutil import svm_save_model

def svm(X, Y, Xval, Yval):
    """
    Retorna o melhor valor para os parâmetros custo e gamma do SVM radial.
    
    Parâmetros
    ----------
    X : matriz com os dados de treinamento
    
    y : vetor com classes de cada dados de treinamento
    
    Xval : matriz com os dados de validação
    
    yval : vetor com as classes dos dados de validação
    
    Retorno
    -------
    custo, gamma : os melhores valores para os parêmetros custo e gamma.
    
     """
    
    #inicializa as variáveis que deverão ser retornadas pela função
    custo = 1000
    gamma = 1000
        
    C = [0.01,0.03,0.1,0.3,1,3,10,30] #Cost
    G = [0.01,0.03,0.1,0.3,1,3,10,30] #Gamma
        
    K = 2 #Kernel
    acuracia = 0.0 #Acc
    acuracia_atual = 0.0 #Actual acc
    
    for cost in C:
        for gamm in G:
            model = svm_train(Y, X, '-c %f -t %d -g %f' %(cost, K, gamm))
            acuracia_atual = svm_predict(Yval, Xval, model)[1][0]
            if acuracia < acuracia_atual:
                acuracia = acuracia_atual
                custo = cost
                gamma = gamm
    
    ##########################################################################

    return custo, gamma