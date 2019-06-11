# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_confusionMatrix(Y_test, Y_pred, classes):
    """
    Retorna a matriz de confusao, onde o numero de linhas e 
        e numero de colunas e igual ao numero de classes
        
    Parametros
    ----------   
    Y_test: vetor com as classes verdadeiras dos dados
    
    Y_pred: vetor com as classes preditas pelo metodo de classificacao
    
    classes: classes do problema
    
    
    Retorno
    -------
    cm: matriz de confusao (array numpy, em que o numero de linhas e de colunas
        e igual ao numero de classes)
    
    """
    
    # inicia a matriz de confusão
    cm = np.zeros( [len(classes),len(classes)], dtype=int )

    #Versão para classes 0..N
    #for i in range(0,len(Y_pred)):
    #    cm[Y_test[i]][int(Y_pred[i])] += 1 
    
    for idxRow, classesRow in enumerate(classes):
        for idxColumn, classesColumn in enumerate(classes):
            cm[idxRow][idxColumn] = len(np.where((Y_test == classesRow) & (Y_pred == classesColumn))[0])
    
    return cm

def relatorioDesempenho(matriz_confusao, classes, imprimeRelatorio=False):
    """
    Funcao usada calcular as medidas de desempenho da classificação.

    Parametros
    ----------   
    matriz_confusao: array numpy que representa a matriz de confusao 
                   obtida na classificacao. O numero de linhas e de colunas
                   dessa matriz e igual ao numero de classes.

    classes: classes do problema

    imprimeRelatorio: variavel booleana que indica se o relatorio de desempenho
                    deve ser impresso ou nao. 

    Retorno
    -------
    resultados: variavel do tipo dicionario (dictionary). As chaves
              desse dicionario serao os nomes das medidas de desempenho; os valores
              para cada chave serao as medidas de desempenho calculadas na funcao.

              Mais especificamente, o dicionario devera conter as seguintes chaves:

               - acuracia: valor entre 0 e 1 
               - revocacao: um vetor contendo a revocacao obtida em relacao a cada classe
                            do problema
               - precisao: um vetor contendo a precisao obtida em relacao a cada classe
                            do problema
               - fmedida: um vetor contendo a F-medida obtida em relacao a cada classe
                            do problema
               - revocacao_macroAverage: valor entre 0 e 1
               - precisao_macroAverage: valor entre 0 e 1
               - fmedida_macroAverage: valor entre 0 e 1
               - revocacao_microAverage: valor entre 0 e 1
               - precisao_microAverage: valor entre 0 e 1
               - fmedida_microAverage: valor entre 0 e 1
    """

    n_teste = sum(sum(matriz_confusao))

    nClasses = len( matriz_confusao ) #numero de classes

    # inicializa as medidas que deverao ser calculadas
    vp=np.zeros( nClasses ) # quantidade de verdadeiros positivos
    vn=np.zeros( nClasses ) # quantidade de verdadeiros negativos
    fp=np.zeros( nClasses ) # quantidade de falsos positivos
    fn=np.zeros( nClasses ) # quantidade de falsos negativos

    acuracia = 0.0 

    revocacao = np.zeros( nClasses ) # nesse vetor, devera ser guardada a revocacao para cada uma das classes
    revocacao_macroAverage = 0.0
    revocacao_microAverage = 0.0

    precisao = np.zeros( nClasses ) # nesse vetor, devera ser guardada a revocacao para cada uma das classes
    precisao_macroAverage = 0.0
    precisao_microAverage = 0.0

    fmedida = np.zeros( nClasses ) # nesse vetor, devera ser guardada a revocacao para cada uma das classes
    fmedida_macroAverage = 0.0
    fmedida_microAverage = 0.0

    np.seterr(divide='ignore', invalid='ignore')

    acuracia = np.diagonal(matriz_confusao).sum() / matriz_confusao.sum()

    revocacao = np.diagonal(matriz_confusao) / matriz_confusao.sum(axis=1)
    precisao = np.diagonal(matriz_confusao) / matriz_confusao.sum(axis=0)
    fmedida = 2 * ((precisao * revocacao) / (precisao + revocacao))

    revocacao_macroAverage = revocacao.sum() / nClasses
    precisao_macroAverage = precisao.sum() / nClasses
    fmedida_macroAverage = fmedida.sum() / nClasses

    revocacao_microAverage = ((np.diagonal(matriz_confusao).sum() / nClasses) / (matriz_confusao.sum(axis=1).sum() / nClasses))
    precisao_microAverage = ((np.diagonal(matriz_confusao).sum() / nClasses) / (matriz_confusao.sum(axis=0).sum() / nClasses))
    fmedida_microAverage = 2 * ((precisao_microAverage * revocacao_microAverage) / (precisao_microAverage + revocacao_microAverage))
    
  # imprimindo os resultados para cada classe
    if imprimeRelatorio:
        print('\n\tRevocacao   Precisao   F-medida   Classe')
        for i in range(0,nClasses):
            print('\t%1.3f       %1.3f      %1.3f      %s' % (revocacao[i], precisao[i], fmedida[i],classes[i] ) )
            print('\t------------------------------------------------');
          #imprime as médias
            print('\t%1.3f       %1.3f      %1.3f      Média macro' % (revocacao_macroAverage, precisao_macroAverage, fmedida_macroAverage) )
            print('\t%1.3f       %1.3f      %1.3f      Média micro\n' % (revocacao_microAverage, precisao_microAverage, fmedida_microAverage) )
            print('\tAcuracia: %1.3f' %acuracia)

  # guarda os resultados em uma estrutura tipo dicionario
    resultados = {'revocacao': revocacao, 'acuracia': acuracia, 'precisao':precisao, 'fmedida':fmedida}
    resultados.update({'revocacao_macroAverage':revocacao_macroAverage, 'precisao_macroAverage':precisao_macroAverage, 'fmedida_macroAverage':fmedida_macroAverage})
    resultados.update({'revocacao_microAverage':revocacao_microAverage, 'precisao_microAverage':precisao_microAverage, 'fmedida_microAverage':fmedida_microAverage})
    resultados.update({'confusionMatrix': matriz_confusao})

    return resultados

def curva_aprendizado(X, Y, Xval, Yval):
    """
    Funcao usada gerar a curva de aprendizado.
  
    Parametros
    ----------
  
    X : matriz com os dados de treinamento
  
    Y : vetor com as classes dos dados de treinamento
  
    Xval : matriz com os dados de validação
  
    Yval : vetor com as classes dos dados de validação
  
    """

    # inicializa as listas que guardarao a performance no treinamento e na validacao
    perf_train = []
    perf_val = []

    # inicializa o parametro de regularizacao da regressao logistica
    lambda_reg = 1
        
    # Configura o numero de interacaoes da regressao logistica
    iteracoes = 500
    
    for idx in range(10, len(X)):
        theta = treinamento(X[:idx], Y[:idx], lambda_reg, iteracoes)
        
        predTr = predicao(X[:idx], theta)
        predVal = predicao(Xval, theta)
        
        cmTr = get_confusionMatrix(Y[:idx], predTr, np.unique(np.sort(Y_test)))
        cmVal = get_confusionMatrix(Yval, predVal, np.unique(np.sort(Y_test)))
        
        relTr = relatorioDesempenho(cmTr, np.unique(np.sort(Y_test)))
        relVal = relatorioDesempenho(cmVal, np.unique(np.sort(Y_test)))
        
        perf_val.append(relVal['acuracia'])
        perf_train.append(relTr['acuracia'])
           
    # Define o tamanho da figura 
    plt.figure(figsize=(20,12))

    # Plota os dados
    plt.plot(perf_train, color='blue', linestyle='-', linewidth=1.5, label='Treino') 
    plt.plot(perf_val, color='red', linestyle='-', linewidth=1.5, label='Validação')

    # Define os nomes do eixo x e do eixo y
    plt.xlabel(r'# Qtd. de dados de treinamento',fontsize='x-large') 
    plt.ylabel(r'Acuracia',fontsize='x-large') 

    # Define o título do gráfico
    plt.title(r'Curva de aprendizado', fontsize='x-large')

    # Acrescenta um grid no gráfico
    plt.grid(axis='both')

    # Plota a legenda
    plt.legend()
    plt.show()
    
def stratified_kfolds(target, k, classes):
    """
    Retorna os indices dos dados de treinamento e teste para cada uma das k rodadas 
    
    Parametros
    ----------   
    target: vetor com as classes dos dados
    
    k: quantidade de folds 
    
    Retorno
    -------
    folds_final: os indices dos dados de treinamento e teste para cada uma das k rodadas 
    
    """

    # Inicializa a variavel que precisa ser retornada. 
    # Cada elemento do vetor folds_final deve ser outro vetor de duas posicoes: a primeira
    #    posicao deve conter um vetor com os indices de treinamento relativos ao i-esimo fold;
    #    a segunda posicao deve conter um vetor com os indices de teste relativos ao i-esimo fold.
    folds_final = np.zeros( k,dtype='object')

    # inicializa o vetor onde o k-esimo elemento guarda os indices dos dados de treinamento 
    # relativos ao k-esimo fold 
    train_index = np.zeros( k,dtype='object')
    
    # inicializa o vetor onde o k-esimo elemento guarda os indices dos dados de teste 
    # relativos ao k-esimo fold 
    test_index = np.zeros( k,dtype='object')
    
    # inicializa cada posicao do vetor folds_final que devera ser retornado pela funcao
    for i in folds_final:
        
        train_index[i] = [] # indices dos dados de treinamento relativos ao fold i
        test_index[i] = [] # indices dos dados de teste relativos ao fold i
        
        # inicializa o i-esimo elemento do vetor que devera ser retornado
        folds_final[i] = np.array( [train_index[i],test_index[i]] )
    
    percPerClass = [len(np.where(target == classe)[0]) / len(target) for classe in classes]
    qtyElemPerK = len(target) / k
    qtyElemPerClass = [qtyElemPerK * perc for perc in percPerClass]
    folders = [[] for i in range(k)]
    
    for idx, classe in enumerate(classes):
        qty = int(qtyElemPerClass[idx])
        for nfolder in range(0, k):
            folders[nfolder] += np.where(classe == target)[0][nfolder * qty : (nfolder * qty) + qty].tolist()
            
    folders = [sorted(nfolder) for nfolder in folders]
    
    for idxff, fold_final in enumerate(folds_final):
        train_index = []
        test_index = []
        for idxf, folder in enumerate(folders):
            if idxf == idxff:
                test_index = folder
            else:
                train_index += folder
        
        folds_final[idxff] = np.array([train_index, test_index])
        
    return folds_final

def mediaFolds( resultados, classes ):
    
    nClasses = len(classes)
    
    acuracia = np.zeros( len(resultados) )

    revocacao = np.zeros( [len(resultados),len(classes)] )
    precisao = np.zeros( [len(resultados),len(classes)] )
    fmedida = np.zeros( [len(resultados),len(classes)] )

    revocacao_macroAverage = np.zeros( len(resultados) )
    precisao_macroAverage = np.zeros( len(resultados) )
    fmedida_macroAverage = np.zeros( len(resultados) )

    revocacao_microAverage = np.zeros( len(resultados) )
    precisao_microAverage = np.zeros( len(resultados) )
    fmedida_microAverage = np.zeros( len(resultados) )


    for i in range(len(resultados)):
        acuracia[i] = resultados[i]['acuracia']
        
        revocacao[i,:] = resultados[i]['revocacao']
        precisao[i,:] = resultados[i]['precisao']
        fmedida[i,:] = resultados[i]['fmedida']

        revocacao_macroAverage[i] = resultados[i]['revocacao_macroAverage']
        precisao_macroAverage[i] = resultados[i]['precisao_macroAverage']
        fmedida_macroAverage[i] = resultados[i]['fmedida_macroAverage']

        revocacao_microAverage[i] = resultados[i]['revocacao_microAverage']
        precisao_microAverage[i] = resultados[i]['precisao_microAverage']
        fmedida_microAverage[i] = resultados[i]['fmedida_microAverage']
        
    # imprimindo os resultados para cada classe
    print('\n\tRevocacao   Precisao   F-medida   Classe')
    for i in range(0,nClasses):
        print('\t%1.3f       %1.3f      %1.3f      %s' % (np.mean(revocacao[:,i]), np.mean(precisao[:,i]), np.mean(fmedida[:,i]), classes[i] ) )

    print('\t---------------------------------------------------------------------')
  
    #imprime as medias
    print('\t%1.3f       %1.3f      %1.3f      Média macro' % (np.mean(revocacao_macroAverage), np.mean(precisao_macroAverage), np.mean(fmedida_macroAverage)) )
    print('\t%1.3f       %1.3f      %1.3f      Média micro\n' % (np.mean(revocacao_microAverage), np.mean(precisao_microAverage), np.mean(fmedida_microAverage)) )

    print('\tAcuracia: %1.3f' %np.mean(acuracia))
    return np.mean(acuracia)

def stratified_holdOut(target, pTrain):
    """
    Retorna os indices dos dados de treinamento e teste 
    
    Parâmetros
    ----------   
    target: vetor com as classes dos dados
    
    pTrain: porcentagem de dados de treinamento
    
    Retorno
    -------
    train_index: índices dos dados de treinamento 
    test_index: índices dos dados de teste 
    
    """
    
    # inicializa as variaveis que precisam ser retornadas 
    train_index = []
    test_index = []

    ########################## COMPLETE O CÓDIGO AQUI  ###############################
    #  Instruções: Complete o codigo para retornar os índices dos dados de  
    #              treinamento e dos dados de teste.
    #              
    #              Obs: - os conjuntos de treinamento e teste devem ser criados
    #                     de maneira estratificada, ou seja, deve ser mantida a 
    #                     a distribuição original dos dados de cada classe em cada 
    #                     conjunto. Em outras palavras, o conjunto de treinamento deve ser
    #                     formado por pTrain% dos dados da primeira classe, pTrain% dos dados da 
    #                     segunda classe e assim por diante. 
    #                   - a porcentagem de dados de teste para cada classe é igual a 
    #                     1-pTrain (parametro da funcao que contem a porcentagem de dados 
    #                     de treinamento)
    
    #Versão p/ N classes
    for classe in np.unique(target).tolist():
        idxClasse = np.argwhere(target == classe)
        idxTrain  = int(len(idxClasse) * pTrain)
        
        train_index += sum(idxClasse[:idxTrain].tolist(), []) 
        test_index  += sum(idxClasse[idxTrain:].tolist(), [])
        
    train_index = sorted(train_index)
    test_index  = sorted(test_index)
    ##################################################################################
    
    return train_index, test_index