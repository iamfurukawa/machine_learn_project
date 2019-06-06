# -*- coding: utf-8 -*-
#pVitoria = sum(Y==1)/len(Y) 
#pDerrota = sum(Y==0)/len(Y)

def calcularProbabilidades(X, Y):
    """
    CALCULARPROBABILIDADES Computa a probabilidade de ocorrencia de cada 
    atributo por rotulo possivel. A funcao retorna dois vetores de tamanho n
    (qtde de atributos), um para cada classe.
    
    CALCULARPROBABILIDADES(X, Y) calcula a probabilidade de ocorrencia de cada atributo em cada classe. 
    Cada vetor de saida tem dimensao (n x 1), sendo n a quantidade de atributos por amostra.
    """
    
    #  inicializa os vetores de probabilidades
    pAtrVitoria = np.zeros(X.shape[1])
    pAtrDerrota = np.zeros(X.shape[1])

    ########################## COMPLETE O CÓDIGO AQUI  ########################
    #  Instrucoes: Complete o codigo para encontrar a probabilidade de
    #                ocorrencia de um atributo para uma determinada classe.
    #                Ex.: para a classe 1 (vitoria), devera ser computada um
    #                vetor pAtrVitoria (n x 1) contendo n valores:
    #                P(Atributo1=1|Classe=1), ..., P(Atributo5=1|Classe=1), e o
    #                mesmo para a classe 0 (derrota):
    #                P(Atributo1=1|Classe=0), ..., P(Atributo5=1|Classe=0).
    # 
    for column in range (0, X.shape[1]):
        for line in range (0, X.shape[0]):
            if line in np.where(Y == 1)[0] and X[line][column] == 1:
                pAtrVitoria[column] += 1  
            elif line in np.where(Y == 0)[0] and X[line][column] == 1:
                pAtrDerrota[column] += 1
                
    pAtrVitoria /= len(np.where(Y == 1)[0])
    pAtrDerrota /= len(np.where(Y == 0)[0])
    ##########################################################################

    return pAtrVitoria, pAtrDerrota

