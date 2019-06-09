# -*- coding: utf-8 -*-


#-----------------------------
# PREPARAÇÕES
#-----------------------------

# Sigmoid é a Regressão aplicada como hipótese
def sigmoid(z):

    z = 1/(1+np.exp(-z))
    
    return z

# Gera Gradiente do Sigmoid
def sigmoidGradient(z):
    
    g = np.zeros(z.shape)
    g = sigmoid(z) * (1 - sigmoid(z))

    return g

# Inicializador de pesos, fornecido no exercício
'''
    Inicializa aleatoriamente os pesos de uma camada usando 
    L_in (conexoes de entrada) e L_out (conexoes de saida).

    W sera definido como uma matriz de dimensoes [L_out, 1 + L_in]
    
    randomSeed: indica a semente para o gerador aleatorio
'''
def inicializaPesosAleatorios(L_in, L_out, randomSeed = None):
    epsilon_init = 0.12
    
    # se for fornecida uma semente para o gerador aleatorio
    if randomSeed is not None:
        W = np.random.RandomState(randomSeed).rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
        
    # se nao for fornecida uma semente para o gerador aleatorio
    else:
        W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
        
    return W

#-----------------------------
# FOWARD PROPAGATION
#-----------------------------
'''
Definião dos parâmetros
    thetas: junção dos thetas nas camadas
    tamanho_entrada: quantidade de neurônios na entrada
    tamanho_intermediaria: quantidade de neurônios na camada intermediária
    num_classes: número de classes
    X: amostras
    y: vetor contendo classes de cada amostra
    vLambda: parâmetro de regularização
'''
def funcaoCusto_reg(thetas, tamanho_entrada, tamanho_intermediaria, num_classes, X, y, vLambda):
    # Forma thetas para cada camada
    Theta1 = np.reshape( thetas[0:tamanho_intermediaria*(tamanho_entrada + 1)], (tamanho_intermediaria, tamanho_entrada+1) )
    Theta2 = np.reshape( thetas[ tamanho_intermediaria*(tamanho_entrada + 1):], (num_classes, tamanho_intermediaria+1) )
    
    # Inicialização de variáveis úteis e de retorno
    m = X.shape[0]
    
    Y = np.zeros( (m, num_labels) )
    for i in range (0, m):
        Y[i][y[i] - 1] = 1
        
    J = 0;
    
    # === Foward Propagation ===
    # insere bias = 1
    X_bias = np.insert(X, 0, 1, axis=1)
    
    # rede oculta
    a2 = sigmoid( np.dot (X_bias, Theta1.T) )
    
    # bias
    a2 = np.insert(a2, 0, 1, axis=1)    
    hip = sigmoid( np.dot(a2, Theta2.T) )
    
    # === Custo ===
    regula = (vLambda / (2 * m) ) * ( np.sum(Theta1.T[1:,:] ** 2) + np.sum(Theta2.T[1:,:] ** 2) )
    J = (1 / m) * np.sum( -Y * np.log(hip) - (1 - Y) * np.log(1 - hip) ) + regula
    
    return J

#-----------------------------
# BACK PROPAGATION
#-----------------------------
'''
Definião dos parâmetros
    thetas: junção dos thetas nas camadas
    tamanho_entrada: quantidade de neurônios na entrada
    tamanho_intermediaria: quantidade de neurônios na camada intermediária
    num_classes: número de classes
    X: amostras
    y: vetor contendo classes de cada amostra
    vLambda: parâmetro de regularização
'''
def funcaoCusto_backp_reg(thetas, tamanho_entrada, tamanho_intermediaria, num_classes, X, y, vLambda):

    # Forma thetas para cada camada
    Theta1 = np.reshape( thetas[0:tamanho_intermediaria*(tamanho_entrada + 1)], (tamanho_intermediaria, tamanho_entrada+1) )
    Theta2 = np.reshape( thetas[ tamanho_intermediaria*(tamanho_entrada + 1):], (num_classes, tamanho_intermediaria+1) )
    
    # Inicialização de variáveis úteis e de retorno
    m = X.shape[0]
    
    Y = np.zeros( (m, num_labels) )
    for i in range (0, m):
        Y[i][y[i] - 1] = 1
        
    J = 0;
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    
    # === Foward Propagation ===
    # insere bias = 1
    X_bias = np.insert(X, 0, 1, axis=1)
    
    # rede oculta
    a2 = sigmoid( np.dot (X_bias, Theta1.T) )
    
    # bias
    a2 = np.insert(a2, 0, 1, axis=1)    
    hip = sigmoid( np.dot(a2, Theta2.T) )
    
    # === Custo ===
    regula = (vLambda / (2 * m) ) * ( np.sum(Theta1.T[1:,:] ** 2) + np.sum(Theta2.T[1:,:] ** 2) )
    J = (1 / m) * np.sum( -Y * np.log(hip) - (1 - Y) * np.log(1 - hip) ) + regula
    
    #--------Backpropagation--------
    
    # Calculando Sigmas
    sigma3 = hip - Y
    sigma2 = np.dot(sigma3, Theta2[:,1:]) * sigmoidGradient( np.dot (X_bias, Theta1.T) )
    
    # Calculando Thetas
    Theta2_grad = Theta2_grad + np.dot(sigma3.T, a2) / m
    Theta1_grad = Theta1_grad + np.dot(sigma2.T, X_bias) /m 
    
    Theta2_grad[:,1:] = Theta2_grad[:,1:] + (vLambda / (m) ) * (Theta2[:,1:])
    Theta1_grad[:,1:] = Theta1_grad[:,1:] + (vLambda / (m) ) * (Theta1[:,1:])

    # Junta os gradientes
    grad = np.concatenate([np.ravel(Theta1_grad), np.ravel(Theta2_grad)])

    return J, grad

