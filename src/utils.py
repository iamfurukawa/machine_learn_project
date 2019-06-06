# -*- coding: utf-8 -*-

def normalize(X):
    mu = []
    sigma = []
    m, n = X.shape
    
    for i in range(0, n):
        mu.append( X[ : , i].mean() )
        sigma.append (X[ : , i].std(ddof=1) )    

    return (X - mu)/sigma, mu, sigma

def save(m, listWords):
    with open('./matriz.csv', 'w+', encoding='utf8') as stream:
        for word in listWords:
            stream.write(word)
            stream.write(',')
        stream.write('class')
        stream.write('\n')

        for x in m:
            stream.write('\n')
            for y in x:
                stream.write(str(y))
                stream.write(',')