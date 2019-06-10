# -*- coding: utf-8 -*-

def normalize(X):
    mu = []
    sigma = []
    m, n = X.shape
    
    for i in range(0, n):
        mu.append( X[ : , i].mean() )
        sigma.append (X[ : , i].std(ddof=1) )    

    return (X - mu)/sigma, mu, sigma

def save(m, features, name):
    name_file = './{}.csv'.format(name)
    with open(name_file, 'w+', encoding='utf8') as stream:
        for word in matrix:
            stream.write(word)
            stream.write(',')
        stream.write('class')
        stream.write('\n')

        for x in m.tolist():
            stream.write('\n')
            for y in x:
                stream.write(str(y))
                stream.write(',')
                
def save_raw(m, name):
    name_file = './{}.csv'.format(name)
    with open(name_file, 'w+', encoding='utf8') as stream:
        for x in m:
            stream.write('\n')
            for y in x:
                stream.write(str(y))
                stream.write(',')