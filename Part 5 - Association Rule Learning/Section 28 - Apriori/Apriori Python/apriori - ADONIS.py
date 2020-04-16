# -*- coding: utf-8 -*-
"""
Created on Fri May 24 06:45:07 2019

@author: Adonis-Note
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
#cada linha do dataset é um resumo das compras dos clientes durante uma semana

# Vamos criar uma lista de listas
transactions = []
for i in range(0, 7501):  # zero é incluído e o 7501 é excluído
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
    
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions
                , min_support = 0.003
                , min_confidence = 0.2
                , min_lift = 3  
                , min_length = 2) # Rules compose with at least two products
#min_support = 3 * 7 / 7501 = produto comprado 3 vezes por dia.

# Visualising the results
# Em R nós ordenamos os resultados pelo lift (muito bom critério para ordenar as rules) mas em python faremos diferente. As rules encontradas por este modelo apriori já são ordenadas pela relevância(combinação do support, confidence and lift)
results = list(rules)

# Readable Results

results_list = []

for i in range(len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) +

                        '\nSUPPORT:\t' + str(results[i][1]) +

                        '\nCONFIDENCE:\t' + str(results[i][2][0][2]) +

                        '\nLIFT:\t' + str(results[i][2][0][3]))