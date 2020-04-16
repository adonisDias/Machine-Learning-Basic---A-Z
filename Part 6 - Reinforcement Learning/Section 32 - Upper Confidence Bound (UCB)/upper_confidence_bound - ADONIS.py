# -*- coding: utf-8 -*-
"""
Created on Thu May 30 06:49:48 2019

@author: Adonis-Note
"""

# Upper Confidence Bound

#################################

# ver a definição do algoritmo nas anotações do curso

#################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Ads_CTR_Optimisation.csv') 

# Implementing UCB

import math

N = 10000 #total rounds
d = 10
ads_selected = [] #list with all rounds and the ad selected in this round
numbers_of_selections = [0] * d  #number of times the ad "i" was selected up to round "n"
sums_of_rewards = [0] * d  #number of rewards of the ad "i" up to round "n"
total_reward = [0]

for n in range(0, N):
    
    ad = 0
    max_upper_bound = 0    
    
    #ao final do primeiro round ad será igual a 0    
    for i in range(0, d): #loop para cada versão do anúncio
        
        if (numbers_of_selections[i] > 0): # na primeira vez esse procedimento não será executado, esse procedimento é executado no round "n" e na primeira computaçao estaremos no zero. Só será executado depois de já termos algumas informações sobre as médias. Neste caso depois dos 10 rounds. 
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i]) # math.log(n + 1)   aqui somamos 1 porque em python o índice inicia com zero
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400  #10 elevado na 400                
            
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i #índice do ad com o maior upper bound para ser armazenado na lista
            
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad] #busca no dataset se o ad selecionado no algoritmo é o ad realmente clicado no dataset (que possui o 1 no valor do índice)
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward #se o reward for 1, significa que o algoritmo selecionou corretamente o anúncio clicado pelo usuário, então ele adiciona na lista de anúncios o 1, que o anúncio foi selecionado corretamente, e 0 caso ele não for selecionado corretamente.
    total_reward = total_reward + reward #soma quantas vezes o algoritmo acertou.

######################

# ao final do algoritmo pode-se verificar a lista ads_selected. Ao final dessa lista haverá o índice do anúncio com a maior conversão, neste caso o anúncio com o índice 4.

######################    
    
# Visualising the results
plt.hist(ads_selected)    
plt.title('Histrogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()