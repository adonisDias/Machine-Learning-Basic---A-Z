# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 06:28:08 2019

@author: Adonis-Note
"""

# Thompson Sampling

#################################

# ver a definição do algoritmo nas anotações do curso

#################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

dataset = pd.read_csv('Ads_CTR_Optimisation.csv') 

N = 10000 #total rounds
d = 10
ads_selected = [] #list with all rounds and the ad selected in this round
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = [0]

for n in range(0, N):
    
    ad = 0
    max_random = 0    
    
    # ao final do primeiro round ad será igual a 0    
    for i in range(0, d): #loop para cada versão do anúncio
        
        # STEP 1 and 2: take the random draws
        # Correspond to the different random draws taken from beta distribution. 
        # betavariate nos dará random draws do beta distribution conforme os parameters que escolhemos
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1) # o + 1 está na fórmula do algoritmo thompson sampling
            
        if random_beta > max_random:
            max_random = random_beta
            ad = i #índice do ad com o maior upper bound para ser armazenado na lista
            
    ads_selected.append(ad)    
    reward = dataset.values[n, ad] #busca no dataset se o ad selecionado no algoritmo é o ad realmente clicado no dataset (que possui o 1 no valor do índice)    
         
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
        
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