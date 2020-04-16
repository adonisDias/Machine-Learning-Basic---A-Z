# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 06:42:19 2019

@author: Adonis-Note
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values #python conta apartir do zero

#Enconding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])  #[0] indica o índice da coluna
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X = X[:, 1:]  #aqui estamos pegando todas as linhas, porém as colunas iniciando em índice 1 até infinito (:). As libraries do python geralmente tratam a DVT, mas neste caso foi colocado manualmente para mostrar no exemplo. Nos modelos sempre desconsideramos uma das Dummy Variables.

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#não necessário o Feature Scaling porque a library do python irá tratar isso
#Feature Scaling - necessário para normalizar os valores das variáveis, assim uma variável não dominará a outra, a diferença entre os valores não serão grandes (eucledian distance)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #no test set não precisamos fit somente transform"""


#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #regressor é objeto da classe LinearRegression
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
#no modelo anterior a constante B0 (B zero) incluída pela library. Porém nesta library que vamos utilizar para avaliar nosso modelo essa constante não é incluída e por isso vamos fazer manualmente. Neste procedimento vamos deixar somente as variáveis independentes que realmente afetam o resultado final do modelo de predição.
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1 ) # adiciona a constante B0 no modelo com uma coluna de números 1, no qual será a primeira coluna de X. O parâmetro axis indica se é coluna ou linha
X_opt = X[:, [0, 1, 2, 3, 4, 5]]  #Step 1 - aqui pegamos todas as colunas das variáveis independentes uma a uma
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #Step 2 - OLS simple ordinary least squares é, simplesmente, o próprio multiple linar regression. Se olhar na especificação desta classe, poderemos verificar que no parâmetro exog a coluna de números "1" deve ser incluída pelo usuário
regressor_OLS.summary() #mostrará as estatísticas do modelo
#Step 4 - removemos o índice 2

#Step 3
#avaliar a coluna P>|t|, se for o maior valor em relação ao significant level (0.05), deve ser excluído do modelo
X_opt = X[:, [0, 1, 3, 4, 5]]  #eliminamos o índice 2
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #Step 2 - OLS simple ordinary least squares é, simplesmente, o próprio multiple linar regression. Se olhar na especificação desta classe, poderemos verificar que no parâmetro exog a coluna de números "1" deve ser incluída pelo usuário
regressor_OLS.summary() #mostrará as estatísticas do modelo
#Step 4 - removemos o índice 1

#Step 3
#avaliar a coluna P>|t|, se for o maior valor em relação ao significant level (0.05), deve ser excluído do modelo
X_opt = X[:, [0, 3, 4, 5]]  #eliminamos o índice 2
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #Step 2 - OLS simple ordinary least squares é, simplesmente, o próprio multiple linar regression. Se olhar na especificação desta classe, poderemos verificar que no parâmetro exog a coluna de números "1" deve ser incluída pelo usuário
regressor_OLS.summary() #mostrará as estatísticas do modelo
#Step 4 - removemos o índice 4

#Step 3
#avaliar a coluna P>|t|, se for o maior valor em relação ao significant level (0.05), deve ser excluído do modelo
X_opt = X[:, [0, 3, 5]]  #eliminamos o índice 2
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #Step 2 - OLS simple ordinary least squares é, simplesmente, o próprio multiple linar regression. Se olhar na especificação desta classe, poderemos verificar que no parâmetro exog a coluna de números "1" deve ser incluída pelo usuário
regressor_OLS.summary() #mostrará as estatísticas do modelo
#Step 4 - removemos o índice 5

X_opt = X[:, [0, 3]]  #eliminamos o índice 2
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #Step 2 - OLS simple ordinary least squares é, simplesmente, o próprio multiple linar regression. Se olhar na especificação desta classe, poderemos verificar que no parâmetro exog a coluna de números "1" deve ser incluída pelo usuário
regressor_OLS.summary() #mostrará as estatísticas do modelo


#-------AUTOMATIC BACKPROPAGATION ALTORITHM-------------


#Backward Elimination with p-values only
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)


#Backward Elimination with p-values and Adjusted R Squared:
import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)