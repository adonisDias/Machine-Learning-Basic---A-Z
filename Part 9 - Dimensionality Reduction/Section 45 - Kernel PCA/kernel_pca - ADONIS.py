# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 06:35:05 2019

@author: Adonis-Note
"""

#KERNEL PCA

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, [2,3]].values #(1:2) upper bound não é considerado. Nos modelos de machine learning, especialmente o regression, a variável independente precisa ser tradada como uma matriz, por esse motivo foi incluído 1:2 ao invés de apenas 1. Assim ele não resulta em um vetor, mas em uma matriz.
y = dataset.iloc[:, 4].values #python conta apartir do zero


#Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Feature Scaling - necessário para normalizar os valores das variáveis, assim uma variável não dominará a outra, a diferença entre os valores não serão grandes (eucledian distance)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #no test set não precisamos fit somente transform

# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')  #rbf is the most commum used and it works very well 
X_train = kpca.fit_transform(X_train)
X_test  = kpca.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,
                      y_pred)

# Visualising the Training results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train #apenas para facilitar a troca de variáveis
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), #aqui montamos um novo grid de valores com as margens maiores em relação aos valores reais. Desse modo os valores reais não ficarão posicionados nos limites do gráfico. Por isso diminuímos 1 e somamos 1.
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green'))) #aqui usamos a regressão para classificar todos os pontos do gráfico (taxa de pixel de 0.01), assim o gráfico fica completamente dividido em red e green.
plt.xlim(X1.min(), X1.max()) #delimitamos os limites do gráfico.
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j) #plotamos os pontos referentes aos valores reais do dataset.
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()    

# Visualising the Test results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()  

