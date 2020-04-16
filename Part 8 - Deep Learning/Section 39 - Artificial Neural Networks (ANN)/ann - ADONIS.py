# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 06:21:42 2019

@author: Adonis-Note
"""

# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Classifier Template

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values #(1:2) upper bound não é considerado. Nos modelos de machine learning, especialmente o regression, a variável independente precisa ser tradada como uma matriz, por esse motivo foi incluído 1:2 ao invés de apenas 1. Assim ele não resulta em um vetor, mas em uma matriz.
y = dataset.iloc[:, 13].values #python conta apartir do zero

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#aqui nós criamos uma dummy variable para evitar o dummy variable trap. Escolhemos a variável país porque ela possui mais categorias do que o gênero Female ou Male
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #retiro uma das dummy variables

#Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Feature Scaling - (quando basear-se na eucledean distance) - necessário para normalizar os valores das variáveis, assim uma variável não dominará a outra, a diferença entre os valores não serão grandes (eucledian distance)
#Aqui como o modelo necessitará de muitos cálculos computacionais vamos aplicar o feature scaling, também para nenhuma variable dominate another.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #no test set não precisamos fit somente transform

# Part 2 - Now let's make the ANN
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer.
classifier.add(Dense(output_dim = 6, #6 para hidden é a metade do total das entradas mais as saídas
                     init = 'uniform', #distribui igualmente os valores dos pesos no início
                     activation = 'relu',
                     input_dim = 11)) 
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, #6 para hidden é a metade do total das entradas mais as saídas
                     init = 'uniform', #distribui igualmente os valores dos pesos no início
                     activation = 'relu')) 
# Adding the output layer
classifier.add(Dense(output_dim = 1, #6 para hidden é a metade do total das entradas mais as saídas
                     init = 'uniform', #distribui igualmente os valores dos pesos no início
                     activation = 'sigmoid')) 

# Compiling the ANN
classifier.compile(optimizer = 'adam', #stochastic gradient descent
                   loss = 'binary_crossentropy', #logarithm loss function is used on sigmoid functions
                   metrics = ['accuracy']) #em geral é utilizada essa métrica mesmo

# Fitting the ANN to the Training set
classifier.fit(X_train,
               y_train,
               batch_size = 10,
               nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #se o valor for maior que 0.5 irá retornar true, do contrário irá retornar false

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,
                      y_pred)