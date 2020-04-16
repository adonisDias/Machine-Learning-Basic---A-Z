# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 07:03:39 2019

@author: Adonis-Note
"""

# Natural Language Processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)  #including the quoting = 3 the double quotes " will be ignored

# Cleaning the texts
import re #library to clean texts
import nltk #library para limpar as palavras desnecessárias para o algoritmo
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# pode ser digitado no console para olhar o primeiro review dataset['Review'][0]
corpus = []  # corpus é um conjunto de textos do mesmo tipo
for i in range(0, 1000): # upper bound is excluded

    review = re.sub('[^a-zA-Z]',
                    ' ', # substituir por um espaço
                    dataset['Review'][i])  # o ^a-z indica que não queremos remover as palavras
    review = review.lower()
    review = review.split() # cria uma lista com todas as palavras do texto
    ps = PorterStemmer()  # keep just the roots of the words
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # criamos uma "set" das palavras porque em python é mais fácil fazer um loop numa set do que numa lista
    review = ' '.join(review)  # juntar as palavras novamente
    corpus.append(review)
    
# Creating the Bag of words model
# vamos criar uma sparse matrix com cada linha correspondendo a um review e cada coluna correspondendo a cada palavra encontrada em todos os reviews.    
# The bag of words model is the sparse matrix itself
# O objetivo de criar essa matriz esparsa é para que possamos obter as variáveis independentes e, para posteriormente, aplicar um modelo de classificação nesses dados. Desse modo poderemos classificar o review em positivo ou negativo.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray() # X é a sparse matrix
y = dataset.iloc[:, 1].values # iloc para pegar o índice da coluna

# Modelos comuns para natural language processing is naive bayes, decision trees e random forest

#Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# não é necessário porque basicamente aqui temos apenas 0, 1, 3 etc..
#Feature Scaling - necessário para normalizar os valores das variáveis, assim uma variável não dominará a outra, a diferença entre os valores não serão grandes (eucledian distance)
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test) #no test set não precisamos fit somente transform

#######################################
# NAIVE BAYES   ---> Escolhido o melhor modelo, f1 score mais alto

# Fitting the Classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,
                      y_pred)
# 0-0 = 55   0-1=42
# 1-0 = 12   1-1=91     

#Accuracy = (TP + TN) / (TP + TN + FP + FN)
#Accuracy = (91 + 55) / (91 + 55 + 42 + 12) = 0.73
#Precision = TP / (TP + FP)
#Precision = 91 / (91 + 42) = 0.6842105263157895
#Recall = TP / (TP + FN)
#Recall = 91 / (91 + 12) = 0.883495145631068
#F1 Score = 2 * Precision * Recall / (Precision + Recall)
#F1 Score = 2 * 0.6842105263157895 * 0.883495145631068 / (0.6842105263157895 + 0.883495145631068)
#         = 0.7711864406779663
################################################

######################################
# DECISION TREES

# Fitting the Classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', # A entropia é normalmente usada na teoria da informação para medir a pureza ou impureza de um determinado conjunto. A pergunta que ela responde é: O quanto diferente/iguais esses elementos são entre si. Se a entropia for igual a zero, significa que os dados são muito homogênios (semelhantes))
                                    random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,
                      y_pred)
# 0-0 = 74   0-1=23
# 1-0 = 35   1-1=68     

#Accuracy = (TP + TN) / (TP + TN + FP + FN)
#Accuracy = (68 + 74) / (68 + 74 + 23 + 35) = 0.71
#Precision = TP / (TP + FP)
#Precision = 68 / (68 + 23) = 0.7391304347826087‬
#Recall = TP / (TP + FN)
#Recall = 68 / (68 + 35) = 0.7311827956989247
#F1 Score = 2 * Precision * Recall / (Precision + Recall)
#F1 Score = 2 * 0.7391304347826087‬ * 0.7311827956989247 / (0.7391304347826087‬ + 0.7311827956989247)
#         = 0.7351345179254472
#################################################

###################################
#RANDOM FOREST

# Fitting the Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10,
                                    criterion = 'entropy',  # entropy avalia a qualidade de um slipt e na física ele mede a desordem, então quanto maior a entropy, mais as suas partículas estão em desordem. O ganho de informação aqui é medido entre a diferença da entropy do parent node com a entropy do child node.
                                    random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,
                      y_pred)
# 0-0 = 87   0-1=10
# 1-0 = 46   1-1=57

#Accuracy = (TP + TN) / (TP + TN + FP + FN)
#Accuracy = (57 + 87) / (57 + 87 + 10 + 46) = 0.72
#Precision = TP / (TP + FP)
#Precision = 57 / (57 + 10) = 0.8507462686567164
#Recall = TP / (TP + FN)
#Recall = 57 / (57 + 46) = 0.5533980582524272
#F1 Score = 2 * Precision * Recall / (Precision + Recall)
#F1 Score = 2 * 0.8507462686567164 * 0.5533980582524272 / (0.8507462686567164 + 0.5533980582524272)
#         = 0.6705882352941177
##################################################