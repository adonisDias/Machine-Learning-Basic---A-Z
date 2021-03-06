# ANN

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[ 4:14]

# Encoding the categorical variable as factor, . (the terms 'category' and 'enumerated type' are also used for factors).
# Necess�rio codificar como numeric factor porque a package de deep learning utilizada requer esse formato.
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain','Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))


#Splitting the dataset into the Test set and the Training set
#install.packages('caTools')
library(caTools) #ou pode selecionar o checbox
set.seed(123) #para ter os mesmos resultados
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting XGBoost to the Training set
# install.packages('xgboost')
library(xgboost)
classifier = xgboost(data = as.matrix(training_set[-11]),
                     label = training_set$Exited,
                     nrounds = 10)

# Applying k-Fold Cross Validation
# install.packages('caret')
library(caret)
folds = createFolds(training_set$Exited, k = 10)
cv = lapply(folds, function(x) {    #lapply - aplica uma fun��o que calcula a accuracy para cada elemento de uma lista, neste caso, folds. x � um dos folds da lista.
  training_fold = training_set[-x, ] #-x para buscar somente o training_set, pois o fold cont�m a posi��o de cada linha, assim aqui estamos pegando todas as linhas desconsiderando a vari�vel dependente, que ser� utilizada no test_fold.
  test_fold = training_set[x, ]
  classifier = xgboost(data = as.matrix(training_set[-11]),
                       label = training_set$Exited,
                       nrounds = 10)
  y_pred = predict(classifier, newdata = as.matrix(test_fold[-11]))
  y_pred = (y_pred >= 0.5)
  cm = table(test_fold[, 11], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))