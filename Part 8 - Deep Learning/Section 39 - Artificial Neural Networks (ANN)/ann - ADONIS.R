# ANN

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[ 4:14]

# Encoding the categorical variable as factor, . (the terms 'category' and 'enumerated type' are also used for factors).
# Necessário codificar como numeric factor porque a package de deep learning utilizada requer esse formato.
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

# Feature Scaling
# scale é necessário neste modelo porque haverão higly intensive computations and besides the package requires it
training_set[, -11] = scale(training_set[, -11])
test_set[, -11] = scale(test_set[, -11])


# Fitting ANN to the Training set
# A primeira razão para utilizar essa package é porque ela é open source, desse modo, permitindo a conexão com o sistema computacional tornando-a mais eficiente.
# Segunda razão é por ela disponibilizar diversas opções de modelos e parametrizações
# Terceiro, ela possui uma opção para setar hyperparameters para tentar otimizar o modelo.
#install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)  #conectando no sistema para ser mais eficiente. -1 indica que todos os cores disponíveis no sistema serão utilizados.
classifier = h2o.deeplearning(y = 'Exited',
                              training_frame = as.h2o(training_set),
                              activation = 'Rectifier',
                              hidden = c(6,6),
                              epochs = 100,
                              train_samples_per_iteration = -2) #paraneter tunning, this is why h2o is a very good library

# Predicting the Test set results
prob_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-11]))
y_pred = (prob_pred > 0.5) # will return a boolean value
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)

h2o.shutdown()