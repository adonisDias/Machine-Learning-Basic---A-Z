# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Data.csv')
#dataset = dataset[, 2:3]

#Splitting the dataset into the Test set and the Training set
#install.packages('caTools')
library(caTools) #ou pode selecionar o checbox
set.seed(123) #para ter os mesmos resultados
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_Set = subset(dataset, split == TRUE)
test_Set = subset(dataset, split == FALSE)

# Feature Scaling
# scale é aplicado somente em números, porém ao usar o factor, as colunas 1 e 4 eram caracteres e depois trocadas para números, por isso pegamos somente as colunas 2 e 3
# training_Set[, 2:3] = scale(training_Set[, 2:3])
# training_Set[, 2:3] = scale(training_Set[, 2:3])