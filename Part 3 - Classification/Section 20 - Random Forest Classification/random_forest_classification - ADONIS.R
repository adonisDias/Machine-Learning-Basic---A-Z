# Random Forest Classification


# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

# Encoding the target feature (categorial variable) as factor, isso pe necessário para resvolver a questão do erro factor(0). (the terms 'category' and 'enumerated type' are also used for factors)
#dataset[,3] = factor(dataset[,3], levels = c(0, 1))
#se não utilizarmos o método factor, o modelo tenta realizar uma regressão e exibe a mensagem abaixo
#Warning message:
#  In randomForest.default(x = training_set[-3], y = training_set$Purchased,  :
#  The response has five or fewer unique values.  Are you sure you want to do regression?


#Splitting the dataset into the Test set and the Training set
#install.packages('caTools')
library(caTools) #ou pode selecionar o checbox
set.seed(123) #para ter os mesmos resultados
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

y_trainingset = training_set[,3]
y_testset     = test_set[,3]
training_set[,3] = factor(training_set[,3], levels = c(0, 1))
test_set[,3]     = factor(test_set[,3], levels = c(0, 1))



# Feature Scaling
# scale é aplicado somente em números, porém ao usar o factor, as colunas 1 e 4 eram caracteres e depois trocadas para números, por isso pegamos somente as colunas 2 e 3
training_set[, 1:2] = scale(training_set[, 1:2])
test_set[, 1:2] = scale(test_set[, 1:2])


# Fitting the Random Forest Classification Model to the dataset
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-3],
                          y = training_set$Purchased,
                          ntree = 10)


# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Visualising the Training results
#install.packages('cowplot') 
#install.packages('ElemStatLearn')   # ElemSatLearn não é suportada na versão 3.5 do R
#library(cowplot)#library(ElemSatLearn)
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class') # incluído type = 'class' para queu o y_pred não exiba as probabilidades, mas sim as classificações 0 ou 1
plot(y_trainingset,
     main = 'Random Forest Classification (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(-3:3), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(y_trainingset == 1, 'green4', 'red3'))


# Visualising the Test results
#install.packages('cowplot') # install.packages('ElemSatLearn')    ElemSatLearn não é suportada na versão 3.5 do R
#library(cowplot)#library(ElemSatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(y_testset,
     main = 'Random Forest Classification (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(y_testset == 1, 'green4', 'red3'))

