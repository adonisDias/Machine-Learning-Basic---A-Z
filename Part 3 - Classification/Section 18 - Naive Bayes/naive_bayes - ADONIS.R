# Naive Bayes


# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

# Encoding the target feature (categorial variable) as factor, isso pe necessário para resvolver a questão do erro factor(0)
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))


#Splitting the dataset into the Test set and the Training set
#install.packages('caTools')
library(caTools) #ou pode selecionar o checbox
set.seed(123) #para ter os mesmos resultados
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# scale é aplicado somente em números, porém ao usar o factor, as colunas 1 e 4 eram caracteres e depois trocadas para números, por isso pegamos somente as colunas 2 e 3
training_set[, 1:2] = scale(training_set[, 1:2])
test_set[, 1:2] = scale(test_set[, 1:2])


# Fitting the Naive Bayes Model to the dataset
# install.packages('e1071')
library(e1071)
classifier = naiveBayes(x = training_set[-3],
                        y = training_set$Purchased)  # nós precisamos retirar da matrix training_set os valores referentes à varipavel dependente. na definição da classe é indicado que o parâmetro x é "a numeric matrix, or a dataframe of categorical and/or numeric values"

# Predicting the Test set results 
y_pred = predict(classifier, newdata = test_set[-3])
#antes do tratamento de y_pred, se executarmos ele no console, teremos um resultado de factor(0) ou seja, um vetor de tamanho zero, porque o naive bayse não reconhece a categorical variable as a factor.

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Visualising the Training results
#install.packages('cowplot') # install.packages('ElemSatLearn')    ElemSatLearn não é suportada na versão 3.5 do R
library(cowplot)#library(ElemSatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, 3],
     main = 'Naive Bayes (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


# Visualising the Test results
#install.packages('cowplot') # install.packages('ElemSatLearn')    ElemSatLearn não é suportada na versão 3.5 do R
#library(cowplot)#library(ElemSatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, 3],
     main = 'Naive Bayes (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

