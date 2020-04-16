# LDA
# Logistic Regression


# Importing the dataset
dataset = read.csv('Wine.csv')

#Splitting the dataset into the Test set and the Training set
#install.packages('caTools')
library(caTools) #ou pode selecionar o checbox
set.seed(123) #para ter os mesmos resultados
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# scale é aplicado somente em números, porém ao usar o factor, as colunas 1 e 4 eram caracteres e depois trocadas para números, por isso pegamos somente as colunas 2 e 3
training_set[, -14] = scale(training_set[, -14])
test_set[, -14] = scale(test_set[, -14])

# Applying LDA
#install.packages('MASS')
library(MASS)
lda = lda(formula = Customer_Segment ~ .,
          data = training_set)
# new 2 linear discriminant
training_set = as.data.frame(predict(lda,
                             training_set))
training_set = training_set[c(5,6,1)] #aqui recriamos o training_Set com a ordem correta das colunas, pois antes a variável dependente estava na primeira coluna. Em R c() indica um vector
test_set = as.data.frame(predict(lda,
                                 test_set))
test_set = test_set[c(5,6,1)] #aqui recriamos o training_Set com a ordem correta das colunas, pois antes a variável dependente estava na primeira coluna. Em R c() indica um vector



# Fitting the SVM to the dataset
library(e1071)
classifier = svm(formula = class ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Visualising the Training results
#install.packages('cowplot') # install.packages('ElemSatLearn')    ElemSatLearn não é suportada na versão 3.5 do R
library(cowplot)#library(ElemSatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('x.LD1', 'x.LD2') #essa linha sempre espera o nome das variáveis independentes do dataset
y_grid = predict(classifier, newdata = grid_set)
plot(set[, 3],
     main = 'SVM (Training set)',
     xlab = 'LD1', ylab = 'LD2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))


# Visualising the Test results
#install.packages('cowplot') # install.packages('ElemSatLearn')    ElemSatLearn não é suportada na versão 3.5 do R
#library(cowplot)#library(ElemSatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('x.LD1', 'x.LD2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, 3],
     main = 'SVM (Test set)',
     xlab = 'LD1', ylab = 'LD1',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))
