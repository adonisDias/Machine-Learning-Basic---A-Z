# Logistic Regression


# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

#Splitting the dataset into the Test set and the Training set
#install.packages('caTools')
library(caTools) #ou pode selecionar o checbox
set.seed(123) #para ter os mesmos resultados
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# scale � aplicado somente em n�meros, por�m ao usar o factor, as colunas 1 e 4 eram caracteres e depois trocadas para n�meros, por isso pegamos somente as colunas 2 e 3
training_set[, 1:2] = scale(training_set[, 1:2])
test_set[, 1:2] = scale(test_set[, 1:2])

# Applying Kernel PCA
#install.packages('kernlab')
library(kernlab)
kpca = kpca(~., #f�rmula com todas vari�veis do dataset
            data = training_set[-3], #training_set sem a vari�vel dependente
            kernel = 'rbfdot', #mais comum utilizado
            features = 2) 
training_set_pca = as.data.frame(predict(kpca, training_set))
training_set_pca$Purchased = training_set$Purchased#criando a coluna da vari�vel dependente
test_set_pca = as.data.frame(predict(kpca, test_set))
test_set_pca$Purchased = test_set$Purchased#criando a coluna da vari�vel dependente


# Fitting the Regression Model to the dataset
classifier = glm(formula = Purchased ~.,
                 family = binomial, #n�o explicou o que � binomial, mas para logistic regression ele � necess�rio
                 data = training_set_pca)

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set_pca[-3]) # response indica que o retorno deve ser em um vetor com todas as probabilidades. Test_Set - 3 para selecionar apenas as vari�veis independentes.
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Visualising the Training results
#install.packages('cowplot') # install.packages('ElemSatLearn')    ElemSatLearn n�o � suportada na vers�o 3.5 do R
library(cowplot)#library(ElemSatLearn)
set = training_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, 3],
     main = 'Logistic Regression (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


# Visualising the Test results
#install.packages('cowplot') # install.packages('ElemSatLearn')    ElemSatLearn n�o � suportada na vers�o 3.5 do R
#library(cowplot)#library(ElemSatLearn)
set = test_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, 3],
     main = 'Logistic Regression (Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))