# Polynomial Regression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

#Splitting the dataset into the Test set and the Training set
#install.packages('caTools')
# library(caTools) #ou pode selecionar o checbox
# set.seed(123) #para ter os mesmos resultados
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_Set = subset(dataset, split == TRUE)
# test_Set = subset(dataset, split == FALSE)

# Feature Scaling
# scale é aplicado somente em números, porém ao usar o factor, as colunas 1 e 4 eram caracteres e depois trocadas para números, por isso pegamos somente as colunas 2 e 3
# training_Set[, 2:3] = scale(training_Set[, 2:3])
# training_Set[, 2:3] = scale(training_Set[, 2:3])

# Fitting Linear Regression to the dataset
lin_reg = lm(formula = Salary ~ .,
             data    = dataset)


# Fitting Polynomial Regression to the dataset
# Aqui criamos as demais variáveis independentes do modelo Polynomail, na verdade ao final temos um modelo Multiple Linear Regression
dataset$Level2 = dataset$Level^2  #na potência 2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,
              data    = dataset)

# Visualising the Linear Regression results
# install.packages('ggplot2') já estava instalado
library(ggplot2)

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Linear Regression)') +
  xlab('Level') +
  ylab('Salary')
  

# Visualising the Polynomial Regression results

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

# Predicting a new result with Linear Regression
y_pred = predict(lin_reg, data.frame(Level = 6.5))

# Predicting a new result with Polynomial Regression
y_pred = predict(poly_reg, data.frame(Level = 6.5,
                                      Level2 = 6.5^2,
                                      Level3 = 6.5^3,
                                      Level4 = 6.5^4))