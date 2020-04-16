# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('50_Startups.csv')
#dataset = dataset[, 2:3]

#Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

#Splitting the dataset into the Test set and the Training set
#install.packages('caTools')
library(caTools) #ou pode selecionar o checbox
set.seed(123) #para ter os mesmos resultados
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_Set = subset(dataset, split == TRUE)
test_Set = subset(dataset, split == FALSE)

# Feature Scaling
# scale é aplicado somente em números, porém ao usar o factor, as colunas 1 e 4 eram caracteres e depois trocadas para números, por isso pegamos somente as colunas 2 e 3
# training_Set[, 2:3] = scale(training_Set[, 2:3])
# training_Set[, 2:3] = scale(training_Set[, 2:3])

# Fitting Multiple Linear Regression to the Training set
regressor = lm(formula = Profit ~ (R.D.Spend + Administration + Marketing.Spend + State),   #as variáveis podem ser substituídas por um único "ponto" indicando todas. aqui a fórmula é Profit será a combinação linear das demais variáveis independentes
               data = training_Set)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_Set)

# Building the optimal model using Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)

summary(regressor)

#Step 4 - eliminated the state variable because it is not significant for the model
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)

summary(regressor)

#Step 4 - eliminated the Administration variable because it is not significant for the model
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)

summary(regressor)
# decide deixar marketing.spend pois o seu pValue era muito próximo ao significant level (0,06 ~ 0,05)


# Automatic algorithm backward elimination
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)
