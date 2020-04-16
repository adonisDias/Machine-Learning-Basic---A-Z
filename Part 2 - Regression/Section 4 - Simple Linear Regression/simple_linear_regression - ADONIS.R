# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Salary_Data.csv')
#dataset = dataset[, 2:3]

#Splitting the dataset into the Test set and the Training set
#install.packages('caTools')
library(caTools) #ou pode selecionar o checbox
set.seed(123) #para ter os mesmos resultados
split = sample.split(dataset$Salary, SplitRatio = 2/3) #salary aqui é a variável dependente
training_Set = subset(dataset, split == TRUE)
test_Set = subset(dataset, split == FALSE)

# ATENCAO a package utilizada aqui no R já aplica o feature scaling, então não há necessidade de fazê-lo
# Feature Scaling
# scale é aplicado somente em números, porém ao usar o factor, as colunas 1 e 4 eram caracteres e depois trocadas para números, por isso pegamos somente as colunas 2 e 3
# training_Set[, 2:3] = scale(training_Set[, 2:3])
# training_Set[, 2:3] = scale(training_Set[, 2:3])

#Fitting Simple Linear Regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_Set)

#digitar no console " summary(regressor) " isso mostrará estatísticas sobre o modelo treinado. As três estrelas mostradas nos coeficientes indicam uma alta correlação estatística entre os dados. Poderia ser nenhuma estrela e no máximo três.
#o " Pr(>|t|) " indica também se há correlação estre os dados. Maior que 5% menos correlação e menor que 5% mais correlação.


# Predicting the Test set results
y_pred = predict(regressor, newdata = test_Set)

# Visualizing the Training set results
#install.packages('ggplot2')
#library(ggplot2)    marca esta package para ser usada
ggplot() +
  geom_point(aes(x = training_Set$YearsExperience, y = training_Set$Salary),
             colour = 'red') + #aes utilizada para especificar os valores de x e y no gráfico
  geom_line(aes(x = training_Set$YearsExperience, y = predict(regressor, newdata = training_Set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')


# Visualizing the Test set results
ggplot() +
  geom_point(aes(x = test_Set$YearsExperience, y = test_Set$Salary),
             colour = 'red') + #aes utilizada para especificar os valores de x e y no gráfico
  geom_line(aes(x = training_Set$YearsExperience, y = predict(regressor, newdata = training_Set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of experience') +
  ylab('Salary')