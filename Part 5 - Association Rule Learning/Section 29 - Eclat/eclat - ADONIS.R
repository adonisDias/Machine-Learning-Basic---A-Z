# ECLAT

# Data Preprocessing
#install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)

# Criar a sparse matrix
dataset = read.transactions('Market_Basket_Optimisation.csv',
                            sep = ',',
                            rm.duplicates = TRUE) # rm.duplicate ir� remover a duplicidade nos registros, uma compra que o caixa tenha passado duas vezes o mesmo produto.
summary(dataset) # exibe informa��es sobre o dataset

itemFrequencyPlot(dataset, topN = 10)

# Training ECLAT on the dataset

rules = eclat(data = dataset,
              parameter = list(support = 0.003,
              minlen = 2)) # O modelo ECLAT retornar� os conjuntos de produtos que s�o mais comprados em conjunto. Sendo assim adicionamos o par�metro minlen, ele far� com que o modelo retorne, ao menos, pares de itens mais comprados.


# Visualising the results
inspect(sort(rules, by = 'support')[1:10])

