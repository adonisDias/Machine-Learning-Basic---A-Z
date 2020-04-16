# ECLAT

# Data Preprocessing
#install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)

# Criar a sparse matrix
dataset = read.transactions('Market_Basket_Optimisation.csv',
                            sep = ',',
                            rm.duplicates = TRUE) # rm.duplicate irá remover a duplicidade nos registros, uma compra que o caixa tenha passado duas vezes o mesmo produto.
summary(dataset) # exibe informações sobre o dataset

itemFrequencyPlot(dataset, topN = 10)

# Training ECLAT on the dataset

rules = eclat(data = dataset,
              parameter = list(support = 0.003,
              minlen = 2)) # O modelo ECLAT retornará os conjuntos de produtos que são mais comprados em conjunto. Sendo assim adicionamos o parâmetro minlen, ele fará com que o modelo retorne, ao menos, pares de itens mais comprados.


# Visualising the results
inspect(sort(rules, by = 'support')[1:10])

