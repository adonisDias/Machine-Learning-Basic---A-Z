# Apriori

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

# Training Apriori on the dataset
# Aqui precisamos definir os coeficientes de support e confidence. Estes valores podem variar dependendo do problema que precisamos resolver. Neste caso, não queremos os produtos que raramente são comprados.
# Support = 3 * 7 / 7500 produtos vendidos 3 vezes por dia vezes os dias da semana dividido pelo total dos produtos vendidos
# Condidence = primeiramente vamos estabelecer um valor padrão para confidence e diminuindo ao longo dos testes. Não queremos uma confidence muito alta, pois isso leva ao modelo estabelecer relações entre os produtos muito óbvias e queremos descobrir outras relações mais subjetivas
rules = apriori(data = dataset,
                parameter = list(support = 0.003, confidence = 0.8))
# no primeiro treinamento o modelo resultou em zero rules, pois uma confidence 0.8 é muito alta, para cada rule, ela precisa estar correta 4 vezes em cada 5 transações.

rules = apriori(data = dataset,
                parameter = list(support = 0.003, confidence = 0.4))

# Visualising the results
# step 4 - Sort the rules by decreasing lift
inspect(sort(rules, by = 'lift')[1:10])

# Visualising the results - nós podemos observar que os dez primeiros resultados trouxeram muitas cestas com os produtos mais comprados da loja, que possuem em suporte alto. Isso resulta num resultado insatisfatório para as regras, pois já são os produtos mais comprados.
# Por esse motivo, vamos reduzir novamente a confidence pela metade para gerar regras mais significativas.
rules = apriori(data = dataset,
                parameter = list(support = 0.003, confidence = 0.2))

inspect(sort(rules, by = 'lift')[1:10])

# Vamos aumentar o suporte para produtos comprados 4 vezes por dia pelo menos. 4*7/7500
rules = apriori(data = dataset,
                parameter = list(support = 0.004, confidence = 0.2))

inspect(sort(rules, by = 'lift')[1:10])