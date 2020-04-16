# Natural Language Processing

# Import the dataset
dataset = read.delim('Restaurant_Reviews.tsv',
                     quote ='',  # ignoring the quotes
                     stringsAsFactors = FALSE) # cada review não é uma entidade única, por isso não definimos como factors
dataset_original = read.delim('Restaurant_Reviews.tsv',
                              quote ='',  # ignoring the quotes
                              stringsAsFactors = FALSE) # cada review não é uma entidade única, por isso não definimos como factors

# Cleaning the texts
#install.packages('tm')
#install.packages('SnowballC')
library(tm) #corpus
library(SnowballC) #stopwords
corpus = VCorpus(VectorSource(dataset$Review))
corpus = tm_map(corpus,
                content_transformer(tolower))
corpus = tm_map(corpus,
                removeNumbers)
corpus = tm_map(corpus,
                removePunctuation)
corpus = tm_map(corpus,
                removeWords, stopwords())
corpus = tm_map(corpus,
                stemDocument)
corpus = tm_map(corpus,
                stripWhitespace)

# Creating the bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm,
                        0.999) #função para manter 99,9% das palavras mais frequentes. Não estamos olhando aqui para o corpus e contando as palavras mais frequentes desse corpus, mas sim, olhar para todas as colunas da matrix esparsa e manter 99% das colunas que tem mais "1".

# Classification

# Para classificação não usamos uma matrix, mas sim umm dataframe.
dataset = as.data.frame(as.matrix(dtm)) # utilizamos as.matrix para ter certeza que estaremos passando o tipo de matrix correta que o modelo está esperando.
dataset$liked = dataset_original$Liked# estamos criando uma nova coluna com a variável dependente


# Encoding the target feature (categorial variable) as factor, isso pe necessário para resvolver a questão do erro factor(0). (the terms 'category' and 'enumerated type' are also used for factors)
#dataset[,3] = factor(dataset[,3], levels = c(0, 1))
dataset$liked = factor(dataset$liked, levels = c(0, 1))
#se não utilizarmos o método factor, o modelo tenta realizar uma regressão e exibe a mensagem abaixo
#Warning message:
#  In randomForest.default(x = training_set[-3], y = training_set$Purchased,  :
#  The response has five or fewer unique values.  Are you sure you want to do regression?


#Splitting the dataset into the Test set and the Training set
#install.packages('caTools')
library(caTools) #ou pode selecionar o checbox
set.seed(123) #para ter os mesmos resultados
split = sample.split(dataset$liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#y_trainingset = training_set[,3]
#y_testset     = test_set[,3]
#training_set[,3] = factor(training_set[,3], levels = c(0, 1))
#test_set[,3]     = factor(test_set[,3], levels = c(0, 1))


# Feature Scaling
# scale é aplicado somente em números, porém ao usar o factor, as colunas 1 e 4 eram caracteres e depois trocadas para números, por isso pegamos somente as colunas 2 e 3
#training_set[, 1:2] = scale(training_set[, 1:2])
#test_set[, 1:2] = scale(test_set[, 1:2])


# Fitting the Random Forest Classification Model to the dataset
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],   #este índice 692 pode ser visualizado ao lado do nome do dataset no environment
                          y = training_set$liked,
                          ntree = 10)


# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
