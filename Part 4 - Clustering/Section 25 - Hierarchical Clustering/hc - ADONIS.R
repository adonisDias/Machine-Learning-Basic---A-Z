# Hierarchical Clustering

# Importing the mall dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

# Using the dendrogram to find the optimal number of clusters
dendrogram = hclust(dist(X, method = 'euclidean'), method = 'ward.D')  # primeiro parâmetro "d" é a distance matrix of the dataset X, que é uma matriz que diz para cada par de customers, a euclidean distance entre os dois. Busca as coordenadas entre as variáveis x e y, e calcula a euclidean distance entre esses dois.
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean distances')

# Fitting hierarchical clustering to the mall dataset
hc = hclust(dist(X, method = 'euclidean'), method = 'ward.D')  # primeiro parâmetro "d" é a distance matrix of the dataset X, que é uma matriz que diz para cada par de customers, a euclidean distance entre os dois. Busca as coordenadas entre as variáveis x e y, e calcula a euclidean distance entre esses dois.
y_hc = cutree(hc, 5)  # método utilizado para gerar o vector com a informação de qual cluster cada customer pertence

# Visualising the clusters
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of Clients'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')