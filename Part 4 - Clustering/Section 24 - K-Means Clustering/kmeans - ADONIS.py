# -*- coding: utf-8 -*-
"""
Created on Wed May 15 07:02:21 2019

@author: Adonis-Note
"""

# resetar todo o ambiente
# %reset - f

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the mail dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the Elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):   # o index 11 é excluído na iteração, não sei porque o python é assim
    kmeans = KMeans(n_clusters = i, 
                    init = 'k-means++', 
                    max_iter = 300,  # valor default para o número máximo de iterações é 300
                    n_init = 10, # número de vezes que o algoritmo será rodado com diferentes centroids, valor default é 10
                    random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()    

# Applying k-means to the mall dataset
kmeans = KMeans(n_clusters = 5, 
                init = 'k-means++', 
                max_iter = 300,  # valor default para o número máximo de iterações é 300
                n_init = 10, # número de vezes que o algoritmo será rodado com diferentes centroids, valor default é 10
                random_state = 0)
y_kmeans = kmeans.fit_predict(X)  # irá nos retornar para cada cliente, qual o seu respectivo cluster


# Visualising the Clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red'    , label = 'Cluster 1')  # s define o tamanho do ponto no gráfico
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue'   , label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green'  , label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan'   , label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of CLients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# Relabeling The Clusters - Visualising the Clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red'    , label = 'Careful')  # s define o tamanho do ponto no gráfico
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue'   , label = 'Standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green'  , label = 'Target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan'   , label = 'Careless')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of CLients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()