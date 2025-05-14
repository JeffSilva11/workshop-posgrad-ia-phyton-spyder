import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Carregar os dados
X = np.loadtxt('data_clustering.txt', delimiter=',')
num_clusters = 5

# Plotar os dados de entrada
plt.figure()
plt.scatter(X[:,0], X[:,1], marker='o', facecolors='none', edgecolors='black', s=80)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('Dados de entrada')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# Criar o objeto que representa o algoritmo
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)

# Treinar o modelo de clusterização
kmeans.fit(X)

# Tamanho do grid
step_size = 0.01

# Definir os pontos do grid para facilitar a visualização dos clusters
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size), 
        np.arange(y_min, y_max, step_size))

# Prever o cluster mais próximo para item do conjunto de dados
output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

# Definir cores diferentes para os clusters 
output = output.reshape(x_vals.shape)
plt.figure()
plt.clf()
plt.imshow(output, interpolation='nearest',
           extent=(x_vals.min(), x_vals.max(), 
               y_vals.min(), y_vals.max()),
           cmap=plt.cm.Paired, 
           aspect='auto', 
           origin='lower')

# sobreposição dos pontos de entrada
plt.scatter(X[:,0], X[:,1], marker='o', facecolors='none', 
        edgecolors='black', s=80)

# Plotar as regiões centrais dos clusters
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:,0], cluster_centers[:,1], 
        marker='o', s=210, linewidths=4, color='black', 
        zorder=12, facecolors='black')

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('Limites de cada cluster')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
