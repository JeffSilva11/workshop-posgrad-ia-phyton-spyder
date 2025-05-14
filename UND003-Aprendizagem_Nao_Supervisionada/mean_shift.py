import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

# Carregar os dados de entrada
X = np.loadtxt('data_clustering.txt', delimiter=',')

# Definindo o bandwidth, parâmetro que afeta a convergência do algoritmo
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# Clusterização
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# Mostrar os valores dos centroides de cada cluster
cluster_centers = meanshift_model.cluster_centers_
print('\nCentroides dos clusters:\n', cluster_centers)

# Estimar o número de clusters
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print("\nNúmero de clusters no conjunto de dados =", num_clusters)

# Mostrar os dados e seus respectivos centroides
plt.figure()
markers = 'o*xvs'
for i, marker in zip(range(num_clusters), markers):
    # Plotar os pontos do conjunto de dados
    plt.scatter(X[labels==i, 0], X[labels==i, 1], marker=marker, color='black')

    # Plotar os centroides
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o', 
            markerfacecolor='red', markeredgecolor='red', 
            markersize=15)

plt.title('Clusters')
plt.show()
