import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

# Ler os dados do arquivo
X = np.loadtxt('data_quality.txt', delimiter=',')

# Plotar os dados de entrada
plt.figure()
plt.scatter(X[:,0], X[:,1], color='black', s=80, marker='o', facecolors='none')
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('Dados de entrada')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# Inicializar as variáveis
scores = []
values = np.arange(2, 10)

# Realiza as interações dentro da faixa determinada
for num_clusters in values:
    # Treinar o modelo de clusterização KMeans
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(X)
    score = metrics.silhouette_score(X, kmeans.labels_, 
                metric='euclidean', sample_size=len(X))

    print("\nNúmero de clusters =", num_clusters)
    print("Silhouette score =", score)
                    
    scores.append(score)

# Plotar o silhouette scores
plt.figure()
plt.bar(values, scores, width=0.7, color='black', align='center')
plt.title('Silhouette score vs Número de clusters')

# Determinar o melhor número de clusters
num_clusters = np.argmax(scores) + values[0]
print('\nNúmero ótimo de clusters =', num_clusters)

plt.show()
