import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# Gerar alguns dados de treinamento
min_val = -15
max_val = 15
num_points = 130
x = np.linspace(min_val, max_val, num_points)
y = 3 * np.square(x) + 5
y /= np.linalg.norm(y)

# Criar os dados e seus rótulos
data = x.reshape(num_points, 1)
labels = y.reshape(num_points, 1)

# Plotar os dados de entrada
plt.figure()
plt.scatter(data, labels)
plt.xlabel('Dimensão 1')
plt.ylabel('Dimensão 2')
plt.title('Dados de entrada')

# Definir uma rede neural multi-camada com duas camadas ocultas
# A primeira camada oculta possui 10 neurônios
# A segunda camada oculta possui seis neurônios
# A camada de saída possui um neurônio
nn = nl.net.newff([[min_val, max_val]], [10, 6, 1])

# Alterar o algoritmo de treinamento para o algoritmo de descida do gradiente
nn.trainf = nl.train.train_gd

# Treinar a rede neural
error_progress = nn.train(data, labels, epochs=2000, show=100, goal=0.01)

# Executar a rede neural sobre o conjunto de dados de treinamento
output = nn.sim(data)
y_pred = output.reshape(num_points)

# Plotar o erro do treinamento
plt.figure()
plt.plot(error_progress)
plt.xlabel('Número de épocas')
plt.ylabel('Erro')
plt.title('Progresso do erro do treinamento')

# Plotar a saída
x_dense = np.linspace(min_val, max_val, num_points * 2)
y_dense_pred = nn.sim(x_dense.reshape(x_dense.size,1)).reshape(x_dense.size)

plt.figure()
plt.plot(x_dense, y_dense_pred, '-', x, y, '.', x, y_pred, 'p')
plt.title('Atual vs Previsão')

plt.show()
