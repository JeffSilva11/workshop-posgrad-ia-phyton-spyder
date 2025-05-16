import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# Carregar os dados de entrada
text = np.loadtxt('data_perceptron.txt')

# Separar os dados de entrada e seus rótulos
data = text[:, :2]
labels = text[:, 2].reshape((text.shape[0], 1))

# Plotar os dados de entrada
plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel('Dimensão 1')
plt.ylabel('Dimensão 2')
plt.title('Dados de entrada')

# Definir os valores máximo e mínimo de cada dimensão
dim1_min, dim1_max, dim2_min, dim2_max = 0, 1, 0, 1

# Número de neurônios na cadama de saída
num_output = labels.shape[1]

# Definir um perceptron com duas entradas
# (existem duas dimensões nos dados de entrada)
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
perceptron = nl.net.newp([dim1, dim2], num_output)

# Treinar os perceptrons usando os dados
error_progress = perceptron.train(data, labels, epochs=100, show=20, lr=0.03)

# Plotar oprogresso do treinamento
plt.figure()
plt.plot(error_progress)
plt.xlabel('Número de épocas')
plt.ylabel('Erro do treinamento')
plt.title('Progresso do erro do treinamento')
plt.grid()

plt.show()
