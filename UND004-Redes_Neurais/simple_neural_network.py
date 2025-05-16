import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# Carregar os dados de entrada
text = np.loadtxt('data_simple_nn.txt')

# Separar os dados de seus rótulos
data = text[:, 0:2]
labels = text[:, 2:]

# Carregar os dados de entrada
plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel('Dimensão 1')
plt.ylabel('Dimensão 2')
plt.title('Dados de entrada')

# Valores máximo e mínimo para cada dimensão
dim1_min, dim1_max = data[:,0].min(), data[:,0].max()
dim2_min, dim2_max = data[:,1].min(), data[:,1].max()

# Definir o número de neurônios da arquitetura
num_output = labels.shape[1]

# Definir a rede neural de uma camada
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
nn = nl.net.newp([dim1, dim2], num_output)

# Treinar a rede neural
error_progress = nn.train(data, labels, epochs=100, show=20, lr=0.03)

# Plotar o progresso do treinamento
plt.figure()
plt.plot(error_progress)
plt.xlabel('Número de épocas')
plt.ylabel('Erro do treinamento')
plt.title('Progresso do erro do treinamento')
plt.grid()

plt.show()

# Executar o classificador usandos os dados de entrada
print('\nResultados do teste:')
data_test = [[0.4, 4.3], [4.4, 0.6], [4.7, 8.1]]
for item in data_test:
    print(item, '-->', nn.sim([item])[0])
