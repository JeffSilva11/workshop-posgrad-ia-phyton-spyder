import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# Arquivo de entrada que contem os dados
input_file = 'data_singlevar_regr.txt' 

# Leitura dos dados
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Tamanho dos Conjuntos de teste e treinamento
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Dados de treinamento
X_train, y_train = X[:num_training], y[:num_training]

# Dados de teste
X_test, y_test = X[num_training:], y[num_training:]

# Criando o objeto responsável pelo linha de regressão
regressor = linear_model.LinearRegression()

# Treinando o modelo com os dados do conjunto de treinamento
regressor.fit(X_train, y_train)

# Prevendo a saída
y_test_pred = regressor.predict(X_test)

# Colocando os resultados no gráfico
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

# Calculando as métricas de desempenho
print("Desempenho da classificação por regressão linear simples:")
print("Erro absoluto médio = ", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Erro quadrático médio = ", round(sm.mean_squared_error(y_test, y_test_pred), 2)) 
print("R2 score = ", round(sm.r2_score(y_test, y_test_pred), 2))

# Modelo de persistência
output_model_file = 'model.pkl'

# Armazenando o modelo
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

# Carregando o modelo
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

# Executando previsão com os dados de teste
y_test_pred_new = regressor_model.predict(X_test)
print("\nNovo erro absoluto médio = ", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))

