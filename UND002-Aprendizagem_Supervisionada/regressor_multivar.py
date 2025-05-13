import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

# Arquivo de entrada que contém os dados
input_file = 'data_multivar_regr.txt'

# Carrega os dados do arquivo de entrada
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Divide os dados em conjuntos de treinamento e teste
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Dados de treinamento
X_train, y_train = X[:num_training], y[:num_training]

# Dados de teste
X_test, y_test = X[num_training:], y[num_training:]

# Cria o modelo de regressão linear
linear_regressor = linear_model.LinearRegression()

# Treina o modelo utilizando os dados de treinamento
linear_regressor.fit(X_train, y_train)

# Predição da saída
y_test_pred = linear_regressor.predict(X_test)

# Métricas de desempenho
print("Desempenho da classificação por regressão linear múltipla:")
print("Erro absoluto médio = ", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Erro quadrático médio = ", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Mediana absoluta média = ", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explicação da variância = ", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score = ", round(sm.r2_score(y_test, y_test_pred), 2))

# Regressão polinomial
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.fit_transform(datapoint)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
print("\nRegressão linear:\n", linear_regressor.predict(datapoint))
print("\nRegressão polinomial:\n", poly_linear_model.predict(poly_datapoint))

