import datetime
import json

import numpy as np
from sklearn import covariance, cluster

import yfinance as yf

# Arquivo de entrada contendo os símbolos
input_file = 'company_symbol_mapping.json'

# Carregar os símbolos (nomes das empresas)
with open(input_file, 'r') as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

# Carregar os dados históricos das ações 
start_date = datetime.datetime(2019, 1, 1)
end_date = datetime.datetime(2019, 1, 31)
quotes = [yf.Ticker(symbol).history(start=start_date, end=end_date) 
                for symbol in symbols]

# extrair dados da abertura e encerramento das cotações
opening_quotes = np.array([quote.Open for quote in quotes]).astype(np.float)
closing_quotes = np.array([quote.Close for quote in quotes]).astype(np.float)

# calcular as diferenças entre abertura e encerramento
quotes_diff = closing_quotes - opening_quotes

# Normalizar os dados
X = quotes_diff.copy().T
X /= X.std(axis=0)

# Criar o modelo
edge_model = covariance.GraphicalLassoCV()

# Treinar o modelo
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Construir o modelo de clusterização baseado em propagação por afinidade
_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=None)
num_labels = labels.max()

# Imprimir os resultados
print('\nClusterização das ações com base na diferença entre as cotações na abertura e no encerramento:\n')
for i in range(num_labels + 1):
    print("Cluster", i+1, "==>", ', '.join(names[labels == i]))
