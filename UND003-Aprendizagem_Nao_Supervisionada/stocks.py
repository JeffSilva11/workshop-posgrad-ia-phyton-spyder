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

# Carregar os dados históricos das ações com tratamento de erros
start_date = datetime.datetime(2019, 1, 1)
end_date = datetime.datetime(2019, 1, 31)

valid_quotes = []
valid_indices = []

for i, symbol in enumerate(symbols):
    try:
        data = yf.Ticker(symbol).history(start=start_date, end=end_date)
        if not data.empty and 'Open' in data.columns and 'Close' in data.columns:
            valid_quotes.append(data)
            valid_indices.append(i)
    except:
        continue

# Filtrar símbolos e nomes válidos
valid_symbols = symbols[valid_indices]
valid_names = names[valid_indices]

# Extrair dados da abertura e encerramento das cotações
opening_quotes = np.array([quote['Open'].values for quote in valid_quotes]).astype(float)
closing_quotes = np.array([quote['Close'].values for quote in valid_quotes]).astype(float)

# Verificar se temos dados suficientes
if len(valid_quotes) == 0:
    raise ValueError("Nenhum dado válido foi obtido para análise")

# Calcular as diferenças entre abertura e encerramento
quotes_diff = closing_quotes - opening_quotes

# Normalizar os dados
X = quotes_diff.copy().T
X /= X.std(axis=0)

# Criar o modelo
edge_model = covariance.GraphicalLassoCV()

# Treinar o modelo
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Construir o modelo de clusterização
_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=None)
num_labels = labels.max()

# Imprimir os resultados
print('\nClusterização das ações com base na diferença entre as cotações:\n')
for i in range(num_labels + 1):
    print(f"Cluster {i+1} ==> {', '.join(valid_names[labels == i])}")