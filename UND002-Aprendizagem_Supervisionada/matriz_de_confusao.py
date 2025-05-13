#importar os pacotes necessários
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Definir os rótulos verdadeiros e falses
true_labels = [2, 0, 0, 2, 4, 4, 1, 0, 3, 3, 3]
pred_labels = [2, 1, 0, 2, 4, 3, 1, 0, 1, 3, 3]

# Criar a matriz de confusão
confusion_mat = confusion_matrix(true_labels, pred_labels)

# Visualizar a matriz de confusão
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Matriz de Confusão')
plt.colorbar()
ticks = np.arange(5) #refere-se ao número de classes distintas
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('Classes verdadeiras')
plt.xlabel('Classes previstas')
plt.show()

# Relatório de classificação
targets = ['Class-0', 'Class-1', 'Class-2', 'Class-3', 'Class-4']
print('\n', classification_report(true_labels, pred_labels, target_names=targets))
