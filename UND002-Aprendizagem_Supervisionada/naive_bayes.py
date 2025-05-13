import numpy as np
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from utilities import visualize_classifier

# Arquivo contendo os dados de entrada
input_file = 'data_multivar_nb.txt'

# Carregar os dados do arquivo
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1] 

# Criar o classificador Naive Bayes
classifier = GaussianNB()

# Treinar o classificador
classifier.fit(X, y)

# Prever os valores para os dados de treinamento
y_pred = classifier.predict(X)

# Calcular a acurácia
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Acurácia do classificador Naive Bayes =", round(accuracy, 2), "%")

# Visualizar o desempenho do classificador
visualize_classifier(classifier, X, y)

##############################
# Validação cruzada (Hold-out)

# Dividir os dados em conjuntos de teste e treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)

# Calcular a acurácia do classificador
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Acurácia do NOVO classificador Naive Bayes =", round(accuracy, 2), "%")

# Visualizar o desempenho do classificador
visualize_classifier(classifier_new, X_test, y_test)

############################
# Validação cruzada (k_Fold)
num_folds = 3
accuracy_values = cross_val_score(classifier, 
        X, y, scoring='accuracy', cv=num_folds)
print("Acurácia: " + str(round(100*accuracy_values.mean(), 2)) + "%")

precision_values = cross_val_score(classifier, 
        X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100*precision_values.mean(), 2)) + "%")

recall_values = cross_val_score(classifier, 
        X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100*recall_values.mean(), 2)) + "%")

f1_values = cross_val_score(classifier, 
        X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100*f1_values.mean(), 2)) + "%")

