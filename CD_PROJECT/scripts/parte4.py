from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregar a base de dados Parkinsons
parkinsons = pd.read_csv('data/parkinsons.data')

# Separar X (features) e y (target)
X = parkinsons.drop(columns=['name', 'status'])  # 'status' é o target
y = parkinsons['status']

# Normalizar os dados (valores entre 0 e 1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Dividir em treinamento e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Divisão do dataset: {80}% treinamento e {20}% teste")

# Configurar o modelo MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000, random_state=42)

# Treinar o modelo
mlp.fit(X_train, y_train)

# Fazer predições
y_pred = mlp.predict(X_test)

# Avaliação do modelo
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Percentual de acerto da base de teste
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPercentual de Acerto da Base de Teste: {accuracy * 100:.2f}%")

# Gráficos para visualização dos resultados

# 1. Curva de Perda do Modelo
plt.figure(figsize=(8, 5))
plt.plot(mlp.loss_curve_)
plt.title('Curva de Perda do Modelo')
plt.xlabel('Iterações')
plt.ylabel('Erro')
plt.grid(True)
plt.show()

# 2. Confusion Matrix como gráfico
plt.figure(figsize=(8, 5))
conf_matrix = confusion_matrix(y_test, y_pred)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.colorbar()
tick_marks = np.arange(len(set(y)))
plt.xticks(tick_marks, set(y))
plt.yticks(tick_marks, set(y))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# 3. Gráfico da acurácia por execução
accuracies = []
for i in range(5):  # Rodar o modelo várias vezes para estabilidade
    mlp_temp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000, random_state=i)
    mlp_temp.fit(X_train, y_train)
    y_pred_temp = mlp_temp.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred_temp))

plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), accuracies, marker='o')
plt.title('Acurácia por Execução')
plt.xlabel('Execução')
plt.ylabel('Acurácia')
plt.grid(True)
plt.show()

# Apresentando a topologia da rede
print("\nTopologia da Rede Neural:")
print(f" - Camadas Ocultas: {mlp.hidden_layer_sizes}")
print(f" - Função de Ativação: {mlp.activation}")
print(f" - Número de Iterações Executadas: {mlp.n_iter_}")
