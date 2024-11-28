# Importação das bibliotecas necessárias
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar a base de dados Penguins
penguins = pd.read_csv('data/penguins.csv')

# Tratar valores nulos
penguins.dropna(inplace=True)

# Separar X (features) e y (target)
X = penguins.drop(columns=['species'])  # 'species' é o target
y = penguins['species']

# Identificar colunas categóricas e numéricas
categorical_columns = ['island', 'sex']
numerical_columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

# Pré-processamento: OneHotEncoding para categóricas e normalização para numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ])

# Criar o pipeline com o MLPClassifier
mlp_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(hidden_layer_sizes=(50, 30), activation='relu', max_iter=1000, random_state=42))
])

# Dividir em treinamento e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Divisão do dataset: {80}% treinamento e {20}% teste")

# Treinar o modelo
mlp_pipeline.fit(X_train, y_train)

# Fazer predições
y_pred = mlp_pipeline.predict(X_test)

# Avaliação do modelo
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Percentual de acerto da base de teste
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPercentual de Acerto da Base de Teste: {accuracy * 100:.2f}%")

# Gráficos para visualização dos resultados

# 1. Curva de Perda do Modelo
plt.figure(figsize=(8, 5))
plt.plot(mlp_pipeline.named_steps['classifier'].loss_curve_)
plt.title('Curva de Perda do Modelo')
plt.xlabel('Iterações')
plt.ylabel('Erro')
plt.grid(True)
plt.show()

# 2. Confusion Matrix como gráfico
plt.figure(figsize=(8, 5))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=mlp_pipeline.classes_, yticklabels=mlp_pipeline.classes_)
plt.title('Matriz de Confusão')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# 3. Gráfico da acurácia por execução
accuracies = []
for i in range(5):  # Rodar o modelo várias vezes para estabilidade
    mlp_temp = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(hidden_layer_sizes=(50, 30), activation='relu', max_iter=1000, random_state=i))
    ])
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
print(f" - Camadas Ocultas: {mlp_pipeline.named_steps['classifier'].hidden_layer_sizes}")
print(f" - Função de Ativação: {mlp_pipeline.named_steps['classifier'].activation}")
print(f" - Número de Iterações Executadas: {mlp_pipeline.named_steps['classifier'].n_iter_}")
