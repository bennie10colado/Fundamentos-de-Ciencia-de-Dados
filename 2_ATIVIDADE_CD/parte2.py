import pandas as pd
import numpy as np

# Criando o DataFrame para exemplo
np.random.seed(42)
df = pd.DataFrame({
    'A': [1, np.nan, 3, np.nan, ' '],
    'B': [4, np.nan, np.nan, np.nan, 3],
    'C': [7, np.nan, 9, np.nan, 11],
    'D': [5, 4, 9, np.nan, ' ']
})

# 1. Identificar a quantidade de dados ausentes
print("Dados originais:\n", df)

# Substituindo " " por NaN
df.replace(" ", np.nan, inplace=True)
print("\nDados após substituir ' ' por NaN:\n", df)

# Quantidade de dados ausentes por coluna
missing_data = df.isnull().sum()
print("\nQuantidade de dados ausentes por coluna:\n", missing_data)

# 2. Remover linhas com valor ausente em qualquer atributo
removed_any = df.dropna(how='any')
print("\nLinhas com pelo menos um valor ausente removidas:\n", removed_any)

# 3. Remover linhas com valor ausente em todos os atributos
removed_all = df.dropna(how='all')
print("\nLinhas com todos os valores ausentes removidas:\n", removed_all)

# 4. Reaplicar as remoções após substituir " " por NaN
# Remoção com 'any'
re_removed_any = df.dropna(how='any')
print("\nApós substituição, linhas com pelo menos um valor ausente removidas:\n", re_removed_any)

# Remoção com 'all'
re_removed_all = df.dropna(how='all')
print("\nApós substituição, linhas com todos os valores ausentes removidas:\n", re_removed_all)

# 5. Substituindo valores ausentes com diferentes estratégias
# Criando um DataFrame de exemplo
fill_example = pd.DataFrame({
    'A': [np.nan, 2, np.nan, 4, 5],
    'B': [1, np.nan, 3, 4, np.nan]
})

# Substituir por moda (valor mais frequente)
fill_mode = fill_example.fillna(fill_example.mode().iloc[0])
print("\nPreenchendo valores ausentes com a moda:\n", fill_mode)

# Substituir por média
fill_mean = fill_example.fillna(fill_example.mean())
print("\nPreenchendo valores ausentes com a média:\n", fill_mean)

# Substituir por mediana
fill_median = fill_example.fillna(fill_example.median())
print("\nPreenchendo valores ausentes com a mediana:\n", fill_median)

# Substituir por valor mais próximo (interpolação)
fill_nearest = fill_example.interpolate()
print("\nPreenchendo valores ausentes com o valor mais próximo (interpolação):\n", fill_nearest)
