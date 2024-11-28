import pandas as pd
import numpy as np

# Criando os DataFrames de exemplo
usuarios = pd.DataFrame({
    'pk': [1, 2, 3, 4, 5, 6],
    'nome': ['Ana', 'Bob', 'Carlos', 'Diana', 'Elisa', 'Ana Lis'],
    'idade': [19, 40, 25, 47, 33, 18]
})

cidades = pd.DataFrame({
    'pk': [1, 2, 3, 4, 5, 6],
    'estado': ['PE', 'SP', 'RJ', 'MG', 'SP', 'PE'],
    'cidade': ['Recife', 'São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Campinas', 'Olinda']
})

# 1. SELECT nome, idade FROM Usuario
resultado_1 = usuarios[['nome', 'idade']]
print("SELECT nome, idade FROM Usuario:\n", resultado_1)

# 2. SELECT * FROM Usuario WHERE idade > 35
resultado_2 = usuarios[usuarios['idade'] > 35]
print("\nSELECT * FROM Usuario WHERE idade > 35:\n", resultado_2)

# 3. SELECT * FROM Usuário Us INNER JOIN Cidade Ci ON Us.pk = Ci.pk
resultado_3 = pd.merge(usuarios, cidades, on='pk', how="inner", validate="many_to_many")
print("\nSELECT * FROM Usuário Us INNER JOIN Cidade Ci ON Us.pk = Ci.pk:\n", resultado_3)

# 4. SELECT nome, estado, COUNT(*) FROM Cidade GROUP BY estado
resultado_4 = cidades.groupby('estado').size().reset_index(name='count')
print("\nSELECT estado, COUNT(*) FROM Cidade GROUP BY estado:\n", resultado_4)

# 5. SELECT estado, SUM(idade) FROM Usuário Us INNER JOIN Cidade Ci ON Us.pk = Ci.pk GROUP BY estado
resultado_5 = resultado_3.groupby('estado')['idade'].sum().reset_index(name='soma_idade')
print("\nSELECT estado, SUM(idade) FROM Usuário INNER JOIN Cidade GROUP BY estado:\n", resultado_5)

# 6. SELECT * FROM Usuario ORDER BY idade
resultado_6 = usuarios.sort_values(by='idade')
print("\nSELECT * FROM Usuario ORDER BY idade:\n", resultado_6)

# 7. SELECT * FROM Usuario LIMIT 3
resultado_7 = usuarios.head(3)
print("\nSELECT * FROM Usuario LIMIT 3:\n", resultado_7)

# 8. SELECT * FROM Usuario ORDER BY idade DESC LIMIT 6
resultado_8 = usuarios.sort_values(by='idade', ascending=False).head(6)
print("\nSELECT * FROM Usuario ORDER BY idade DESC LIMIT 6:\n", resultado_8)

# Quantas linhas tem a tabela Usuário e a tabela Cidade? (COUNT)
count_usuarios = len(usuarios)
count_cidades = len(cidades)
print(f"\nQuantas linhas tem a tabela Usuario? {count_usuarios}")
print(f"Quantas linhas tem a tabela Cidade? {count_cidades}")

# Qual a média das idades dos Usuários? (AVG)
media_idades = usuarios['idade'].mean()
print(f"\nQual a média das idades dos Usuários? {media_idades:.2f}")

# Qual a soma das idades dos Usuários? (SUM)
soma_idades = usuarios['idade'].sum()
print(f"\nQual a soma das idades dos Usuários? {soma_idades}")

# SELECT * FROM Usuario WHERE idade IN (19,47)
resultado_in = usuarios[usuarios['idade'].isin([19, 47])]
print("\nSELECT * FROM Usuario WHERE idade IN (19,47):\n", resultado_in)

# SELECT * FROM Usuario WHERE idade NOT IN (19,47)
resultado_not_in = usuarios[~usuarios['idade'].isin([19, 47])]
print("\nSELECT * FROM Usuario WHERE idade NOT IN (19,47):\n", resultado_not_in)

# SELECT * FROM Usuario WHERE idade BETWEEN 23 AND 50
resultado_between = usuarios[(usuarios['idade'] >= 23) & (usuarios['idade'] <= 50)]
print("\nSELECT * FROM Usuario WHERE idade BETWEEN 23 AND 50:\n", resultado_between)

# SELECT DISTINCT estado FROM Cidade
distinct_estado = cidades['estado'].drop_duplicates().reset_index(drop=True)
print("\nSELECT DISTINCT estado FROM Cidade:\n", distinct_estado)

# SELECT DISTINCT nome, estado FROM Cidade
distinct_nome_estado = cidades[['cidade', 'estado']].drop_duplicates().reset_index(drop=True)
print("\nSELECT DISTINCT nome, estado FROM Cidade:\n", distinct_nome_estado)

# SELECT id, nome, idade, CASE WHEN idade > 18 THEN 'ADULTO' ELSE 'ADOLESCENTE' END AS categoria FROM Usuario
usuarios['categoria'] = usuarios['idade'].apply(lambda x: 'ADULTO' if x > 18 else 'ADOLESCENTE')
print("\nSELECT id, nome, idade, CASE WHEN idade > 18 THEN 'ADULTO' ELSE 'ADOLESCENTE' END AS categoria FROM Usuario:\n", usuarios)

# SELECT estado, COUNT(*) FROM Cidade GROUP BY estado HAVING COUNT(*) > 1
having_count = cidades.groupby('estado').filter(lambda x: len(x) > 1).groupby('estado').size().reset_index(name='count')
print("\nSELECT estado, COUNT(*) FROM Cidade GROUP BY estado HAVING COUNT(*) > 1:\n", having_count)

# CREATE INDEX idx_nome ON Usuario (nome)
usuarios.set_index('nome', inplace=True)
print("\nCREATE INDEX idx_nome ON Usuario (nome):\n", usuarios.reset_index())

# SELECT id, nome FROM Usuario UNION SELECT id, nome, estado FROM Cidade
union_distinct = pd.concat([
    usuarios.reset_index()[['pk', 'nome']],
    cidades[['pk', 'cidade']].rename(columns={'cidade': 'nome'})
]).drop_duplicates()
print("\nSELECT id, nome FROM Usuario UNION SELECT id, nome FROM Cidade:\n", union_distinct)
