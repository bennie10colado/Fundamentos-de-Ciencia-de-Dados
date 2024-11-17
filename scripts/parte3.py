import pandas as pd
import numpy as np

# Parte 3: Manipulação de Dados Tabulares com Pandas

# Questão 3.1 - Criar o DataFrame 'Cidades' conforme especificado
cidades = pd.DataFrame({
    'ESTADO': ['Paraíba', 'Rio Grande do Norte', 'Pernambuco', 'Ceará', 'Bahia'],
    'CIDADE': ['João Pessoa', 'Natal', 'Recife', 'Fortaleza', 'Salvador'],
    'HABITANTES': [604564, 765298, 890765, 149087, 1920613],
    'MÉDIA SALARIAL': [1500, 2000, 3023, 3200, 2199]
})
print("Questão 3.1 - DataFrame Cidades:")
print(cidades)

# Questão 3.2 - Operações diversas com o DataFrame
print("\nApenas as 2 primeiras linhas:\n", cidades.head(2))
print("\nApenas as 2 últimas linhas:\n", cidades.tail(2))
print("\nNúmero de linhas e colunas:", cidades.shape)
print("\nInformações Gerais:")
print(cidades.info())
print("\nNome de todas as colunas:", cidades.columns.tolist())
print("\nMedidas estatísticas:\n", cidades[['MÉDIA SALARIAL', 'HABITANTES']].describe())

# Alterando o tipo da coluna 'HABITANTES' para float
cidades['HABITANTES'] = cidades['HABITANTES'].astype(float)

# Inserindo uma nova coluna com datas aleatórias
cidades['Data'] = pd.date_range('2024-01-01', periods=5, freq='M')

# Renomeando colunas
cidades.rename(columns={'HABITANTES': 'Nº Habitantes', 'Data': 'Data – Censo'}, inplace=True)

# Localizações específicas
print("\nInformações da cidade de Natal:\n", cidades[cidades['CIDADE'] == 'Natal'])
print("\nInformações das cidades João Pessoa e Fortaleza:\n", cidades[cidades['CIDADE'].isin(['João Pessoa', 'Fortaleza'])])
print("\nInformações entre Recife e Salvador:\n", cidades[cidades['CIDADE'].between('Recife', 'Salvador')])

# Reordenando e selecionando colunas
print("\nColunas na ordem: Cidade, Média Salarial, Estado e Nº Habitantes:")
print(cidades[['CIDADE', 'MÉDIA SALARIAL', 'ESTADO', 'Nº Habitantes']])

# Inserindo nova coluna de siglas
cidades.insert(2, 'Sigla', ['PB', 'RN', 'PE', 'CE', 'BA'])
cidades['Sigla-Repetida'] = cidades['Sigla']

# Removendo colunas usando pop e drop
cidades.pop('Sigla-Repetida')
cidades['Sigla-Repetida'] = cidades['Sigla']
cidades.drop(columns=['Sigla-Repetida'], inplace=True)

# Filtros diversos
print("\nCidades com Média Salarial < 2100:\n", cidades[cidades['MÉDIA SALARIAL'] < 2100])
print("\nApenas cidades do estado de PE:\n", cidades[cidades['ESTADO'] == 'Pernambuco'])
print("\nCidades com Nº Habitantes entre 500000 e 1500000:\n", cidades[(cidades['Nº Habitantes'] > 500000) & (cidades['Nº Habitantes'] < 1500000)])

# Ordenações
print("\nOrdenar cidades por nome:\n", cidades.sort_values(by='CIDADE'))
print("\nOrdenar cidades por Média Salarial:\n", cidades.sort_values(by='MÉDIA SALARIAL'))

# Questão 3.3 - Criar o DataFrame cidades_salarios
cidades_salarios = pd.DataFrame({
    'ESTADO': ['Paraíba', 'Rio Grande do Norte', 'Pernambuco', 'Ceará', 'Bahia'],
    'CIDADE': ['João Pessoa', 'Natal', 'Recife', 'Fortaleza', 'Salvador'],
    'MÉDIA SALARIAL': [1500, 2000, 3023, 3200, 2199]
})
print("\nDataFrame cidades_salarios:\n", cidades_salarios)

# Questão 3.4 - Criar o DataFrame cidades_pop
cidades_pop = pd.DataFrame({
    'CIDADE': ['João Pessoa', 'Natal', 'Recife', 'Fortaleza', 'Salvador', 'Teresina'],
    'HABITANTES': [604564, 765298, 890765, 149087, 1920613, 1678943]
})
print("\nDataFrame cidades_pop:\n", cidades_pop)

# Questão 3.5 - Realizar joins entre os DataFrames
inner_join = pd.merge(cidades_salarios, cidades_pop, on='CIDADE', how='inner')
print("\nInner Join entre cidades_salarios e cidades_pop:\n", inner_join)

outer_join = pd.merge(cidades_salarios, cidades_pop, on='CIDADE', how='outer')
print("\nOuter Join entre cidades_salarios e cidades_pop:\n", outer_join)

# Questão 3.6 - Concatenação de DataFrames
df1 = cidades_pop.iloc[:3]
df2 = cidades_pop.iloc[3:]

concat_nao_ignora = pd.concat([df1, df2], ignore_index=False)
print("\nConcatenação sem ignorar índices:\n", concat_nao_ignora)

concat_ignora = pd.concat([df1, df2], ignore_index=True)
print("\nConcatenação ignorando índices:\n", concat_ignora)

# Questão 3.7 - Salvando os DataFrames em diferentes formatos
cidades.to_csv('Cid_01.csv', index=False)
cidades.to_html('Cid_01.html', index=False)
cidades.to_json('Cid_01.json', orient='records')
print("\nDataFrames salvos nos formatos CSV, HTML e JSON.")

# Questões 3.8 e 3.9 - Leitura de arquivos CSV locais e remotos
url = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.user'
dados_remotos = pd.read_csv(url, sep='|', header=None)
print("\nQuestão 3.8 - Dados remotos:\n", dados_remotos.head())

dados_locais = pd.read_csv('Cid_01.csv')
print("\nQuestão 3.9 - Dados locais:\n", dados_locais.head())
