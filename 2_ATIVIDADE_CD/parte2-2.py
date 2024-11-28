import pandas as pd

# Carregando os datasets
usuarios = pd.read_csv("u.user", sep="|", header=None, names=["user_id", "age", "gender", "occupation", "zip_code"])
filmes = pd.read_csv("u.item", sep="|", header=None, encoding="latin-1",
                     names=["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL", "unknown",
                            "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
                            "Fantasy", "FilmNoir", "Horror", "Musical", "Mystery", "Romance", "SciFi", "Thriller",
                            "War", "Western"])
avaliacoes = pd.read_csv("u.data", sep="\t", header=None, names=["user_id", "movie_id", "rating", "timestamp"])

# 1. Transformar colunas categóricas
usuarios['gender'] = usuarios['gender'].replace({'M': 'Masculino', 'F': 'Feminino'}).astype('category')
usuarios['occupation'] = usuarios['occupation'].astype('category')
usuarios['zip_code'] = usuarios['zip_code'].astype('category')

# 2. Converter release_date e video_release_date para datetime
filmes['release_date'] = pd.to_datetime(filmes['release_date'], format='%d-%b-%Y', errors='coerce')
filmes['video_release_date'] = pd.to_datetime(filmes['video_release_date'], format='%d-%b-%Y', errors='coerce')

# 3. Ordenar valores únicos da coluna rating na tabela avaliacoes
avaliacoes['rating'] = pd.Categorical(avaliacoes['rating'], categories=sorted(avaliacoes['rating'].unique()), ordered=True)
print("Valores únicos de rating ordenados:", avaliacoes['rating'].unique())

# 4. Estatísticas de rating
print("Descrição estatística de rating:\n", avaliacoes['rating'].describe())

# 5. Converter timestamp para datetime
avaliacoes['timestamp'] = pd.to_datetime(avaliacoes['timestamp'], unit='s')
print("Descrição estatística de timestamp:\n", avaliacoes['timestamp'].describe())

# 6. Verificar dados ausentes na tabela filmes
missing_filmes = filmes.isnull().sum()
print("Dados ausentes em filmes:\n", missing_filmes)

# 7. Remover linhas com valores ausentes em release_date e IMDb_URL
filmes_cleaned = filmes.dropna(subset=['release_date', 'IMDb_URL'])

# 8. Remover colunas completamente ausentes
filmes_cleaned = filmes_cleaned.dropna(axis=1, how='all')

# 9. Remover filme inconsistente (sem release_date)
filmes_cleaned = filmes_cleaned[filmes_cleaned['release_date'].notna()]

# 10. Preencher valores ausentes com a moda
# Preencher valores ausentes com a moda (corrigido)
for column in filmes.columns:
    if filmes[column].isnull().sum() > 0:
        # Verificar se existe moda para a coluna
        moda = filmes[column].mode()
        if not moda.empty:
            filmes[column].fillna(moda[0], inplace=True)

# Mostrar o DataFrame de filmes após o preenchimento
print("Tabela de filmes após preenchimento com moda:\n", filmes.head())


# 11. Criar um DataFrame e preencher com estratégias diferentes
df_missing = pd.DataFrame({
    'col1': [1, None, 3, None, 5],
    'col2': [2, 2, None, 4, 4]
})

print("\nPreenchendo dados ausentes:")
# Preencher com moda
df_mode = df_missing.fillna(df_missing.mode().iloc[0])
print("Com moda:\n", df_mode)

# Preencher com média
df_mean = df_missing.fillna(df_missing.mean())
print("Com média:\n", df_mean)

# Preencher com mediana
df_median = df_missing.fillna(df_missing.median())
print("Com mediana:\n", df_median)

# Preencher com valor mais próximo
df_nearest = df_missing.interpolate()
print("Com valor mais próximo:\n", df_nearest)

# 12. Verificar duplicados na tabela avaliacoes
duplicados_avaliacoes = avaliacoes.duplicated().sum()
print(f"\nDados duplicados em avaliacoes: {duplicados_avaliacoes}")

# 13. Verificar duplicados na coluna movie_id
duplicados_movie_id = avaliacoes.duplicated(subset=['movie_id']).sum()
print(f"\nDados duplicados em movie_id: {duplicados_movie_id}")

# 14. Remover duplicados da coluna movie_id
avaliacoes_unique = avaliacoes.drop_duplicates(subset=['movie_id'])
print("Tabela avaliacoes limpa de duplicados:\n", avaliacoes_unique.head())
