# !pip install nltk spacy stanza regex

import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import pandas as pd
from spacy import displacy
import stanza

# Configurações iniciais
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Texto do corpus
corpus = """A Ana Lis mora em uma cidade linda, chamada João Pessoa. Humanos compram
livros e Ana Lis comprou um livro sobre biologia marinha. O João leu o jornal na
biblioteca durante a manhã. O poeta escreveu uma carta para o professor da faculdade
e para seus discípulos. O cachorro da Ana <Lis> perseguiu um gato no quintal,
lembrando que gatos são felinos e cachorros são caninos*. A professora explicou ou
ensinou o conteúdo da aula para os alunos. O menino comeu uma maçã depois do
almoço. O carteiro entregou uma encomenda para a vizinha. A vaca comeu o capim do
pasto e por isso é herbívora e não é carnívora. O médico examinou o paciente com
cuidado no consultório. [O artista pintou um quadro no ateliê]."""

# 1. Limpeza do texto
def clean_text(text):
    text = text.lower()  # Converte para minúsculas
    text = re.sub(r'[áàâãä]', 'a', text)
    text = re.sub(r'[éèêë]', 'e', text)
    text = re.sub(r'[íìîï]', 'i', text)
    text = re.sub(r'[óòôõö]', 'o', text)
    text = re.sub(r'[úùûü]', 'u', text)
    text = re.sub(r'[ç]', 'c', text)
    text = re.sub(r'[^a-z\s]', '', text)  # Remove pontuação e números
    text = re.sub(r'\s+', ' ', text).strip()  # Remove espaços extras
    return text

cleaned_corpus = clean_text(corpus)
print("Texto limpo:\n", cleaned_corpus)

# 2. Tokenização
tokens = word_tokenize(cleaned_corpus)
print("\nTokens:\n", tokens)

# 3. Stopwords
stop_words = set(stopwords.words('portuguese'))
tokens_no_stopwords = [word for word in tokens if word not in stop_words]
print("\nTokens sem stopwords:\n", tokens_no_stopwords)

# 4. POS Tagging com NLTK
pos_tags = pos_tag(tokens_no_stopwords, lang='por')
print("\nPOS Tagging:\n", pos_tags)

# 5. Lematização
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens_no_stopwords]
print("\nTokens Lematizados:\n", lemmatized_tokens)

# 6. Análise de Dependências e Entidades Nomeadas com spaCy
nlp_spacy = spacy.load('pt_core_news_sm')
doc = nlp_spacy(cleaned_corpus)

# Dependências sintáticas
print("\nDependências sintáticas:")
for token in doc:
    print(f"{token.text} -> {token.dep_} -> {token.head.text}")

# Entidades Nomeadas
print("\nEntidades Nomeadas:")
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")

# 7. Criar Dataset organizado por classe gramatical
dataset_pos = pd.DataFrame(pos_tags, columns=['Token', 'Classe Gramatical'])
print("\nDataset por Classe Gramatical:\n", dataset_pos.head())

# 8. Extração de Relações (Sujeito - Predicado - Objeto)
relations = []
for token in doc:
    if token.dep_ in ("nsubj", "obj"):
        relations.append((token.head.text, token.text, token.dep_))

relations_df = pd.DataFrame(relations, columns=["Predicado", "Token", "Relação"])
print("\nRelações extraídas:\n", relations_df)

# 9. Avaliação (percentual de acerto e erro)
# Aqui podemos comparar manualmente as relações extraídas e os tokens categorizados
# Gráficos
import matplotlib.pyplot as plt

# Frequência de classes gramaticais
freq_classes = dataset_pos['Classe Gramatical'].value_counts()
freq_classes.plot(kind='bar', title='Frequência de Classes Gramaticais', xlabel='Classe', ylabel='Frequência')
plt.show()

# Relações extraídas
relations_df['Relação'].value_counts().plot(kind='bar', title='Frequência de Relações', xlabel='Tipo de Relação', ylabel='Frequência')
plt.show()
