# pip install nltk gensim stanza spacy regex matplotlib

import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
import stanza
import spacy
import matplotlib.pyplot as plt

# Baixar recursos necessários para as bibliotecas
nltk.download('stopwords')

# Inicializar Stanza sem o NER
stanza.download('pt')
nlp_stanza = stanza.Pipeline('pt', processors='tokenize,pos,lemma,depparse')
nlp_spacy = spacy.load("pt_core_news_sm")

# Corpus
corpus = """A Ana Lis mora em uma cidade linda, chamada João Pessoa. Humanos compram
livros e Ana Lis comprou um livro sobre biologia marinha. O João leu o jornal na
biblioteca durante a manhã. O poeta escreveu uma carta para o professor da faculdade
e para seus discípulos. O cachorro da Ana <Lis> perseguiu um gato no quintal,
lembrando que gatos são felinos e cachorros são caninos*. A professora explicou ou
ensinou o conteúdo da aula para os alunos. O menino comeu uma maçã depois do
almoço. O carteiro entregou uma encomenda para a vizinha. A vaca comeu o capim do
pasto e por isso é herbívora e não é carnívora. O médico examinou o paciente com
cuidado no consultório. [O artista pintou um quadro no ateliê]."""

# 1. Limpeza do Corpus
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[áàâãä]', 'a', text)
    text = re.sub(r'[éèêë]', 'e', text)
    text = re.sub(r'[íìîï]', 'i', text)
    text = re.sub(r'[óòôõö]', 'o', text)
    text = re.sub(r'[úùûü]', 'u', text)
    text = re.sub(r'[ç]', 'c', text)
    text = re.sub(r'[^a-z\s]', '', text)  # Remove pontuação e números
    text = re.sub(r'\s+', ' ', text).strip()
    return text

cleaned_corpus = clean_text(corpus)
print("Texto Limpo:\n", cleaned_corpus)

# 2. Tokenização com Stanza
doc_stanza = nlp_stanza(cleaned_corpus)
tokens = [word.text for sentence in doc_stanza.sentences for word in sentence.words]
print("\nTokens (Stanza):\n", tokens)

# Remoção de Stopwords
stop_words = set(stopwords.words('portuguese'))
tokens_no_stopwords = [word for word in tokens if word not in stop_words]
print("\nTokens sem Stopwords:\n", tokens_no_stopwords)

# 3. POS Tagging com Stanza
pos_tags_stanza = [(word.text, word.upos) for sentence in doc_stanza.sentences for word in sentence.words]
df_pos_tags = pd.DataFrame(pos_tags_stanza, columns=["Token", "Classe Gramatical"])
print("\nPOS Tagging (Stanza):\n", df_pos_tags)

# 4. Lematização com Stanza
lemmas = [(word.text, word.lemma) for sentence in doc_stanza.sentences for word in sentence.words]
df_lemmas = pd.DataFrame(lemmas, columns=["Token", "Lema"])
print("\nLematização (Stanza):\n", df_lemmas)

# 5. Dependências Sintáticas e Entidades Nomeadas com spaCy
doc_spacy = nlp_spacy(cleaned_corpus)

# Dependências Sintáticas
dependencies = [(token.text, token.dep_, token.head.text) for token in doc_spacy]
df_dependencies = pd.DataFrame(dependencies, columns=["Token", "Relação", "Cabeça"])
print("\nDependências Sintáticas:\n", df_dependencies)

# Entidades Nomeadas
entities = [(ent.text, ent.label_) for ent in doc_spacy.ents]
df_entities = pd.DataFrame(entities, columns=["Entidade", "Tipo"])
print("\nEntidades Nomeadas (spaCy):\n", df_entities)

# 6. Relações Sujeito-Predicado-Objeto
relations = [(token.head.text, token.text, token.dep_) for token in doc_spacy if token.dep_ in ["nsubj", "obj"]]
df_relations = pd.DataFrame(relations, columns=["Predicado", "Token", "Relação"])
print("\nRelações (Sujeito-Predicado-Objeto):\n", df_relations)

# 7. Visualizações

# Gráfico 1: Frequência de Classes Gramaticais
freq_pos = df_pos_tags["Classe Gramatical"].value_counts()
freq_pos.plot(kind='bar', title='Frequência de Classes Gramaticais', xlabel='Classe Gramatical', ylabel='Frequência')
plt.savefig("frequencia_classes_gramaticais.png")
plt.close()

# Gráfico 2: Frequência de Tokens
freq_tokens = pd.Series(tokens_no_stopwords).value_counts()
freq_tokens.head(10).plot(kind='bar', title='Frequência de Tokens', xlabel='Tokens', ylabel='Frequência')
plt.savefig("frequencia_tokens.png")
plt.close()

# Gráfico 3: Relações Sujeito-Predicado-Objeto
relation_freq = df_relations["Relação"].value_counts()
relation_freq.plot(kind='bar', title='Frequência de Relações', xlabel='Relações', ylabel='Frequência')
plt.savefig("frequencia_relacoes.png")
plt.close()
