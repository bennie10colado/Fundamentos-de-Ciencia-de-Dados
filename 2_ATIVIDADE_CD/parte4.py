# pip install transformers torch matplotlib sklearn torchviz

import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from torchviz import make_dot
from nltk.corpus import stopwords

# Certifique-se de baixar os stopwords
import nltk
nltk.download("stopwords")

# Corpus da questão anterior
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

# 2. Carregar Modelo e Tokenizador BERT
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 3. Tokenização com BERT
inputs = tokenizer(cleaned_corpus, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)

# 4. Extração de Embeddings
embeddings = outputs.last_hidden_state
print("\nTamanho dos Embeddings:", embeddings.size())

# 5. Ajustar Tokens com Embeddings
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  # Tokens alinhados com os embeddings
token_embeddings = embeddings[0].detach().numpy()

# Criar Dataset com Embeddings
df_embeddings = pd.DataFrame(token_embeddings, index=tokens)
print("\nDataset de Tokens e Embeddings:\n", df_embeddings.head())

# Filtrar tokens removendo stopwords
stop_words = set(stopwords.words('portuguese'))
tokens_no_stopwords = [token for token in tokens if token.lower() not in stop_words and token.isalpha()]

# 6. Visualizações

# Gráfico 1: Frequência de Tokens
freq_tokens = pd.Series(tokens_no_stopwords).value_counts()
freq_tokens.head(10).plot(kind='bar', title='Frequência de Tokens (BERT)', xlabel='Tokens', ylabel='Frequência')
plt.savefig("bert_freq_tokens.png")
plt.close()

# Gráfico 2: Similaridade entre Tokens
similarities = cosine_similarity(token_embeddings)
plt.imshow(similarities, cmap='viridis')
plt.title("Similaridade Coseno entre Tokens (BERT)")
plt.colorbar()
plt.savefig("bert_similaridade_tokens.png")
plt.close()

# Gráfico 3: Distribuição de Embeddings
plt.figure(figsize=(10, 6))
plt.hist(token_embeddings.flatten(), bins=50, color="blue", alpha=0.7)
plt.title("Distribuição de Embeddings (BERT)")
plt.xlabel("Valor do Embedding")
plt.ylabel("Frequência")
plt.savefig("bert_distribuicao_embeddings.png")
plt.close()

# 7. Comparação com Parte 3 (valores fictícios de Parte 3)
df_pos_tags = pd.DataFrame({"Token": tokens_no_stopwords, "Classe Gramatical": ["-"] * len(tokens_no_stopwords)})
df_lemmas = pd.DataFrame({"Token": tokens_no_stopwords, "Lema": tokens_no_stopwords})
df_entities = pd.DataFrame({"Entidade": ["-"] * len(tokens_no_stopwords), "Tipo": ["-"] * len(tokens_no_stopwords)})

comparison = pd.DataFrame({
    "Método": ["Parte 3 (Stanza/Spacy)", "Parte 4 (BERT)"],
    "Tokens Identificados": [len(tokens_no_stopwords), len(tokens)],
    "Classes Gramaticais": [len(df_pos_tags), "Automático via Embeddings"],
    "Lematização": [len(df_lemmas), "Via Modelo Embeddings"],
    "Entidades Nomeadas": [len(df_entities), "Via Tokenização BERT"]
})
print("\nComparação entre Parte 3 e Parte 4:\n", comparison)

# 8. Topologia da Rede (Visualização)
dot = make_dot(outputs.last_hidden_state, params=dict(model.named_parameters()))
dot.render("bert_topologia", format="png", cleanup=True)
