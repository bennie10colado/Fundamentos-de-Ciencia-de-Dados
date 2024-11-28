
# CD_Project - Guia de Execução

## 1. Pré-requisitos
Certifique-se de que o **Python 3.x** está instalado no ambiente. Em seguida, instale as bibliotecas necessárias executando o seguinte comando:

```bash
pip install -r requirements.txt
```

---

## 2. Como Rodar os Scripts

### Lista 1

#### Parte 1 (NumPy)
Executa os exercícios práticos de NumPy.
```bash
python scripts/parte1.py
```

#### Parte 2 (SciPy)
Executa os exercícios práticos de SciPy.
```bash
python scripts/parte2.py
```

#### Parte 3 (Pandas)
Executa os exercícios práticos de manipulação de dados tabulares com Pandas.
```bash
python scripts/parte3.py
```
Arquivos gerados:
- `outputs/Cid_01.csv`
- `outputs/Cid_01.html`
- `outputs/Cid_01.json`

#### Parte 4 (Parkinsons - MLP)
Executa o modelo de rede neural MLP para a base Parkinsons.
```bash
python scripts/parte4.py
```

#### Parte 4-2 (Penguins - MLP)
Executa o modelo de rede neural MLP para a base Penguins.
```bash
python scripts/parte4-2.py
```

---

### Lista 2

#### Parte 3 - Processamento de Linguagem Natural (Stanza/Spacy)
Este script realiza análise textual utilizando bibliotecas como Stanza e spaCy:
- Tokenização.
- Lematização.
- POS Tagging.
- Dependências sintáticas e entidades nomeadas.
- Visualizações gráficas de relações.

Para executar:
```bash
python parte3.py
```
Gráficos gerados:
- Frequência de Tokens.
- Classes Gramaticais.
- Relações Sujeito-Predicado-Objeto.

#### Parte 4 - Análise com BERT
Este script realiza a análise textual com o modelo BERT:
- Tokenização e embeddings.
- Visualização de similaridades e distribuição de embeddings.
- Comparação entre métodos tradicionais e BERT.

Para executar:
```bash
python parte4.py
```
Gráficos gerados:
- Frequência de Tokens (BERT).
- Similaridade Coseno entre Tokens.
- Distribuição de Embeddings.

#### Observação
Certifique-se de que o **Graphviz** está instalado no sistema para gerar a topologia da rede BERT.

---

## 3. Visualização dos Resultados

- **Lista 1**:
  - **Partes 1 e 2:** Saídas impressas no terminal.
  - **Parte 3:** Arquivos gerados na pasta `outputs/`.
  - **Partes 4 e 4-2:** Gráficos exibidos automaticamente.

- **Lista 2**:
  - **Parte 3:** Gráficos salvos no diretório de execução (`frequencia_classes.png`, etc.).
  - **Parte 4:** Gráficos e a topologia da rede (`bert_topologia.png`).

---

## 4. Contato
Caso haja dúvidas ou problemas durante a execução, pode entrar em contato comigo!

**Email**: benj7100@gmail.com
