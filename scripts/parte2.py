from scipy import special, stats
from scipy.datasets import face
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform

# Parte 2: SciPy

# Questão 2.1 - Carregar e exibir a imagem "face" no formato original
face_image = face()
plt.imshow(face_image)
plt.title('Questão 2.1 - Imagem Original')
plt.axis('off')
plt.show()

# Questão 2.2 - Exibir a imagem anterior em escala de cinza
gray_face = face(gray=True)
plt.imshow(gray_face, cmap='gray')
plt.title('Questão 2.2 - Imagem em Escala de Cinza')
plt.axis('off')
plt.show()

# Questão 2.3 - Apresentar o array NumPy referente à imagem
print("Questão 2.3 - Array da Imagem em Escala de Cinza:\n", gray_face)

# Questão 2.5 - Redimensionar a imagem para 50% do tamanho original
resized_face = transform.resize(face_image, (face_image.shape[0] // 2, face_image.shape[1] // 2),
                                anti_aliasing=True)
plt.imshow(resized_face)
plt.title('Questão 2.5 - Imagem Redimensionada para 50%')
plt.axis('off')
plt.show()

# Questão 2.6 - Voltar a imagem para o tamanho original usando interpolação bilinear
original_size_face = transform.resize(resized_face, face_image.shape, anti_aliasing=True)
plt.imshow(original_size_face)
plt.title('Questão 2.6 - Imagem Redimensionada de Volta ao Tamanho Original')
plt.axis('off')
plt.show()

# Questão 2.7 - Calcular e apresentar o fatorial do número 4
fatorial_4 = special.factorial(4)
print("Questão 2.7 - Fatorial de 4:", fatorial_4)

# Questão 2.8 - Gráficos das funções de Bessel e de erro Gaussiana
x = np.linspace(0, 10, 100)
j1 = special.jv(1, x)  # Função de Bessel
error_func = special.erf(x)  # Função de erro Gaussiana

plt.plot(x, j1, label='Função de Bessel (J1)')
plt.plot(x, error_func, label='Função de Erro Gaussiana')
plt.legend()
plt.title('Questão 2.8 - Gráficos de Bessel e Erro Gaussiana')
plt.show()

# Questão 2.9 - Gráfico da função Gama
gamma_func = special.gamma(x)
plt.plot(x, gamma_func, label='Função Gama')
plt.legend()
plt.title('Questão 2.9 - Gráfico da Função Gama')
plt.show()

# Questão 2.10 - Gráficos dos polinômios de Legendre
legendre_func = special.legendre(3)
x_legendre = np.linspace(-1, 1, 100)
plt.plot(x_legendre, legendre_func(x_legendre), label='Polinômio de Legendre (3ª ordem)')
plt.legend()
plt.title('Questão 2.10 - Gráfico do Polinômio de Legendre')
plt.show()

# Questão 2.11 - Gráficos das funções PDF e CDF
from scipy.stats import norm

pdf_values = norm.pdf(x)
cdf_values = norm.cdf(x)

plt.plot(x, pdf_values, label='PDF')
plt.plot(x, cdf_values, label='CDF')
plt.legend()
plt.title('Questão 2.11 - Gráficos PDF e CDF')
plt.show()

# Questão 2.12 - Gráfico de PMF para a distribuição binomial
from scipy.stats import binom

n, p = 10, 0.5  # parâmetros
x_binom = np.arange(0, n+1)
pmf_values = binom.pmf(x_binom, n, p)

plt.bar(x_binom, pmf_values, color='blue', alpha=0.7)
plt.title('Questão 2.12 - PMF da Distribuição Binomial')
plt.xlabel('Número de Sucessos')
plt.ylabel('Probabilidade')
plt.show()

# Questão 2.13 - Amostra aleatória e valores estatísticos t e p
np.random.seed(0)
sample_data = np.random.normal(loc=0, scale=1, size=30)
t_stat, p_value = stats.ttest_1samp(sample_data, 0)
print("Questão 2.13 - Estatística t:", t_stat, "Valor p:", p_value)

# Questão 2.14 - Correlação de Pearson e Spearman com gráfico
np.random.seed(0)
x = np.random.rand(100)
y = 2*x + np.random.rand(100) * 0.1

pearson_corr = stats.pearsonr(x, y)
spearman_corr = stats.spearmanr(x, y)

plt.scatter(x, y, label='Dados')
plt.plot(x, 2*x, color='red', label='Regressão Linear')
plt.legend()
plt.title('Questão 2.14 - Correlação e Regressão Linear')
plt.show()

print("Correlação de Pearson:", pearson_corr)
print("Correlação de Spearman:", spearman_corr)
