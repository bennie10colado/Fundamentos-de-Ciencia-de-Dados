import numpy as np

# func aux
def print_section(title, content=None):
    print("\n\n\n" + "="*50)
    print(f"{title.center(50)}")
    print("="*50)
    if content is not None:
        print(content)
    print("="*50 + "\n")


# Parte 1: NumPy

# Questão 1.1 - Criar um array com 3 elementos distintos
array1 = np.array([1, 2, 3])
print_section("Questão 1.1 - Array com 3 elementos distintos", array1)

# Questão 1.2 - Criar um array de zeros
array_zeros = np.zeros(5)
print_section("Questão 1.2 - Array de zeros", array_zeros)

# Questão 1.3 - Criar um array de uns
array_uns = np.ones(5)
print_section("Questão 1.3 - Array de uns", array_uns)

# Questão 1.4 - Criar um array de 4 elementos arbitrários
array_arbitrario = np.array([10, 20, 30, 40])
print_section("Questão 1.4 - Array com 4 elementos arbitrários", array_arbitrario)

# Questão 1.5 - Criar um array a partir de um intervalo de números (sequência)
array_intervalo = np.arange(10, 21)
print_section("Questão 1.5 - Array com intervalo de números", array_intervalo)

# Questão 1.6 - Criar um array com intervalo de números e sequência de 3 em 3
array_sequencia = np.arange(0, 30, 3)
print_section("Questão 1.6 - Array com sequência de 3 em 3", array_sequencia)

# Questão 1.7 - Explicar o resultado de np.linspace(0, 22, 5)
linspace_result = np.linspace(0, 22, 5)
print_section("Questão 1.7 - Resultado de np.linspace(0,22,5)", linspace_result)
print("***OBS: Explicação é que essa funcao gera 5 valores igualmente espaçados entre 0 e 22.\n")

# Questão 1.8 - Criar um array de 21 elementos e exibir informações
array_21 = np.arange(21)
print_section("Questão 1.8 - Informações sobre o array")
print(f"Tipo: {array_21.dtype}\nNúmero de elementos: {array_21.size}\n"
      f"Consumo de bytes por elemento: {array_21.itemsize}\nDimensões: {array_21.ndim}")

# Questão 1.9 - Lista de 3 listas com 3 elementos cada
lista_A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print_section("Questão 1.9 - Lista A", lista_A)

# Questão 1.10 - Array multidimensional
multi_B = np.array(lista_A)
print_section("Questão 1.10 - Array multidimensional multi_B", multi_B)

# Questão 1.11 - Número de elementos em cada dimensão
print_section("Questão 1.11 - Número de elementos em cada dimensão", multi_B.shape)

# Questão 1.12 - Usar reshape para transformar uma lista de 12 elementos em (3,4)
array_12 = np.arange(12).reshape(3, 4)
print_section("Questão 1.12 - Array reshaped (3,4)", array_12)

# Questão 1.13 - Transformar o array anterior em uma lista de 12 elementos
array_flat = array_12.reshape(-1)
print_section("Questão 1.13 - Array reshape em lista de 12 elementos", array_flat)

# Questão 1.14 - Transformar em um array (2,2,3)
array_2x2x3 = array_flat.reshape(2, 2, 3)
print_section("Questão 1.14 - Array reshape em (2,2,3)", array_2x2x3)

# Questão 1.19 - Operações com duas listas
lista1 = [1, 2, 3]
lista2 = [4, 5, 6]
soma = np.add(lista1, lista2)
multiplicacao = np.multiply(lista1, lista2)
divisao = np.divide(lista1, lista2)
print_section("Questão 1.19 - Operações com duas listas")
print(f"Soma: {soma}\nMultiplicação: {multiplicacao}\nDivisão: {divisao}")

# Questão 1.21 - Polinômios usando poly1d
p1 = np.poly1d([3, 4])
p2 = np.poly1d([4, 3, 2, 1])
p3 = np.poly1d([2, 0, 0, 3])
p4 = np.poly1d([1, 2, 3])
multiplicacao_polinomios = np.polymul(p3, p4)
derivada_p4 = np.polyder(p4)
integral_p4 = np.polyint(p4)
print_section("Questão 1.21 - Polinômios e operações")
print(f"p1(x):\n{p1}\n\n")
print(f"p2(x):\n{p2}\n\n")
print(f"p3(x):\n{p3}\n\n")
print(f"p4(x):\n{p4}\n\n")
print(f"Multiplicação dos polinômios p3(x) * p4(x):\n{multiplicacao_polinomios}\n")
print(f"Derivada de p4(x):\n{derivada_p4}\n")
print(f"Integral de p4(x):\n{integral_p4}")
