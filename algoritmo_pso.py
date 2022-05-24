import copy
import numpy as np
import pandas as pd
import random as rd
import time


class Particula(object):
    def __init__(self, dimencao: int, func_fitness, inicializar_pos: tuple = (0, 1)) -> None:
        self.posicao = np.random.uniform(inicializar_pos[0], inicializar_pos[1], dimencao)
        self.velocidade = np.random.uniform(-abs(inicializar_pos[1] - inicializar_pos[0]),
                                            abs(inicializar_pos[1] - inicializar_pos[0]), dimencao)
        self.melhorPosicao = copy.copy(self.posicao)
        self.melhorFitness = func_fitness(self.melhorPosicao)
        self.fitness = func_fitness(self.posicao)

    def __str__(self) -> str:
        return ("Posicao: " + str(self.posicao) + " | Velocidade: " + str(self.velocidade) + " | Melhor posicao: " +
                str(self.melhorPosicao))


# Função a ser otimizada (minimizada) ====================================================================
dim = 10  # dimensão D do problema (tamanho do vetor solução)

# Shift Data
shiftData1Load = pd.read_table("shift_data_1.txt", delimiter='\s+', index_col=False, header=None)
shiftData2Load = pd.read_table("shift_data_2.txt", delimiter='\s+', index_col=False, header=None)
shiftData1 = shiftData1Load.values.reshape((-1))[:dim]
shiftData2 = shiftData2Load.values.reshape((-1))[:dim]

# Matrix Data
matrixData1Load = pd.read_table("M_1_D10.txt", delimiter='\s+', index_col=False, header=None)
matrixData2Load = pd.read_table("M_2_D10.txt", delimiter='\s+', index_col=False, header=None)
matrixData1 = matrixData1Load.values
matrixData2 = matrixData2Load.values


# Funções básicas

def f2_bent_cigar__(solution=None):
    return solution[0]**2 + 10**6 * sum(solution[1:]**2)


def f1_elliptic__(solution=None):
    result = 0
    for i in range(len(solution)):
        result += (10**6)**(i/(len(solution)-1)) * solution[i]**2
    return result


# F1
def func_objetivo(solution,bias=100):
    z = np.dot(solution- shiftData1, matrixData1)
    return f1_elliptic__(z) + bias
# F2
# def func_objetivo(solution, bias=200):
#     z = np.dot(solution - shiftData2, matrixData2)
#     return f2_bent_cigar__(z) + bias


# Fronteiras do espaço de busca
limInferior = -100
limSuperior = 100

# Parâmetros do algoritmo
w = 0.6  # inercia
phiP = 1.1  # coeficiente cognitivo
phiG = 2.1  # coeficiente social
numParticulas = 50
maxIter = 20000
# Vmax = np.inf
Vmax = 5
repeticoes = 1

wInicial = 0.9
wFinal = 0.4

inicio = time.time()

solRepeticoes = []
for repeticao in range(repeticoes):

    # Inicializar População  =================================================================================
    populacao = [Particula(dimencao=dim, inicializar_pos=(limInferior, limSuperior), func_fitness=func_objetivo) for x in range(numParticulas)]
    # Inicializar melhor partícula
    melhorParticulaIndex = 0
    for particula in range(numParticulas):
        melhorParticula = particula
        for vizinhos in range(numParticulas):
            if populacao[vizinhos].fitness < func_objetivo(populacao[melhorParticulaIndex].posicao):
                melhorParticulaIndex = vizinhos
    globalFitness = func_objetivo(populacao[melhorParticulaIndex].posicao)

    # Loop Principal =========================================================================================
    iteracao = 0
    criterioParadaFlag = False
    while not criterioParadaFlag:

        # Critério de Parada ----------------------------------------------------------------------------
        if (abs(100 - globalFitness) < 10 ** (-8)) or iteracao >= maxIter:
            criterioParadaFlag = True
        else:
            # Atualizar população ---------------------------------------------------------------------------
            wt = (wInicial - wFinal) * ((maxIter - iteracao) / maxIter) + wFinal  # Decremento linear de W
            iteracao += 1

            # Atualizar melhor posições de cada indivíduo
            for particula in range(numParticulas):
                if populacao[particula].fitness < populacao[particula].melhorFitness:
                    populacao[particula].melhorPosicao = copy.copy(populacao[particula].posicao)
                    populacao[particula].melhorFitness = populacao[particula].fitness

            # Atualizar melhor particula global
            for vizinhos in range(numParticulas):
                if populacao[vizinhos].melhorFitness < globalFitness:
                    melhorParticulaIndex = vizinhos
                    globalFitness = populacao[melhorParticulaIndex].melhorFitness

            # Atualizar velocidade e posição
            for particula in range(numParticulas):
                for componente in range(dim):
                    rp = rd.uniform(0, 1)
                    rg = rd.uniform(0, 1)
                    # rp = 1
                    # rg = 1
                    populacao[particula].velocidade[componente] = (w * populacao[particula].velocidade[componente]
                                                                   + phiP * rp * (populacao[particula].melhorPosicao[
                                                                                      componente] -
                                                                                  populacao[particula].posicao[
                                                                                      componente])
                                                                   + phiG * rg * (populacao[melhorParticulaIndex].melhorPosicao[componente] -
                                                                                  populacao[particula].posicao[
                                                                                      componente]))
                    # Verificar intervalo (-Vmax,Vmax)
                    if populacao[particula].velocidade[componente] > Vmax:
                        populacao[particula].velocidade[componente] = Vmax
                    elif populacao[particula].velocidade[componente] < - Vmax:
                        populacao[particula].velocidade[componente] = - Vmax
                    populacao[particula].posicao[componente] += populacao[particula].velocidade[componente]
                # Atualizar fitness de casa partícula
                populacao[particula].fitness = func_objetivo(populacao[particula].posicao)
            if iteracao % 100 == 0:
                print("Repetição: {} | Iteração: {} | Melhor FItness: {}".format(repeticao + 1, iteracao, globalFitness))

            # print("{}".format(iteracao))
    solRepeticoes.append(globalFitness)
    # print("Repetição: {} | Iteração Máxima: {} | Melhor FItness: {}".format(repeticao + 1, iteracao, globalFitness))
fim = time.time()

# Imprimir resultados =======================================================
print("Média das repetições: {:.8f}\n".format(np.mean(solRepeticoes)), end="")
print("Desvio padrão das repetições: {:.2f}\n".format(np.std(solRepeticoes)), end="")

# Imprimir tempo de execução =================================================
if fim - inicio <= 60:
    print(fim - inicio, "segundos")
else:
    print((fim - inicio) / 60, "minutos")
