import numpy as np
import pandas as pd
import random as rd
import time


class Particula(object):
    def __init__(self, dimencao: int, inicializarPos: tuple = (0, 1)) -> None:
        self.posicao = np.random.uniform(inicializarPos[0], inicializarPos[1], dimencao)
        self.velocidade = np.random.uniform(-abs(inicializarPos[1]-inicializarPos[0]), abs(inicializarPos[1]-inicializarPos[0]), dimencao)
        self.melhorPosicao = self.posicao

    def __str__(self) -> str:
        return "Posicao: " + str(self.posicao) + " | Velocidade: " + str(self.velocidade) + " | Melhor posicao: " + str(self.melhorPosicao)

# Função a ser otimizada (minimizada) ====================================================================


dim = 10 #dimenção D do problema (tamanho do vetor solução)

# Shift Data
shiftData1Load = pd.read_table("shift_data_1.txt",delimiter='\s+', index_col=False, header=None)
shiftData2Load = pd.read_table("shift_data_2.txt",delimiter='\s+', index_col=False, header=None)
shiftData1 = shiftData1Load.values.reshape((-1))[:dim]
shiftData2 = shiftData2Load.values.reshape((-1))[:dim]

# Matrix Data
matrixData1Load = pd.read_table("M_1_D10.txt",delimiter='\s+', index_col=False, header=None)
matrixData2Load = pd.read_table("M_2_D10.txt",delimiter='\s+', index_col=False, header=None)
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
# def func_objetivo(solution,bias=100):
#     z = np.dot(solution- shiftData1, matrixData1)
#     return f1_elliptic__(z) + bias
# F2
def func_objetivo(solution,bias=200):
    z = np.dot(solution- shiftData2, matrixData2)
    return f2_bent_cigar__(z) + bias


# Fronteiras do espaço de busca
limInferior = -100
limSuperior = 100

# Parâmetros do algoritmo
w = 0.729  # inercia
phiP = 1.49445  # coeficiente cognitivo
phiG = 1.49445  # coeficiente social
numParticulas = 50
maxIter = 10000
# Vmax = np.inf
Vmax = 100
repeticoes = 30

wInicial = 0.6
wFinal = 0.1

inicio = time.time()

solRepeticoes = []
for repeticao in range(repeticoes):

    # Inicializar População  =================================================================================
    populacao = [Particula(dimencao=dim, inicializarPos=(limInferior, limSuperior)) for x in range(numParticulas)]
    # Inicializar melhor partícula
    melhorParticulaIndex = 0
    for particula in range(numParticulas):
        melhorParticula = particula
        for vizinhos in range(numParticulas):
            if func_objetivo(populacao[vizinhos].posicao) < func_objetivo(populacao[melhorParticulaIndex].posicao):
                melhorParticulaIndex = vizinhos
    melhorFitness = func_objetivo(populacao[melhorParticulaIndex].posicao)
    melhorPos = populacao[melhorParticulaIndex].melhorPosicao

    # Loop Principal =========================================================================================
    iteracao = 0
    criterioParadaFlag = False
    while not criterioParadaFlag:

        # Critério de Parada ----------------------------------------------------------------------------
        if (abs(200 - melhorFitness) < 10 ** (-8)) or iteracao >= maxIter:
            criterioParadaFlag = True
        else:
            # Atualizar população ---------------------------------------------------------------------------
            wt = (wInicial - wFinal) * ((maxIter - iteracao) / maxIter) + wFinal  # Decremento linear de W
            iteracao += 1

            # Atualizar melhor posições de cada indivíduo
            for particula in range(numParticulas):
                if func_objetivo(populacao[particula].posicao) < func_objetivo(populacao[particula].melhorPosicao):
                    populacao[particula].melhorPosicao = populacao[particula].posicao

            # Atualizar melhor particula global
            for vizinhos in range(numParticulas):
                if func_objetivo(populacao[vizinhos].melhorPosicao) < melhorFitness:
                    melhorParticulaIndex = vizinhos
                    melhorFitness = func_objetivo(populacao[melhorParticulaIndex].melhorPosicao)
                    melhorPos = populacao[melhorParticulaIndex].melhorPosicao

            # Atualizar velocidade e posição
            for particula in range(numParticulas):
                for componente in range(dim):
                    rp = rd.uniform(0, 1)
                    rg = rd.uniform(0, 1)
                    # rp = 1
                    # rg = 1
                    populacao[particula].velocidade[componente] = (wt * populacao[particula].velocidade[componente]
                                                                   + phiP * rp * (populacao[particula].melhorPosicao[
                                                                                      componente] -
                                                                                  populacao[particula].posicao[
                                                                                      componente])
                                                                   + phiG * rg * (melhorPos[componente] -
                                                                                  populacao[particula].posicao[
                                                                                      componente]))
                    # Verificar intervalo (-Vmax,Vmax)
                    if populacao[particula].velocidade[componente] > Vmax:
                        populacao[particula].velocidade[componente] = Vmax
                    elif populacao[particula].velocidade[componente] < - Vmax:
                        populacao[particula].velocidade[componente] = - Vmax
                    populacao[particula].posicao[componente] += populacao[particula].velocidade[componente]
                    # if populacao[particula].posicao[componente] > limSuperior:
                    #     populacao[particula].posicao[componente] = limSuperior
                    # elif populacao[particula].posicao[componente] < limInferior:
                    #     populacao[particula].posicao[componente] = limInferior
            # if iteracao % 100 == 0:
            # print("Repetição: {} | Iteração: {} | Melhor FItness: {}".format(repeticao + 1, iteracao, melhorFitness))

            # print("{}".format(iteracao))
    solRepeticoes.append(melhorFitness)
    print("Repetição: {} | Iteração Máxima: {} | Melhor FItness: {}".format(repeticao + 1, iteracao, melhorFitness))
fim = time.time()

# Imprimir resultados =======================================================
print("Média das repetições: {:.8f}\n".format(np.mean(solRepeticoes)), end="")
print("Desvio padrão das repetições: {:.2f}\n".format(np.std(solRepeticoes)), end="")

# Imprimir tempo de execução =================================================
if (fim - inicio <= 60):
    print(fim - inicio, "segundos")
else:
    print((fim - inicio) / 60, "minutos")