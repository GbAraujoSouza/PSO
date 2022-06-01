import copy
import numpy as np
import pandas as pd
import random as rd
import time


class Particula(object):
    def __init__(self, dimensao: int, func_fitness, inicializar_pos: tuple = (-1, 1)) -> None:
        self.posicao = np.random.uniform(inicializar_pos[0], inicializar_pos[1], dimensao)
        self.velocidade = np.random.uniform(-abs(inicializar_pos[1] - inicializar_pos[0]),
                                            abs(inicializar_pos[1] - inicializar_pos[0]), dimensao)
        self.melhorPosicao = copy.copy(self.posicao)
        self.melhorFitness = func_fitness(self.melhorPosicao)
        self.fitness = func_fitness(self.posicao)


# Função objetivo (minimização) ========================================================================================
DIMENSAO = 10  # Dimensão D do problema (tamanho do vetor solução)

# Dados para as transformações das funções cec 2014 (extraídos da biblioteca opfunu)
# Shit Data
shiftData1Load = pd.read_table("shift_data_1.txt", delimiter='\s+', index_col=False, header=None)
shiftData2Load = pd.read_table("shift_data_2.txt", delimiter='\s+', index_col=False, header=None)
shiftData1 = shiftData1Load.values.reshape((-1))[:DIMENSAO]
shiftData2 = shiftData2Load.values.reshape((-1))[:DIMENSAO]

# Matrix Data
matrixData1Load = pd.read_table("M_1_D10.txt", delimiter='\s+', index_col=False, header=None)
matrixData2Load = pd.read_table("M_2_D10.txt", delimiter='\s+', index_col=False, header=None)
matrixData1 = matrixData1Load.values
matrixData2 = matrixData2Load.values

# Fronteiras do espaço de busca
LIMITE_INFERIOR = -100
LIMITE_SUPERIOR = 100


# Funções cec-2014 básicas
def f2_bent_cigar__(solution=None):
    return solution[0]**2 + 10**6 * sum(solution[1:]**2)


def f1_elliptic__(solution=None):
    result = 0
    for i in range(len(solution)):
        result += (10**6)**(i/(len(solution)-1)) * solution[i]**2
    return result


# F1
def func_objetivo(solution, bias=100):
    z = np.dot(solution - shiftData1, matrixData1)
    return f1_elliptic__(z) + bias
# F2
# def func_objetivo(solution, bias=200):
#     z = np.dot(solution - shiftData2, matrixData2)
#     return f2_bent_cigar__(z) + bias


# Função para otimizar a função objetivo ===============================================================================
def otimiza(func_fitness, dimensao, w, phi_p, phi_g,
            num_particulas, max_iter, v_max, otimo_global):

    # Inicializar enxame
    enxame = [Particula(dimensao=dimensao,
                        func_fitness=func_fitness,
                        inicializar_pos=(LIMITE_INFERIOR, LIMITE_SUPERIOR)) for particula in range(num_particulas)]
    # Inicializar melhor partícula
    melhor_particula_index = 0
    for particula in range(num_particulas):
        melhor_particula_index = particula
        if enxame[particula].fitness < func_fitness(enxame[melhor_particula_index].posicao):
            melhor_particula_index = particula
    global_fitness = func_fitness(enxame[melhor_particula_index].posicao)

    # Encontrar melhor solução
    iteracao = 0
    criterio_parada_flag = False
    while not criterio_parada_flag:

        # Avaliar critério de parada
        if (abs(otimo_global - global_fitness) < 10 ** -8) or iteracao >= max_iter:
            criterio_parada_flag = True
        else:
            iteracao += 1
            # Atualizar melhor posição de cada partícula
            for particula in range(num_particulas):
                if enxame[particula].fitness < enxame[particula].melhorFitness:
                    enxame[particula].melhorPosicao = copy.copy(enxame[particula].posicao)
                    enxame[particula].melhorFitness = enxame[particula].fitness

            # Atualizar melhor particula global
            for vizinhos in range(numParticulas):
                if enxame[vizinhos].melhorFitness < global_fitness:
                    melhor_particula_index = vizinhos
                    global_fitness = enxame[melhor_particula_index].melhorFitness

            # Atualizar velocidade e posição
            for particula in range(numParticulas):
                for componente in range(dimensao):
                    rp = rd.uniform(0, 1)
                    rg = rd.uniform(0, 1)
                    enxame[particula].velocidade[componente] = (w * enxame[particula].velocidade[componente] +
                                                                phi_p * rp *
                                                                (enxame[particula].melhorPosicao[componente] -
                                                                 enxame[particula].posicao[componente])
                                                                + phi_g * rg *
                                                                (enxame[melhor_particula_index].melhorPosicao[
                                                                     componente] -
                                                                 enxame[particula].posicao[componente]))
                    # Verificar intervalo (-Vmax,Vmax)
                    if enxame[particula].velocidade[componente] > v_max:
                        enxame[particula].velocidade[componente] = v_max
                    elif enxame[particula].velocidade[componente] < - v_max:
                        enxame[particula].velocidade[componente] = - v_max
                    enxame[particula].posicao[componente] += enxame[particula].velocidade[componente]

                # Atualizar fitness de cada partícula
                enxame[particula].fitness = func_objetivo(enxame[particula].posicao)
            if iteracao % 1000 == 0:
                print("Repetição: {} | Iteração: {} | Melhor Fitness: {}".format(repeticao + 1, iteracao, global_fitness))
    return global_fitness, iteracao


# Parâmetros do algoritmo ==============================================================================================
W = 0.6  # inercia
phiP = 0  # coeficiente cognitivo
phiG = 2  # coeficiente social
numParticulas = 100
maxIter = 20000
Vmax = 1
repeticoes = 1

inicio = time.time()

solRepeticoes = []  # Lista para armazenar a melhor solução de cada repetição

for repeticao in range(repeticoes):
    melhor_fitness, iteracao_limite = otimiza(func_fitness=func_objetivo,
                                              dimensao=DIMENSAO,
                                              w=W,
                                              phi_p=phiP,
                                              phi_g=phiG,
                                              num_particulas=numParticulas,
                                              max_iter=maxIter,
                                              v_max=Vmax,
                                              otimo_global=100)
    solRepeticoes.append(melhor_fitness)
    print("Repetição: {} | Iteração Máxima: {} | Melhor Fitness: {:.8f}".format(repeticao + 1, iteracao_limite,
                                                                                melhor_fitness))

fim = time.time()

# Imprimir resultados ==================================================================================================
print("Média das repetições: {:.8f}".format(np.mean(solRepeticoes)))
print("Desvio padrão das repetições: {:.2f}".format(np.std(solRepeticoes)))

# Imprimir tempo de execução ===========================================================================================
if fim - inicio <= 60:
    print(fim - inicio, "segundos")
else:
    print((fim - inicio) / 60, "minutos")
