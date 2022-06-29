import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd
import time


class Particula(object):
    def __init__(self, dimensao: int, func_fitness: callable, inicializar_pos: tuple = (-1, 1)) -> None:
        self.posicao = np.random.uniform(inicializar_pos[0], inicializar_pos[1], dimensao)
        self.velocidade = np.random.uniform(-abs(inicializar_pos[1] - inicializar_pos[0]),
                                            abs(inicializar_pos[1] - inicializar_pos[0]), dimensao)
        self.melhor_posicao = copy.copy(self.posicao)
        self.melhor_fitness = func_fitness(self.melhor_posicao)
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
def func_objetivo_1(solution, bias=100):
    z = np.dot(solution - shiftData1, matrixData1)
    return f1_elliptic__(z) + bias


# F2
def func_objetivo_2(solution, bias=200):
    z = np.dot(solution - shiftData2, matrixData2)
    return f2_bent_cigar__(z) + bias


# Função para otimizar a função objetivo ===============================================================================
def otimiza(func_fitness: callable, dimensao: int, w: float, phi_p: float, phi_g: float,
            num_particulas: int, max_iter: int, v_max: float, otimo_global: float, gbest_mutation: bool) -> float and int:

    # Inicializar enxame
    enxame = [Particula(dimensao=dimensao,
                        func_fitness=func_fitness,
                        inicializar_pos=(LIMITE_INFERIOR, LIMITE_SUPERIOR)) for _ in range(num_particulas)]
    # Inicializar melhor partícula
    melhor_particula_index = 0
    for particula in range(num_particulas):
        if enxame[particula].fitness < func_fitness(enxame[melhor_particula_index].posicao):
            melhor_particula_index = particula
    global_fitness = func_fitness(enxame[melhor_particula_index].posicao)

    # Encontrar melhor solução
    iteracao = 0
    criterio_parada_flag = False
    melhores_fitness = []
    while not criterio_parada_flag:

        # Avaliar critério de parada
        if (abs(otimo_global - global_fitness) < 10 ** -8) or iteracao >= max_iter:
            criterio_parada_flag = True
        else:
            iteracao += 1
            # Atualizar melhor posição de cada partícula
            for particula in range(num_particulas):
                if enxame[particula].fitness < enxame[particula].melhor_fitness:
                    enxame[particula].melhor_posicao = copy.copy(enxame[particula].posicao)
                    enxame[particula].melhor_fitness = enxame[particula].fitness

            # Atualizar melhor particula global
            for particulas in range(numParticulas):
                if enxame[particulas].melhor_fitness < global_fitness:
                    melhor_particula_index = particulas
                    global_fitness = enxame[melhor_particula_index].melhor_fitness

            # Atualizar velocidade e posição
            for particula in range(numParticulas):
                for componente in range(dimensao):
                    rp = rd.uniform(0, 1)
                    rg = rd.uniform(0, 1)
                    enxame[particula].velocidade[componente] = (w * enxame[particula].velocidade[componente] +
                                                                phi_p * rp *
                                                                (enxame[particula].melhor_posicao[componente] -
                                                                 enxame[particula].posicao[componente])
                                                                + phi_g * rg *
                                                                (enxame[melhor_particula_index].melhor_posicao[
                                                                     componente] -
                                                                 enxame[particula].posicao[componente]))
                    # Verificar intervalo (-Vmax,Vmax)
                    if enxame[particula].velocidade[componente] > v_max:
                        enxame[particula].velocidade[componente] = v_max
                    elif enxame[particula].velocidade[componente] < - v_max:
                        enxame[particula].velocidade[componente] = - v_max

                # Atualizar posição da particula
                enxame[particula].posicao += enxame[particula].velocidade

            # Atualizar fitness de cada partícula
            for particula in range(num_particulas):
                enxame[particula].fitness = func_fitness(enxame[particula].posicao)

            # Mutação em gbest
            if gbest_mutation:
                if rd.uniform(0, 1) < 1 / dimensao:
                    n_normal = np.random.normal()
                    tau = 1 / np.sqrt(2 * num_particulas)
                    tau_linha = 1 / np.sqrt(2 * np.sqrt(num_particulas))
                    particula_mutante = np.zeros(dimensao)
                    for componente in range(dimensao):
                        n_normal_componente = np.random.normal()
                        beta_linha = 3 * np.exp(tau * n_normal + tau_linha * n_normal_componente)
                        particula_mutante[componente] = enxame[melhor_particula_index].melhor_posicao[
                                                            componente] + beta_linha * np.random.beta(0.5, 0.5)

                    if func_fitness(particula_mutante) < global_fitness:
                        enxame[melhor_particula_index].melhor_posicao = particula_mutante
                        global_fitness = func_fitness(particula_mutante)


    return global_fitness, iteracao, melhores_fitness


# Parâmetros do algoritmo ==============================================================================================
W = 0.6  # inercia
phiP = 1.0  # coeficiente cognitivo
phiG = 2.0  # coeficiente social
numParticulas = 50
maxIter = 100000
Vmax = 1
repeticoes = 1

inicio = time.time()

solRepeticoes = []  # Lista para armazenar a melhor solução de cada repetição

for repeticao in range(repeticoes):
    melhor_fitness, iteracao_limite, melhores_fitness = otimiza(func_fitness=func_objetivo_2,
                                                                dimensao=DIMENSAO,
                                                                w=W,
                                                                phi_p=phiP,
                                                                phi_g=phiG,
                                                                num_particulas=numParticulas,
                                                                max_iter=maxIter,
                                                                v_max=Vmax,
                                                                otimo_global=200,
                                                                gbest_mutation=True)

fim = time.time()

# Imprimir resultados ==================================================================================================
print("Média das repetições: {:.8f}".format(np.mean(solRepeticoes)))
print("Desvio padrão das repetições: {:.2f}".format(np.std(solRepeticoes)))

# Imprimir tempo de execução ===========================================================================================
if fim - inicio <= 60:
    print(fim - inicio, "segundos")
else:
    print((fim - inicio) / 60, "minutos")
