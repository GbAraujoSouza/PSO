import copy
import numpy as np
import pandas as pd
import random as rd
import time
from scipy.stats import variation


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
shiftData1Load = pd.read_table("./arquivos_suporte/shift_data_1.txt", delimiter='\s+', index_col=False, header=None)
shiftData2Load = pd.read_table("./arquivos_suporte/shift_data_2.txt", delimiter='\s+', index_col=False, header=None)
shiftData1 = shiftData1Load.values.reshape((-1))[:DIMENSAO]
shiftData2 = shiftData2Load.values.reshape((-1))[:DIMENSAO]

matrixData1Load = pd.read_table("./arquivos_suporte/M_1_D10.txt", delimiter='\s+', index_col=False, header=None)
matrixData2Load = pd.read_table("./arquivos_suporte/M_2_D10.txt", delimiter='\s+', index_col=False, header=None)
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


# Função para imprimir barra de progresso
def barra_progresso(progresso: int, total: int) -> None:
    porcentagem = 100 * (progresso / float(total))
    barra = '█' * int(porcentagem) + '-' * (100 - int(porcentagem))
    print("\r|{}| {:.2f}%".format(barra, porcentagem), end="\r")


# Função para otimizar a função objetivo ===============================================================================
def otimiza(func_fitness: callable, dimensao: int, phi_p: float, phi_g: float,
            num_particulas: int, max_iter: int, v_max: float, otimo_global: float, gbest_mutation_beta: bool, pbest_mutation_beta: bool, w: float | tuple, metodo_inercia: str = "static") -> float and int:

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
    contador_mutacao_cv = 0
    criterio_parada_flag = False
    historico_posicoes = {'p{}'.format(x): [] for x in range(num_particulas)}
    while not criterio_parada_flag:

        # Avaliar critério de parada
        if (abs(otimo_global - global_fitness) < 10 ** -8) or iteracao >= max_iter:
            criterio_parada_flag = True
        else:
            
            # Atualizar melhor posição de cada partícula
            for particula in range(num_particulas):
                if enxame[particula].fitness < enxame[particula].melhor_fitness:
                    enxame[particula].melhor_posicao = copy.copy(enxame[particula].posicao)
                    enxame[particula].melhor_fitness = enxame[particula].fitness

            # Atualizar melhor particula global
            for particula in range(numParticulas):
                if enxame[particula].melhor_fitness < global_fitness:
                    melhor_particula_index = particula
                    global_fitness = enxame[melhor_particula_index].melhor_fitness

            # Verificar metodo de atualizacao do coeficiente de inercia
            if metodo_inercia == "linear":
                if type(w) != tuple:
                    raise TypeError('Para inercia linear, \'w\' precisar ser do tipo (w_inicial, w_final)')
                inercia = (w[0] - w[1]) * ((max_iter - iteracao) / max_iter) + w[1]
            
            elif metodo_inercia == "static":
                if type(w) not in (float, int):
                    raise TypeError('Método static de inercia requer valor numérico em \'w\'')
                inercia = w
            
            else:
                raise TypeError('\'metodo_inercia\' precisa ser \"static\" ou \"linear\"')
            
            # Atualizar velocidade e posição
            for particula in range(numParticulas):
                for componente in range(dimensao):
                    rp = rd.uniform(0, 1)
                    rg = rd.uniform(0, 1)
                    enxame[particula].velocidade[componente] = (inercia * enxame[particula].velocidade[componente] +
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
                enxame[particula].fitness = func_fitness(enxame[particula].posicao)


                # Armazenar historico da melhor posicao de cada particula para avaliar o CV
                historico_posicoes[f'p{particula}'].append(enxame[particula].melhor_posicao)
            
            # Mutação beta em gbest
            if gbest_mutation_beta:
                if rd.uniform(0, 1) < 1 / dimensao:
                    n_normal = np.random.normal()
                    tau = 1 / np.sqrt(2 * num_particulas)
                    tau_linha = 1 / np.sqrt(2 * np.sqrt(num_particulas))
                    for componente in range(dimensao):
                        n_normal_componente = np.random.normal()
                        beta_linha = 3 * np.exp(tau * n_normal + tau_linha * n_normal_componente)
                        enxame[melhor_particula_index].melhor_posicao[componente] += beta_linha * np.random.beta(0.5, 0.5)
                    global_fitness = func_fitness(enxame[melhor_particula_index].melhor_posicao)
                    for particula in range(particula):
                        if enxame[particula].melhor_fitness < global_fitness:
                                melhor_particula_index = particula
                                global_fitness = enxame[melhor_particula_index].melhor_fitness
            
            # Mutação beta em pbest
            if pbest_mutation_beta:
                for particula in range(num_particulas):
                    if rd.uniform(0, 1) < 1 / dimensao:
                        n_normal = np.random.normal()
                        tau = 1 / np.sqrt(2 * num_particulas)
                        tau_linha = 1 / np.sqrt(2 * np.sqrt(num_particulas))
                        for componente in range(dimensao):
                            n_normal_componente = np.random.normal()
                            beta_linha = 3 * np.exp(tau * n_normal + tau_linha * n_normal_componente)
                            enxame[particula].melhor_posicao[componente] += rd.betavariate(0.5, 0.5) * beta_linha
                        enxame[particula].melhor_fitness = func_fitness(enxame[particula].melhor_posicao)
                        if enxame[particula].fitness < enxame[particula].melhor_fitness:
                            enxame[particula].melhor_posicao = copy.copy(enxame[particula].posicao)
                            enxame[particula].melhor_fitness = enxame[particula].fitness
                        if enxame[particula].melhor_fitness < global_fitness:
                            melhor_particula_index = particula
                            global_fitness = enxame[melhor_particula_index].melhor_fitness

            # Mutacao a partir do CV
            # percebeu-se que a partir da iteracao 200 as componentes variavam pouco
            if contador_mutacao_cv == 500:
                for particula in range(num_particulas):
                    for componente in range(dimensao):
                        if abs(variation(historico_posicoes[f'p{particula}'])[componente]) < 0.01:
                            enxame[particula].melhor_posicao[componente] += enxame[particula].melhor_posicao[componente]*rd.uniform(0, 1)
                    historico_posicoes[f'p{particula}'] = []
                contador_mutacao_cv = 0

            iteracao += 1
            contador_mutacao_cv += 1
            barra_progresso(iteracao, max_iter)
    
    return global_fitness, iteracao


# Parâmetros do algoritmo ==============================================================================================
W = 0.6  # inercia
phiP = 1.0  # coeficiente cognitivo
phiG = 2.0  # coeficiente social
numParticulas = 100
maxAvaliacao = 100000
maxIter = maxAvaliacao // numParticulas
Vmax = np.inf
repeticoes = 30

inicio = time.time()

solRepeticoes = []  # Lista para armazenar a melhor solução de cada repetição

for repeticao in range(repeticoes):
    melhor_fitness, iteracao_limite = otimiza(func_fitness=func_objetivo_1, 
                                              dimensao=DIMENSAO,
                                              w=W,
                                              metodo_inercia="static",
                                              phi_p=phiP,
                                              phi_g=phiG,
                                              num_particulas=numParticulas,
                                              max_iter=maxIter,
                                              v_max=Vmax,
                                              otimo_global=100,
                                              gbest_mutation_beta=False,
                                              pbest_mutation_beta=False)

    solRepeticoes.append(melhor_fitness)
    print("\nRepetição: {:>2} | Iteração Máxima: {:>6} | Melhor Fitness: {:.9f}".format(repeticao + 1, iteracao_limite,
                                                                                melhor_fitness))



fim = time.time()

# Imprimir resultados ==================================================================================================
print("Média das repetições: {:.9f}".format(np.mean(solRepeticoes)))
print("Desvio padrão das repetições: {:.2f}".format(np.std(solRepeticoes)))

# Imprimir tempo de execução ===========================================================================================
if fim - inicio <= 60:
    print(fim - inicio, "segundos")
else:
    print((fim - inicio) / 60, "minutos")