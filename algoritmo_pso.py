import numpy as np
import pandas as pd
import random as rd
import time

class Particula(object):
    def __init__(self, dimencao: int, inicializarPos: tuple = (0,1)) -> None:
        self.posicao = np.random.uniform(inicializarPos[0], inicializarPos[1], dimencao)
        self.velocidade = np.random.uniform(-abs(inicializarPos[1]-inicializarPos[0]),abs(inicializarPos[1]-inicializarPos[0]),dimencao)
        self.melhorPosicao = self.posicao

    def __str__(self) -> str:
        return "Posicao: "+ str(self.posicao) + " | Velocidade: "+ str(self.velocidade) + " | Melhor posicao: " + str(self.melhorPosicao)

# Função a ser otimizada (minimizada) ====================================================================

dim = 10 #dimenção D do problema (tamanho do vetor)

shiftDataLoad = pd.read_table("shift_data_1.txt",delimiter='\s+', index_col=False, header=None)
shiftData = shiftDataLoad.values.reshape((-1))[:dim]

matrixDataLoad = pd.read_table("M_1_D10.txt",delimiter='\s+', index_col=False, header=None)
matrixData = matrixDataLoad.values

def f1_elliptic__(solution=None):
    result = 0
    for i in range(len(solution)):
        result += (10**6)**(i/(len(solution)-1)) * solution[i]**2
    return result

def func_objetivo(solution=None, bias = 100):
    z = np.dot(solution - shiftData, matrixData)
    return f1_elliptic__(z) + bias

# Froneiras do espaço de busca
limInferior = -100
limSuperior = 100

# Parâmetros do algoritmo
w = 0.11 #inercia
phiP = 1.6 #coeficiente cognitivo
phiG = 2.1 #coeficiente social 
numIndividuos = 50
maxIter = 1000*dim
Vmax = 100
repeticoes = 1
inicio = time.time()

solRepeticoes = []
for repeticao in range(repeticoes):
  print("Repetição ",repeticao+1)

  # Inicializar População  =================================================================================
  populacao = [Particula(dimencao=dim, inicializarPos=(limInferior,limSuperior)) for x in range(numIndividuos)]
  # Inicializar melhor partícula
  melhorParticula = None
  for particula in populacao:
      melhorParticula = particula
      for vizinhos in populacao:
          if func_objetivo(vizinhos.melhorPosicao) < func_objetivo(melhorParticula.posicao):
              melhorParticula = vizinhos

  # Loop Principal =========================================================================================
  melhoresSolucoes = []
  iteracao = 0
  criterioParadaFlag = False
  while not criterioParadaFlag:
    iteracao+=1
    # Critério de Parada ----------------------------------------------------------------------------
    if (abs(200 - func_objetivo(melhorParticula.posicao)) <= 10**(-8)) or iteracao >= maxIter :
      criterioParadaFlag = True
    else:
      # Atualizar população ---------------------------------------------------------------------------
      for particula in populacao:

        # atualizar melhor posições de cada indivíduo
        if func_objetivo(particula.posicao) < func_objetivo(particula.melhorPosicao):
          particula.melhorPosicao = particula.posicao

        # Atualizar melhor Partícula 
        melhorParticula = particula
        # Checar se melhor partícula é de fato a melhor partícula
        for vizinhos in populacao:
          if func_objetivo(vizinhos.posicao) < func_objetivo(melhorParticula.posicao):
            melhorParticula = vizinhos

        # Atualizar velocidade e posição
        for componente in range(dim):
          rp = rd.uniform(0,1)
          rg = rd.uniform(0,1)
          #rp = 1
          #rg = 1
          particula.velocidade[componente] = (w* particula.velocidade[componente] 
                                              + phiP*rp*(particula.melhorPosicao[componente] - particula.posicao[componente]) 
                                              + phiG*rg*(melhorParticula.posicao[componente] - particula.posicao[componente]))
          # Verificar intervalo (-Vmax,Vmax)
          if particula.velocidade[componente] > Vmax:
            particula.velocidade[componente] = Vmax
          if particula.velocidade[componente] < -Vmax:
            particula.velocidade[componente] = -Vmax
        particula.posicao += particula.velocidade

      print("iteração ",iteracao)
    melhoresSolucoes.append(melhorParticula.posicao)
  solRepeticoes.append(melhoresSolucoes[-1])
fim = time.time()

# Imprimir resultados =======================================================
print("Média das repetições:",np.mean([func_objetivo(x) for x in solRepeticoes]))
print("Desvio padrão das repetições:",np.std([func_objetivo(x) for x in solRepeticoes]))

# Imprimir tempo de execução =================================================
print()
if (fim-inicio<=60):
    print(fim - inicio,"segundos")
else:
    print((fim - inicio)/60,"minutos")


