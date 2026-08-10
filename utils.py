import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, cohen_kappa_score

#Funções responsáveis por puxar os áudios
def get_files(folder_path):
    return [str(p) for p in Path(folder_path).iterdir() if p.is_file()]
def filtrar_audios(lista, terminacao_desejada):
    print("Tamanho da lista antes da filtragem: ", len(lista))
    nova_lista = []
    for opcao in lista:
        if opcao.endswith(terminacao_desejada) and not("TEST" in opcao):
            nova_lista.append(opcao)
    print("Tamanho da lista pós filtragem: ", len(nova_lista))
    return nova_lista
def separar_listas(lista, grupo1, grupo2):
    lista_grupo1, lista_grupo2 = [], []
    for elemento in lista:
        if grupo1 in elemento:
            lista_grupo1.append(elemento)
        elif grupo2 in elemento:
            lista_grupo2.append(elemento)
    print(f"-- O grupo {grupo1} tem {len(lista_grupo1)} elementos, {grupo2} tem {len(lista_grupo2)} ")
    return lista_grupo1, lista_grupo2
#--------------------------------------------------

def test_perf_model(Y_hat, Y, tipo="regressao"):
  if tipo == "regressao":
    discretized_outputs = (Y_hat >= 0.5).float()
  else:
    discretized_outputs = Y_hat.argmax(dim=1)
  # Compare with the Y_tensor and count where they are equal
  matches = (discretized_outputs == Y).sum().item()
  total_elements = Y.numel()
  acc = matches / total_elements * 100
  kappa = cohen_kappa_score(discretized_outputs, Y)
  return acc, kappa

def gerar_log_saida(listas, colunas):
    log_saida = []
    for indice_lista in range(len(listas[0])):
        dic = {}
        for metrica in range(len(listas)):
            nome_metrica = colunas[metrica]
            valor_metrica = listas[metrica][indice_lista]
            dic[nome_metrica] = valor_metrica
        log_saida.append(dic)
    df = pd.DataFrame(log_saida)
    df.to_csv(NOME_RUN, index=False)


