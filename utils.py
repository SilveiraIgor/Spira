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

def get_arquivo_audios():
    df = pd.read_csv("../dados_spira/clean/Metadados/arquivo2.csv")
    return df

#--------------------------------------------------
def criar_dic():
  return {0: {'X': [], 'Y':[], 'data': [], 'id': []},
          1: {'X': [], 'Y':[], 'data': [], 'id': []},
          2: {'X': [], 'Y':[], 'data': [], 'id': []},
          3: {'X': [], 'Y':[], 'data': [], 'id': []},
          4: {'X': [], 'Y':[], 'data': [], 'id': []},
          5: {'X': [], 'Y':[], 'data': [], 'id': []}}

def gerar_rotulos(linha_pesquisa):
  if linha_pesquisa == "Controle":
    return 0
  elif linha_pesquisa == "Asma":
    return 1
  elif linha_pesquisa == "Insuficiência Respiratória":
    return 2
  elif linha_pesquisa == "Tabagismo":
    return 3
  elif linha_pesquisa == "Parkinson":
    return 4
  else:
    assert True == False, "Linha de pesquisa desconhecida"

def criar_csv_preds(df, preds, dic_filtrado, nome_csv, nome_colunas):
  df[nome_colunas[0]] = None
  df[nome_colunas[1]] = None
  for i in range(5):
    pred_split = preds[i]
    for indice in range(len(pred_split)):
      data_valor = dic_filtrado[i]['data'][indice]
      id_valor = dic_filtrado[i]['id'][indice]
      #print(data_valor, id_valor)
      filtro = (df["id"] == id_valor) & (df["data_coleta"] == data_valor)
      df.loc[filtro, nome_colunas[0]] = pred_split[indice][0]
      df.loc[filtro, nome_colunas[1]] = pred_split[indice][1]
  df.to_csv(nome_csv, index=False)

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

def gerar_log_saida(NOME_RUN,listas, colunas):
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


