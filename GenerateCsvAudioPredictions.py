import torch
import numpy as np
import librosa
from panns_inference import AudioTagging
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils import *
from RedeNeural import NeuralNetwork

at = AudioTagging(checkpoint_path=None, device='cuda')

def gerar_dataset(df, coluna):
    apendice = "../dados_spira/clean/"
    arquivos_disponiveis = get_files(apendice)
    dataset = criar_dic()
    for idx,linha in df.iterrows():
        if not pd.isna(linha[coluna]):
            nome_editado = apendice+linha[coluna].replace("%", "_").replace("?","_")
            if nome_editado not in arquivos_disponiveis:
                print(f"O arquivo: {nome_editado} não está disponível")
                break            
        (audio, _) = librosa.load(nome_editado, sr=32000, mono=True)
        audio = audio[None, :]  # (batch_size, segment_samples)
        #at = AudioTagging(checkpoint_path=None, device='cuda')
        try:
            (clipwise_output, embedding) = at.inference(audio)
            split = int(linha["split"])
            label = gerar_rotulos(linha["nome_linha_estudo"])
            dataset[split]['X'].append(torch.from_numpy(embedding))
            dataset[split]['Y'].append(label)
            dataset[split]['data'].append(linha['data_coleta'])
            dataset[split]['id'].append(linha['id'])
        except RuntimeError as e:
            print(f"Problema na hora de fazer o embedding do audio {nome_editado}, ele será então ignorado.")
    return dataset

def gerar_tensores(lista_label_0, lista_label_1):
    embeddings_train = torch.empty((0, 2048))
    embeddings_test = torch.empty((0, 2048))
    Y = []
    #at = AudioTagging(checkpoint_path=None, device='cuda')
    Y.extend([0]*len(lista_label_0))
    Y.extend([1]*len(lista_label_1))
    listas = lista_label_0 + lista_label_1
    X_train, X_test, Y_train, Y_test = train_test_split(listas, Y, test_size=0.2, stratify=Y)
    for idx, audio_paciente in enumerate(X_train):
        #print(audio_paciente)
        audio_path = audio_paciente
        (audio, _) = librosa.load(audio_path, sr=32000, mono=True)
        audio = audio[None, :]  # (batch_size, segment_samples)
        #at = AudioTagging(checkpoint_path=None, device='cuda')
        try:
            (clipwise_output, embedding) = at.inference(audio)
            embeddings_train = torch.cat([embeddings_train, torch.from_numpy(embedding)], dim=0)
        except RuntimeError as e:
            pos = X_train.index(audio_paciente)
            Y_train.pop(pos)
    for idx, audio_paciente in enumerate(X_test):
        audio_path = audio_paciente
        (audio, _) = librosa.load(audio_path, sr=32000, mono=True)
        audio = audio[None, :]  # (batch_size, segment_samples)
        #at = AudioTagging(checkpoint_path=None, device='cuda')
        try:
            (clipwise_output, embedding) = at.inference(audio)
            embeddings_test = torch.cat([embeddings_test, torch.from_numpy(embedding)], dim=0)
        except RuntimeError as e:
            pos = X_test.index(audio_paciente)
            Y_test.pop(pos)
    return embeddings_train, np.array(Y_train), embeddings_test, np.array(Y_test)

def train_model(model, X_data, Y_data, X_teste, Y_teste, epochs=10):
  acc_treino, acc_teste = [], []
  ks_treino, ks_teste = [], []
  losses = []
  X_tensor = X_data #torch.from_numpy(X_data).float()
  Y_tensor = torch.tensor(Y_data).long().squeeze() # Reshape Y to (N, 1) for MSE loss
  X_teste_tensor = X_teste#torch.from_numpy(X_teste).float()
  Y_teste_tensor = torch.tensor(Y_teste).long().squeeze() # Reshape Y to (N, 1) for MSE loss
  # Define Loss and Optimizer
  criterion = nn.CrossEntropyLoss() # Mean Squared Error Loss
  optimizer = optim.Adam(model.parameters(), lr=0.0001) # Adam optimizer with a learning rate
  optimizer.zero_grad() # Clear gradients
  with torch.no_grad():
    outputs = model(X_tensor)
    acc, k = test_perf_model(outputs, Y_tensor, tipo="classificacao")
    acc_treino.append(acc)
    ks_treino.append(k)
    outputs2 = model(X_teste_tensor)
    acc, k = test_perf_model(outputs2, Y_teste_tensor, tipo="classificacao")
    acc_teste.append(acc)
    ks_teste.append(k)
    losses.append(-1)
  optimizer.zero_grad() # Clear gradients
  #print("Starting model training...")
  for epoch in range(epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, Y_tensor)
    losses.append(loss.item())
    # Backward and optimize
    loss.backward()       # Backpropagation
    optimizer.step()      # Update weights
    with torch.no_grad():
      outputs = model(X_tensor)
      acc, k = test_perf_model(outputs, Y_tensor, tipo="classificacao")
      acc_treino.append(acc)
      ks_treino.append(k)
      outputs2 = model(X_teste_tensor)
      acc, k = test_perf_model(outputs2, Y_teste_tensor, tipo="classificacao")
      acc_teste.append(acc)
      ks_teste.append(k)
    optimizer.zero_grad() # Clear gradients
  #print(f"K de treino: {ks_treino[:20]}")
  #print(f"K de teste: {ks_teste[:20]}")
  return losses, acc_treino, acc_teste, ks_treino, ks_teste, outputs2

def formar_dataset_split(dic, bin):
  print("Estou formando o dataset ", bin)
  X_train, X_test, Y_train, Y_test = [], [], [], []
  for i in range(5):
    if i == bin:
      X_test = dic[i]['X']
      Y_test = dic[i]['Y']
    else:
      X_train += dic[i]['X']
      Y_train += dic[i]['Y']
  print(len(X_train), len(Y_train), len(X_test), len(Y_test))
  X_train = torch.stack(X_train).squeeze()
  X_test = torch.stack(X_test).squeeze()
  return X_train, X_test, Y_train, Y_test

def filtrar_dataset(dic, indices_selecionar):
  dic_filtrado, outro_dic, = criar_dic(), criar_dic()
  for i in range(0, 5):
    X_filtrado, Y_filtrado = [], []
    data_filtrada, id_filtrado = [], []
    data_nao_filtrada, id_nao_filtrado = [], []
    X_nao_filtrado, Y_nao_filtrado = [], []
    for x,y, id, data in zip(dic[i]['X'], dic[i]['Y'], dic[i]['id'], dic[i]['data']):
      if y in indices_selecionar:
        X_filtrado.append(x)
        Y_filtrado.append(y)
        data_filtrada.append(data)
        id_filtrado.append(id)
      else:
        X_nao_filtrado.append(x)
        Y_nao_filtrado.append(y)
        data_nao_filtrada.append(data)
        id_nao_filtrado.append(id)
    dic_filtrado[i]['X'] = X_filtrado
    dic_filtrado[i]['Y'] = Y_filtrado
    dic_filtrado[i]['data'] = data_filtrada
    dic_filtrado[i]['id'] = id_filtrado
    outro_dic[i]['X'] = X_nao_filtrado
    outro_dic[i]['Y'] = Y_nao_filtrado
    outro_dic[i]['data'] = data_nao_filtrada
    outro_dic[i]['id'] = id_nao_filtrado
  return dic_filtrado, outro_dic

def renomear_Y(dic_splits, traducao):
  for split in range(5):
    lista = dic_splits[split]['Y']
    for i in range(len(lista)):
      if lista[i] in traducao:
        lista[i] = traducao[lista[i]]

def gerar_medias(ds):
    preds = []
    for i in range(5):
        X_train, X_test, Y_train, Y_test = formar_dataset_split(ds,i)
        model = NeuralNetwork(2048, 27,2)
        losses, acc_treino, acc_teste, ks_treino, ks_teste, pred = train_model(model, X_train, Y_train, X_test, Y_test, epochs=350)
        print(acc_teste[-1])
        print(ks_teste[-1])
        preds.append(pred)
    return preds
    """
    accs, kappas = [], []
    label_0, label_1 = separar_listas(files, L0, L1)
    for i in range(100):
        if (i+1)%10 == 0:
            print(i)
        model = NeuralNetwork(2048, 27, 2)
        X_treino, Y_treino, X_teste, Y_teste = gerar_tensores(label_0, label_1)
        losses, acc_treino, acc_teste, ks_treino, ks_teste = train_model(model, X_treino, Y_treino, X_teste, Y_teste, epochs=EPOCHS)
        accs.append(acc_teste[-1])
        kappas.append(ks_teste[-1])
    print(f"Média de acc: {np.mean(accs)} e desvio padrão: {np.std(accs)} ")
    print(f"Média de kappa: {np.mean(kappas)} e desvio padrão: {np.std(kappas)} ")
    """


torch.manual_seed(42)
NOME_RUN = "classificacao-teste.csv"
print("Pegando o .csv")
df = get_arquivo_audios()
variavel = "parlenda"
doenca = "tabagism"
dataset = gerar_dataset(df, "coleta_"+variavel)
indices = [0,3]
dataset_atual, dataset_outro = filtrar_dataset(dataset, indices)
renomear_Y(dataset_atual, {indices[0]:0, indices[1]:1})
preds = gerar_medias(dataset_atual)
criar_csv_preds(df, preds, dataset_atual, "preds_"+doenca+"_"+variavel+".csv", [variavel+"_0", variavel+"_1"])
"""
files = get_files("../dados_spira/clean/")

files = filtrar_audios(files, "RHYME.wav")
EPOCHS = 1
"""
#Os labels são IR, PARK, ASMA, CTRL, TABA
"""
model = NeuralNetwork(2048, 27, 2)
if avaliacao == "medias":
    gerar_medias(L0, L1, files)
elif avaliacao == "runs":
    label_0, label_1 = separar_listas(files, L0, L1)
    X_treino, Y_treino, X_teste, Y_teste = gerar_tensores(label_0, label_1)
    print("Fim da geração dos tensores: ", X_treino.shape, len(Y_treino))
    losses, acc_treino, acc_teste, ks_treino, ks_teste = train_model(model, X_treino, Y_treino, X_teste, Y_teste, epochs=EPOCHS)
    gerar_log_saida(NOME_RUN, 
                    [losses, acc_treino, ks_treino, acc_teste, ks_teste],
                    ["loss", "acc_treino", "K_treino", "acc_teste", "K_teste"])
    #print("Ultimas accs: ", acc_teste[-10:])
    #print("Ultimos Kappas: ", ks_teste[-10:])
else:
    print("opção de avaliação inválida")
"""
