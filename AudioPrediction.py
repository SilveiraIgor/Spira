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

def gerar_tensores(lista_label_0, lista_label_1):
    embeddings_train = torch.empty((0, 2048))
    embeddings_test = torch.empty((0, 2048))
    Y = []
    #at = AudioTagging(checkpoint_path=None, device='cuda')
    Y.extend([0]*len(lista_label_0))
    Y.extend([1]*len(lista_label_1))
    listas = lista_label_0 + lista_label_1
    X_train, X_test, Y_train, Y_test = train_test_split(listas, Y, test_size=0.2, stratify=Y)
    for audio_paciente in X_train:
        audio_path = audio_paciente
        (audio, _) = librosa.load(audio_path, sr=32000, mono=True)
        audio = audio[None, :]  # (batch_size, segment_samples)
        #at = AudioTagging(checkpoint_path=None, device='cuda')
        (clipwise_output, embedding) = at.inference(audio)
        embeddings_train = torch.cat([embeddings_train, torch.from_numpy(embedding)], dim=0)
    for audio_paciente in X_test:
        audio_path = audio_paciente
        (audio, _) = librosa.load(audio_path, sr=32000, mono=True)
        audio = audio[None, :]  # (batch_size, segment_samples)
        #at = AudioTagging(checkpoint_path=None, device='cuda')
        (clipwise_output, embedding) = at.inference(audio)
        embeddings_test = torch.cat([embeddings_test, torch.from_numpy(embedding)], dim=0)
    return embeddings_train, np.array(Y_train), embeddings_test, np.array(Y_test)

def train_model(model, X_data, Y_data, X_teste, Y_teste, epochs=10):
  acc_treino, acc_teste = [], []
  ks_treino, ks_teste = [], []
  losses = []
  X_tensor = X_data #torch.from_numpy(X_data).float()
  Y_tensor = torch.from_numpy(Y_data).float().view(-1, 1) # Reshape Y to (N, 1) for MSE loss
  X_teste_tensor = X_teste#torch.from_numpy(X_teste).float()
  Y_teste_tensor = torch.from_numpy(Y_teste).float().view(-1, 1) # Reshape Y to (N, 1) for MSE loss
  # Define Loss and Optimizer
  criterion = nn.MSELoss() # Mean Squared Error Loss
  optimizer = optim.Adam(model.parameters(), lr=0.0001) # Adam optimizer with a learning rate
  optimizer.zero_grad() # Clear gradients
  with torch.no_grad():
    outputs = model(X_tensor)
    acc, k = test_perf_model(outputs, Y_tensor)
    acc_treino.append(acc)
    ks_treino.append(k)
    outputs2 = model(X_teste_tensor)
    acc, k = test_perf_model(outputs2, Y_teste_tensor)
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
      acc, k = test_perf_model(outputs, Y_tensor)
      acc_treino.append(acc)
      ks_treino.append(k)
      outputs2 = model(X_teste_tensor)
      acc, k = test_perf_model(outputs2, Y_teste_tensor)
      acc_teste.append(acc)
      ks_teste.append(k)
    optimizer.zero_grad() # Clear gradients
  #print(f"K de treino: {ks_treino[:20]}")
  #print(f"K de teste: {ks_teste[:20]}")
  return losses, acc_treino, acc_teste, ks_treino, ks_teste


def gerar_medias(L0, L1, files):
    accs, kappas = [], []
    label_0, label_1 = separar_listas(files, L0, L1)
    for i in range(2):
        if (i+1)%10 == 0:
            print(i)
        output_size = 1
        model = NeuralNetwork(2048, 27, output_size)
        X_treino, Y_treino, X_teste, Y_teste = gerar_tensores(label_0, label_1)
        losses, acc_treino, acc_teste, ks_treino, ks_teste = train_model(model, X_treino, Y_treino, X_teste, Y_teste, epochs=EPOCHS)
        accs.append(acc_teste[-1])
        kappas.append(ks_teste[-1])
    print(f"Média de acc: {np.mean(accs)} e desvio padrão: {np.std(accs)} ")
    print(f"Média de kappa: {np.mean(kappas)} e desvio padrão: {np.std(kappas)} ")




NOME_RUN = "Controle-Tabagismo00.csv"
avaliacao = "medias"
L0, L1 = "CTRL", "TABA"
print("Gerando os tensores")
files = get_files("../dados_spira/clean/")
files = filtrar_audios(files, "VOWEL.wav")
EPOCHS = 350
"""
Os labels são IR, PARK, ASMA, CTRL, TABA
"""
model = NeuralNetwork(2048, 27, 1)
if avaliacao == "medias":
    gerar_medias(L0, L1, files)
else:
    label_0, label_1 = separar_listas(files, L0, L1)
    X_treino, Y_treino, X_teste, Y_teste = gerar_tensores(label_0, label_1)
    print("Fim da geração dos tensores: ", X_treino.shape, len(Y_treino))
    losses, acc_treino, acc_teste, ks_treino, ks_teste = train_model(model, X_treino, Y_treino, X_teste, Y_teste, epochs=EPOCHS)
    gerar_log_saida([losses, acc_treino, ks_treino, acc_teste, ks_teste],
                ["loss", "acc_treino", "K_treino", "acc_teste", "K_teste"])
    #print("Ultimas accs: ", acc_teste[-10:])
    #print("Ultimos Kappas: ", ks_teste[-10:])
