import torch
import os
import numpy as np
from pathlib import Path
import librosa
import panns_inference
from panns_inference import AudioTagging
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score

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

def gerar_tensores(lista_label_0, lista_label_1):
    embeddings_train = torch.empty((0, 2048))
    embeddings_test = torch.empty((0, 2048))
    Y = []
    at = AudioTagging(checkpoint_path=None, device='cuda')
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

# Define the Neural Network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.relu = nn.ReLU()                          # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size) # Hidden layer to output layer
        self.sigmoid = nn.Sigmoid()                    # Sigmoid for binary classification

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


def test_acc_model(Y_hat, Y):
  discretized_outputs = (Y_hat >= 0.5).float()
  # Compare with the Y_tensor and count where they are equal
  matches = (discretized_outputs == Y).sum().item()
  total_elements = Y.numel()
  #print(f"Número de correspondências: {matches}")
  #print(f"Número total de elementos: {total_elements}")
  acc = matches / total_elements * 100
  #print(f"Porcentagem de correspondências (Accuracy): {acc:.2f}%")
  kappa = cohen_kappa_score(discretized_outputs, Y)
  return acc, kappa

def train_model(model, X_data, Y_data, X_teste, Y_teste, epochs=10):
  # Convert numpy arrays to PyTorch tensors
  acc_treino, acc_teste = [], []
  ks_treino, ks_teste = [], []
  X_tensor = X_data #torch.from_numpy(X_data).float()
  Y_tensor = torch.from_numpy(Y_data).float().view(-1, 1) # Reshape Y to (N, 1) for MSE loss
  X_teste_tensor = X_teste#torch.from_numpy(X_teste).float()
  Y_teste_tensor = torch.from_numpy(Y_teste).float().view(-1, 1) # Reshape Y to (N, 1) for MSE loss
  # Define Loss and Optimizer
  criterion = nn.MSELoss() # Mean Squared Error Loss
  optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimizer with a learning rate
  optimizer.zero_grad() # Clear gradients
  with torch.no_grad():
    outputs = model(X_tensor)
    acc, k = test_acc_model(outputs, Y_tensor)
    acc_treino.append(acc)
    ks_treino.append(k)
    outputs2 = model(X_teste_tensor)
    acc, k = test_acc_model(outputs2, Y_teste_tensor)
    acc_teste.append(acc)
    ks_teste.append(k)
  optimizer.zero_grad() # Clear gradients
  #print("Starting model training...")
  for epoch in range(epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, Y_tensor)
    # Backward and optimize
    loss.backward()       # Backpropagation
    optimizer.step()      # Update weights
    with torch.no_grad():
      outputs = model(X_tensor)
      acc, k = test_acc_model(outputs, Y_tensor)
      acc_treino.append(acc)
      ks_treino.append(k)
      outputs2 = model(X_teste_tensor)
      acc, k = test_acc_model(outputs2, Y_teste_tensor)
      acc_teste.append(acc)
      ks_teste.append(k)
    optimizer.zero_grad() # Clear gradients
    # if (epoch + 1) % 10 == 0: # Print loss every epoch
    #   print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
  #print("Training complete!")
  print(f"K de treino: {ks_treino}")
  print(f"K de teste: {ks_teste}")
  return acc_treino, acc_teste




print("Gerando os tensores")
files = get_files("../dados_spira/clean/")
files = filtrar_audios(files, "VOWEL.wav")
for f in files[:10]:
    print(f)
"""
Os labels são IR, PARK, ASMA, CTRL, TABA
"""
label_0, label_1 = separar_listas(files, "CTRL", "ASMA")
X_treino, Y_treino, X_teste, Y_teste = gerar_tensores(label_0, label_1)
print("Fim da geração dos tensores: ", X_treino.shape, len(Y_treino))

output_size = 1
model = NeuralNetwork(2048, 27, output_size)
treino, teste = train_model(model, X_treino, Y_treino, X_teste, Y_teste, epochs=10)
print(f"Performance no treino: {treino}")
print(f"Performance no teste: {teste}")
