import torch
import os
from pathlib import Path
import librosa
import panns_inference
from panns_inference import AudioTagging
import torch
import torch.nn as nn
import torch.optim as optim

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
    embeddings = torch.empty((0, 2048))
    Y = []
    at = AudioTagging(checkpoint_path=None, device='cuda')
    listas = lista_label_0 + lista_label_1
    for audio_paciente in listas:
        audio_path = audio_paciente
        (audio, _) = librosa.load(audio_path, sr=32000, mono=True)
        audio = audio[None, :]  # (batch_size, segment_samples)
        #at = AudioTagging(checkpoint_path=None, device='cuda')
        (clipwise_output, embedding) = at.inference(audio)
        embeddings = torch.cat([embeddings, torch.from_numpy(embedding)], dim=0)
    Y.extend([0]*len(lista_label_0))
    Y.extend([1]*len(lista_label_1))
    return embeddings, Y



files = get_files("../dados_spira/clean/")
files = filtrar_audios(files, "VOWEL.wav")
for f in files[:10]:
    print(f)
label_0, label_1 = separar_listas(files, "IR", "PARK")
tensores, Y = gerar_tensores(label_0, label_1)
print(tensores.shape, len(Y))

