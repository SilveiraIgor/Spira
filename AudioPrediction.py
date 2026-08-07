import torch
import os
from pathlib import Path

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


files = get_files("../dados_spira/clean/")
files = filtrar_audios(files, "VOWEL.wav")
for f in files[:10]:
    print(f)

