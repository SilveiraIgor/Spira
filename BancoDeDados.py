import pandas as pd
import math
import re
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from scipy.stats import binomtest
import numpy as np
from utils import *


def audios_em_df(df, audios):
    coluna_parlenda = df.coleta_parlenda
    coluna_vogal = df.coleta_vogal
    coluna_frase = df.coleta_frase
    print(coluna_vogal[:5])
    problemas_vogal, problemas_parlenda, problemas_frase = 0,0,0
    for nome_audio in audios:
        audio = nome_audio[21:]
        audio = audio.replace("__F_", "%_F_")
        audio = audio.replace("__M_", "%_M_")
        #print(audio)
        #break
        if audio.endswith("VOWEL.wav"):
            if not(audio in coluna_vogal.values):
                print(audio)
                problemas_vogal += 1
        elif audio.endswith("RHYME.wav"):
            if not(audio in coluna_parlenda.values):
                problemas_parlenda += 1
        elif audio.endswith("PHRASE.wav"):
            if not(audio in coluna_frase.values):
                problemas_frase += 1
        else:
            pass
    print(f"""Número de problemas: {problemas_vogal} de vogal, 
          {problemas_parlenda} de parlenda, 
          {problemas_frase} de frase""")


def df_em_audios(df, audios, coluna):
    print(f">>> Fazendo os testes para {coluna}")
    problemas_nan = 0
    problemas_ausencia = 0
    for idx, row in df.iterrows():
        nome_audio_puro = row[coluna]
        if pd.isna(nome_audio_puro):
            print(f"Paciente {row['identificador_paciente']} não tem áudio em {coluna}")
            problemas_nan += 1
        else:
            #print(nome_audio_puro)
            nome_audio = "../dados_spira/clean/"+ nome_audio_puro.replace("%","_")
            if nome_audio not in audios:
                print(f"-- Problema com o audio {idx}: {nome_audio}")
                problemas_ausencia += 1
    print(f">>> O numero total de problemas foi {problemas_nan} de nans e {problemas_ausencia} de ausencia")

def filtrar_testes(df):
  # Preenche valores NaN na coluna 'local_coleta' com string vazia para evitar erros
  # e então filtra as linhas que contêm 'TESTE' (case-insensitive)
  df_filtrado = df[~df['local_coleta'].fillna('').str.contains('TESTE', case=False, na=False)]
  df_filtrado = df_filtrado[~df_filtrado['local_coleta'].fillna('').str.contains('TESTJAQUELINE', case=False, na=False)]
  df_filtrado = df_filtrado[~df_filtrado['local_coleta'].fillna('').str.contains('MARCELO', case=False, na=False)]
  df_filtrado = df_filtrado[~df_filtrado['local_coleta'].fillna('').str.contains('TEST', case=False, na=False)]
  #df_filtrado = df_filtrado[~df_filtrado['local_coleta'].fillna('').str.contains('TESTE', case=False, na=False)]
  df_filtrado = df_filtrado[~df_filtrado['local_coleta'].fillna('').str.contains('HOME', case=False, na=False)]
  return df_filtrado


testar = "dataset"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("../dados_spira/clean/Metadados/ColetaSPIRA_RECENTE-clean.csv")
print(f"Tamanho do dataset antes de filtrar: {df.shape}")
df_filtrado = filtrar_testes(df)
print(f"Tamanho do dataset depois de filtrar: {df_filtrado.shape}")
audios = get_files("../dados_spira/clean/")
audios = filtrar_audios(audios, ".wav")
print("Quantidade de audios: ", len(audios))
if testar == "dataset":
    df_em_audios(df_filtrado, audios, 'coleta_vogal')
    df_em_audios(df_filtrado, audios, 'coleta_parlenda')
    df_em_audios(df_filtrado, audios, 'coleta_frase')
elif testar == "audios":
    audios_em_df(df, audios)
