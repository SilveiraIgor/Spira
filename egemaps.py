import opensmile
from utils import *



smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
files = get_files("../dados_spira/clean/")
files = filtrar_audios(files, "PHRASE.wav")
df = pd.DataFrame()
for f in files:
    y = smile.process_file(f)
    df = pd.concat([df, y], ignore_index=True)
print(df.shape)
df.to_csv("egemaps-frase.csv", index=False)
