import opensmile
from utils import *



smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
for tipo_audio in ["VOWEL", "RHYME", "PHRASE"]:
    files = get_files("../dados_spira/clean/")
    files = filtrar_audios(files, tipo_audio+".wav")
    df = pd.DataFrame()
    for f in files:
        y = smile.process_file(f)
        y['audio_'+tipo_audio] = f
        df = pd.concat([df, y], ignore_index=True)
    print(df.shape)
    df.to_csv("egemaps-"+tipo_audio+".csv", index=False)
