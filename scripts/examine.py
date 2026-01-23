import pandas as pd

# Cargar el fichero TSV
file_path = "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/ab_ar.train.tsv"
df = pd.read_csv(file_path, sep='\t')

# Número de speakers únicos
num_speakers = df['client_id'].nunique()
print(f"Número de speakers distintos: {num_speakers}")

# Número de muestras por speaker
samples_per_speaker = df.groupby('client_id').size()
print("Número de muestras por speaker:")
print(samples_per_speaker)

# Cargar el fichero TSV
file_path = "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/ab_cs.train.tsv"
df = pd.read_csv(file_path, sep='\t')

# Número de speakers únicos
num_speakers = df['client_id'].nunique()
print(f"Número de speakers distintos: {num_speakers}")

# Número de muestras por speaker
samples_per_speaker = df.groupby('client_id').size()
print("Número de muestras por speaker:")
print(samples_per_speaker)
