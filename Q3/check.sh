# Find a labelled clip
python3 -c "
import pandas as pd, os
df = pd.read_csv('data/sps-corpus-3.0-2026-03-09-en/ss-corpus-en.tsv', sep='\t')
row = df.dropna(subset=['gender','age']).iloc[0]
print('File:', row['audio_file'])
print('Gender:', row['gender'])
print('Age:', row['age'])
"