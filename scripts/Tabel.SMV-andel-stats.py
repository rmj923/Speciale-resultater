import pandas as pd
import os

# 1. Indlæs data
file_path = '/Users/emilerbas/Desktop/SUP_renset.csv'
data = pd.read_csv(file_path)

# 2. Filtrér relevante CB_ITEM og bankstørrelser
relevante_cb = ['E0035', 'E0038']
relevante_størrelser = ['SL30', 'ST10', 'ST20', 'SM20', 'GSIB']
data = data[data['CB_ITEM'].isin(relevante_cb) & data['SBS_BREAKDOWN'].isin(relevante_størrelser)]

# 3. Pivotér data så vi har total og SME-lån i kolonner
pivot = data.pivot_table(index=['TIME_PERIOD', 'SBS_BREAKDOWN'], columns='CB_ITEM', values='OBS_VALUE').reset_index()
pivot = pivot.rename(columns={'E0035': 'total_loans', 'E0038': 'sme_loans'})

# 4. Beregn andel af SMV-lån
pivot['andel_smv'] = pivot['sme_loans'] / pivot['total_loans']
pivot = pivot.dropna(subset=['andel_smv'])

# 5. Mapping til læsbare labels
labels = {
    'SL30': 'Små banker',
    'ST10': 'Mellemstore banker',
    'ST20': 'Store banker',
    'SM20': 'Meget store banker',
    'GSIB': 'Mega banker'
}
pivot['bank_stoerrelse'] = pivot['SBS_BREAKDOWN'].map(labels)

# 6. Beregn statistik for hver bankstørrelse
stats = pivot.groupby('bank_stoerrelse')['andel_smv'].agg([
    ('Gennemsnit', 'mean'),
    ('Median', 'median'),
    ('Standardafvigelse', 'std'),
    ('Minimum', 'min'),
    ('Maksimum', 'max')
]).reset_index()

# 7. Afrund og sorter rækkefølge manuelt
stats = stats.round(3)
rækkefølge = ['Små banker', 'Mellemstore banker', 'Store banker', 'Meget store banker', 'Mega banker']
stats['bank_stoerrelse'] = pd.Categorical(stats['bank_stoerrelse'], categories=rækkefølge, ordered=True)
stats = stats.sort_values('bank_stoerrelse')

# 8. Vis tabellen i output
print("\nStatistik over SMV-andel for hver bankstørrelse (sorteret):\n")
print(stats)



# 10. Gem som TXT-fil med formateret tabel
output_txt_path = os.path.join(output_folder, 'smv_andel_statistik.txt')
with open(output_txt_path, 'w', encoding='utf-8') as f:
    f.write(stats.to_string(index=False))

print(f"\nStatistikken er også gemt som: {output_txt_path}")
