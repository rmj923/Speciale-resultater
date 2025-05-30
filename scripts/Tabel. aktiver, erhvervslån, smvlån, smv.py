# 1. Importer pandas
import pandas as pd

# 2. Indlæs datasættet
file_path = '/Users/emilerbas/Desktop/SUP_renset.csv'
data = pd.read_csv(file_path)

# 3. Filtrér relevante CB_ITEMs
relevante_cb = ['A0000', 'E0035', 'E0038']
filtered = data[data['CB_ITEM'].isin(relevante_cb)]

# 4. Filtrér kun ønskede bankgrupper
ønskede_grupper = ['SL30', 'ST10', 'ST20', 'SM20', 'GSIB']
filtered = filtered[filtered['SBS_BREAKDOWN'].isin(ønskede_grupper)]

# 5. Beregn gennemsnit pr. gruppe og CB_ITEM
gennemsnit = filtered.groupby(['SBS_BREAKDOWN', 'CB_ITEM'])['OBS_VALUE'].mean().reset_index()

# 6. Pivotér så CB_ITEMs bliver kolonner
pivot = gennemsnit.pivot(index='SBS_BREAKDOWN', columns='CB_ITEM', values='OBS_VALUE').reset_index()

# 7. Omdøb kolonner for klarhed
pivot = pivot.rename(columns={
    'A0000': 'Gns. totale aktiver',
    'E0035': 'Gns. erhvervslån',
    'E0038': 'Gns. SMV-lån'
})

# 8. Beregn andel SMV-lån
pivot['Andel SMV-lån'] = pivot['Gns. SMV-lån'] / pivot['Gns. erhvervslån']

# 9. Definér labels til SBS_BREAKDOWN
labels = {
    'SL30': 'Små banker',
    'ST10': 'Mellemstore banker',
    'ST20': 'Store banker',
    'SM20': 'Meget store banker',
    'GSIB': 'Mega banker'
}

# 10. Tilføj kolonne med labels
pivot['Bankstørrelse'] = pivot['SBS_BREAKDOWN'].map(labels)

# 11. Sortér efter ønsket rækkefølge
pivot['Bankstørrelse'] = pd.Categorical(
    pivot['Bankstørrelse'],
    categories=['Små banker', 'Mellemstore banker', 'Store banker', 'Meget store banker', 'Mega banker'],
    ordered=True
)

pivot = pivot.sort_values('Bankstørrelse')

# 12. Rund værdier til 3 decimaler
pivot = pivot.round(3)

# 13. Vælg og omarrangér kolonner til slutoutput
kolonner = ['Bankstørrelse', 'Gns. totale aktiver', 'Gns. erhvervslån', 'Gns. SMV-lån', 'Andel SMV-lån']
output = pivot[kolonner]

# 14. Udskriv som tabel
print("\nTabel 1.2 – Gennemsnit for bankaktiver, erhvervslån og SMV-lån pr. bankstørrelsesgruppe:\n")
print(output.to_string(index=False))

# 15. Gem som txt-fil på skrivebordet
output_path = '/Users/emilerbas/Desktop/Tabeller/Tabel_1_2_output.txt'
with open(output_path, 'w') as f:
    f.write("Tabel 1.2 – Gennemsnit for bankaktiver, erhvervslån og SMV-lån pr. bankstørrelsesgruppe\n\n")
    f.write(output.to_string(index=False))