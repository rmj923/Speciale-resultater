import pandas as pd
import matplotlib.pyplot as plt


# Konfigurer matplotlib til at bruge serif og Computer Modern
plt.rcParams.update({
    "text.usetex": False,  # Ingen ekstern LaTeX
    "font.family": "serif",
    "font.serif": ["Computer Modern"],  # Brug LaTeX-lignende font
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 14,
})

# 1. Indlæs data
file_path = '/Users/emilerbas/Desktop/SUP_renset.csv'
data = pd.read_csv(file_path)

# 2. Filtrér relevante CB_ITEM og bankstørrelser
relevante_cb = ['E0035', 'E0038']
relevante_størrelser = ['SL30', 'ST10', 'ST20', 'SM20', 'GSIB']
data = data[data['CB_ITEM'].isin(relevante_cb) & data['SBS_BREAKDOWN'].isin(relevante_størrelser)]

# 3. Pivotér data
pivot = data.pivot_table(index=['TIME_PERIOD', 'SBS_BREAKDOWN'], columns='CB_ITEM', values='OBS_VALUE').reset_index()
pivot = pivot.rename(columns={'E0035': 'total_loans', 'E0038': 'sme_loans'})

# 4. Beregn andel af SMV-lån
pivot['andel_smv'] = pivot['sme_loans'] / pivot['total_loans']
pivot = pivot.dropna(subset=['andel_smv'])

# 5. Mapping til labels
labels = {
    'SL30': 'Små banker',
    'ST10': 'Mellemstore banker',
    'ST20': 'Store banker',
    'SM20': 'Meget store banker',
    'GSIB': 'Mega banker'
}
pivot['bank_stoerrelse'] = pivot['SBS_BREAKDOWN'].map(labels)

# 6. Konverter tid og tilføj kvartalslabel
pivot['TIME_PERIOD'] = pd.to_datetime(pivot['TIME_PERIOD'])
pivot['kvartal'] = pivot['TIME_PERIOD'].dt.to_period('Q').astype(str)
pivot['kvartal'] = pd.Categorical(pivot['kvartal'], ordered=True, categories=sorted(pivot['kvartal'].unique()))

# 7. Bankgrupper og rækkefølge
rækkefoelge = ['Små banker', 'Mellemstore banker', 'Store banker', 'Meget store banker', 'Mega banker']

# 8. Definér farver pr. bankstørrelse
farver = {
    'Små banker': '#1f77b4',
    'Mellemstore banker': '#ff7f0e',
    'Store banker': '#2ca02c',
    'Meget store banker': '#d62728',
    'Mega banker': '#9467bd'
}

# 9. Linjestile
styles = {
    'Små banker': {'color': farver['Små banker'], 'linestyle': '-.'},
    'Mellemstore banker': {'color': farver['Mellemstore banker'], 'linestyle': '--'},
    'Store banker': {'color': farver['Store banker'], 'linestyle': ':'},
    'Meget store banker': {'color': farver['Meget store banker'], 'linestyle': '-.'},
    'Mega banker': {'color': farver['Mega banker'], 'linestyle': '-'},
}

plt.figure(figsize=(12, 8))

for bank in rækkefoelge:
    df = pivot[pivot['bank_stoerrelse'] == bank]
    if df.empty:
        print(f"[ADVARSEL] Ingen data for {bank}")
        continue
    plt.plot(
        df['kvartal'].astype(str),
        df['andel_smv'],
        label=bank,
        color=styles[bank]['color'],
        linestyle=styles[bank]['linestyle']
    )

# 11. Akser og layout
plt.ylabel('Andel SMV-lån', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Bankstørrelser', loc='best', fontsize=12, title_fontsize=12)
plt.tight_layout()
plt.savefig("smv_andel_linjediagram2.pdf", bbox_inches='tight')
plt.show()
