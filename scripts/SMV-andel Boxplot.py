import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
relevante_cb = ['E0035', 'E0038']  # Total loans og SME loans
relevante_størrelser = ['SL30', 'ST10', 'ST20', 'SM20', 'GSIB']
data = data[data['CB_ITEM'].isin(relevante_cb) & data['SBS_BREAKDOWN'].isin(relevante_størrelser)]

# 3. Pivotér data
pivot = data.pivot_table(index=['TIME_PERIOD', 'SBS_BREAKDOWN'], columns='CB_ITEM', values='OBS_VALUE').reset_index()
pivot = pivot.rename(columns={'E0035': 'total_loans', 'E0038': 'sme_loans'})

# 4. Beregn andel af SMV-lån og fjern NaN
pivot['andel_smv'] = pivot['sme_loans'] / pivot['total_loans']
pivot = pivot.dropna(subset=['andel_smv'])

# 5. Tilføj bankstørrelseslabels
labels = {
    'SL30': 'Små banker',
    'ST10': 'Mellemstore banker',
    'ST20': 'Store banker',
    'SM20': 'Meget store banker',
    'GSIB': 'Mega banker'
}
pivot['bank_stoerrelse'] = pivot['SBS_BREAKDOWN'].map(labels)

# 6. Definér rækkefølge manuelt
rækkefoelge = ['Små banker', 'Mellemstore banker', 'Store banker', 'Meget store banker', 'Mega banker']

# 7. Definér farvepalette
farver = {
    'Små banker': '#1f77b4',          # blå
    'Mellemstore banker': '#ff7f0e',  # orange
    'Store banker': '#2ca02c',        # grøn
    'Meget store banker': '#d62728',  # rød
    'Mega banker': '#9467bd'          # lilla
}

# 8. Plot
plt.figure(figsize=(12, 8))

sns.boxplot(
    data=pivot,
    x='bank_stoerrelse',
    y='andel_smv',
    palette=farver,
    order=rækkefoelge
)

plt.title('', fontsize=14)
plt.xlabel('')
plt.ylabel('Andel SMV-lån', fontsize=12)
plt.xticks(fontsize=14, rotation='')
plt.yticks(fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Andel__boxplot.pdf", bbox_inches='tight')
plt.show()
