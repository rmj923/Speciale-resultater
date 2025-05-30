import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
import warnings


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
relevante_cb = ['E0035', 'E0038', 'I2004', 'I7000', 'I2513', 'KSV12']
relevante_størrelser = ['SL30', 'ST10', 'ST20', 'SM20', 'GSIB']
data = data[data['CB_ITEM'].isin(relevante_cb) & data['SBS_BREAKDOWN'].isin(relevante_størrelser)]

# 3. Pivotér data til brede form
pivot = data.pivot_table(index=['TIME_PERIOD', 'SBS_BREAKDOWN'], columns='CB_ITEM', values='OBS_VALUE').reset_index()
pivot = pivot.rename(columns={
    'E0035': 'total_loans',
    'E0038': 'sme_loans',
    'I2004': 'return_on_assets',
    'I7000': 'npl_ratio',
    'I2513': 'net_interest_income_ratio',
    'KSV12': 'leverage_ratio'
})

# 4. Beregn andel af SMV-lån
pivot['andel_smv'] = pivot['sme_loans'] / pivot['total_loans']

# 5. Fjern rækker med manglende værdier
pivot = pivot.dropna(subset=['andel_smv', 'return_on_assets', 'npl_ratio', 'net_interest_income_ratio', 'leverage_ratio'])

# 6. Tilføj bankstørrelseslabels og tidstrend
labels = {'SL30': 'Små banker', 'ST10': 'Mellemstore banker', 'ST20': 'Store banker', 'SM20': 'Meget store banker', 'GSIB': 'Mega banker'}
pivot['bank_stoerrelse'] = pivot['SBS_BREAKDOWN'].map(labels)
pivot['time_trend'] = np.arange(len(pivot))

# 7. Estimer Model 3 med robuste standardfejl og kontrolvariable
model_controls = smf.ols(
    'andel_smv ~ C(bank_stoerrelse, Treatment(reference="Små banker")) + time_trend + return_on_assets + npl_ratio + net_interest_income_ratio + leverage_ratio',
    data=pivot
).fit(cov_type='HC1')

# Tilføj residualer til datasættet
pivot['residuals'] = model_controls.resid

# 8. Print resultater
print(model_controls.summary())

# (resten af din plot-kode følger her)


farver = {
    'Små banker': '#1f77b4',          # blå
    'Mellemstore banker': '#ff7f0e',  # orange
    'Store banker': '#2ca02c',        # grøn
    'Meget store banker': '#d62728',  # rød
    'Mega banker': '#9467bd'          # lilla
}
# Sørg for at bank_stoerrelse er kategorisk og sorteret i den ønskede rækkefølge
kategori_ordning = ['Små banker', 'Mellemstore banker', 'Store banker', 'Meget store banker', 'Mega banker']
pivot['bank_stoerrelse'] = pd.Categorical(pivot['bank_stoerrelse'], categories=kategori_ordning, ordered=True)

# Opret farveliste i samme rækkefølge
palette = [farver[kat] for kat in kategori_ordning]

# Plot boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(data=pivot, x='bank_stoerrelse', y='residuals', palette=palette)
plt.axhline(0, color='black', linestyle='--', linewidth=1.2)
plt.xlabel('')  # Fjern "bank_stoerrelse"-label
plt.ylabel('Residualer', fontsize=14)
plt.xticks(fontsize=14)
plt.tight_layout()
plt.savefig("Residualer.pdf", bbox_inches='tight')
plt.show()