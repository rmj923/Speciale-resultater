import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import seaborn as sns
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
file_path = '/Users/emilerbas/Desktop/data.2.csv'
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
    'I2004': 'ROA',
    'I7000': 'NPL_ratio',
    'I2513': 'NII_ratio',
    'KSV12': 'leverage_ratio'
})

# 4. Beregn andel af SMV-lån
pivot['andel_smv'] = pivot['sme_loans'] / pivot['total_loans']

# 5. Fjern rækker med manglende værdier
pivot = pivot.dropna(subset=['andel_smv', 'ROA', 'NPL_ratio', 'NII_ratio', 'leverage_ratio'])

# 6. Tilføj bankstørrelseslabels og tidstrend
labels = {'SL30': 'Små banker', 'ST10': 'Mellemstore banker', 'ST20': 'Store banker', 'SM20': 'Meget store banker', 'GSIB': 'Mega banker'}
pivot['bank_stoerrelse'] = pivot['SBS_BREAKDOWN'].map(labels)
pivot['time_trend'] = np.arange(len(pivot))

# 7. Estimer Model 3 med robuste standardfejl og kontrolvariable
model_controls = smf.ols(
    'andel_smv ~ C(bank_stoerrelse, Treatment(reference="Små banker")) + time_trend + ROA + NPL_ratio + NII_ratio + leverage_ratio',
    data=pivot
).fit(cov_type='HC1')  # HC1 = robuste (Huber-White) standardfejl

# 8. Print resultater
print(model_controls.summary())



# 9. Udvælg kontrolvariabler
kontroller = pivot[['ROA', 'NPL_ratio', 'NII_ratio', 'leverage_ratio']]

# 10. Beregn korrelationsmatrix
correlation_matrix = kontroller.corr()

# 11. Omdøb labels til pænere navne
navne = {
    'ROA': 'ROA',
    'NPL_ratio': 'NPL-ratio',
    'NII_ratio': 'NII-ratio',
    'leverage_ratio': 'Leverage ratio'
}
correlation_matrix = correlation_matrix.rename(index=navne, columns=navne)



plt.figure(figsize=(12, 8))
sns.set(style="white", font_scale=1.2)
heat = sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=.5,
    cbar_kws={"shrink": .75}
)

# Flyt labels til toppen og venstre
heat.xaxis.set_ticks_position('top')
heat.xaxis.set_label_position('top')
heat.yaxis.set_tick_params(labelright=False, labelleft=True)

# Fjern aksetitler
heat.set_xlabel('')
heat.set_ylabel('')

# Roter og placer labels korrekt
heat.set_xticklabels(heat.get_xticklabels(), fontsize=14, rotation=0)
heat.set_yticklabels(heat.get_yticklabels(), fontsize=14, rotation=0)

# ✨ Fjern tick marks (små streger)
heat.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)

# Titel og layout
plt.tight_layout()
plt.savefig("korrelationsmatrix_kontrolvariable.pdf", dpi=300)
plt.show()
