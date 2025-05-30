import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
relevante_cb = [
    'E0035',  # total_loans
    'E0038',  # sme_loans
    'I2004', 'KSV12', 'I7000', 'I2513'
]
relevante_størrelser = ['SL30', 'ST10', 'ST20', 'SM20', 'GSIB']
data = data[data['CB_ITEM'].isin(relevante_cb) & data['SBS_BREAKDOWN'].isin(relevante_størrelser)]

# 3. Pivotér data
pivot = data.pivot_table(index=['TIME_PERIOD', 'SBS_BREAKDOWN'], columns='CB_ITEM', values='OBS_VALUE').reset_index()

# 4. Omdøb kolonner
pivot = pivot.rename(columns={
    'E0035': 'total_loans',
    'E0038': 'sme_loans',
    'I2004': 'return_on_assets',
    'KSV12': 'leverage_ratio',
    'I7000': 'npl_ratio',
    'I2513': 'net_interest_income_ratio'
})

# 5. Beregn andel af SMV-lån
pivot['andel_smv'] = pivot['sme_loans'] / pivot['total_loans']

# 6. Fjern rækker med manglende værdier
pivot = pivot.dropna(subset=['return_on_assets', 'leverage_ratio', 'npl_ratio', 'net_interest_income_ratio'])

# 7. Tilføj labels og korrekt rækkefølge
labels = {
    'SL30': 'Små banker',
    'ST10': 'Mellemstore banker',
    'ST20': 'Store banker',
    'SM20': 'Meget store banker',
    'GSIB': 'Mega banker'
}
pivot['bank_stoerrelse'] = pivot['SBS_BREAKDOWN'].map(labels)

bank_rækkefølge = ['Små banker', 'Mellemstore banker', 'Store banker', 'Meget store banker', 'Mega banker']
pivot['bank_stoerrelse'] = pd.Categorical(pivot['bank_stoerrelse'], categories=bank_rækkefølge, ordered=True)


# 8. Definér faste farver pr. bankstørrelse (kraftige farver)
farver = {
    'Små banker': '#1f77b4',          # blå
    'Mellemstore banker': '#ff7f0e',  # orange
    'Store banker': '#2ca02c',        # grøn
    'Meget store banker': '#d62728',  # rød
    'Mega banker': '#9467bd'          # lilla
}

# 9. Variabler og titler
variabler = ['return_on_assets', 'net_interest_income_ratio', 'npl_ratio', 'leverage_ratio']
titler = [
    'Return on Assets',
    'Net Interest Income Ratio',
    'Non-Performing Loans Ratio',
    'Leverage Ratio'
]

# 10. Lav 2x2 subplot grid med mindre x-labels (bankstørrelse)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# Fontsize-indstillinger
title_size = 14
label_size = 12
tick_size_x = 10  # 👈 mindre skriftstørrelse på bankstørrelser
tick_size_y = 10

for i, var in enumerate(variabler):
    sns.boxplot(
        data=pivot,
        x='bank_stoerrelse',
        y=var,
        palette=farver,
        ax=axes[i]
    )
    axes[i].set_title(titler[i], fontsize=title_size)
    axes[i].set_xlabel('', fontsize=label_size)
    axes[i].set_ylabel('', fontsize=label_size)
    axes[i].tick_params(axis='x', labelsize=tick_size_x, rotation=15)
    axes[i].tick_params(axis='y', labelsize=tick_size_y)

    # 👇 Betinget formattering af y-aksen
    if var == 'return_on_assets':
        axes[i].yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.2f}".replace('.', ','))
        )
    else:
        axes[i].yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x)}")
        )

plt.tight_layout()
fig.savefig("Kontrolvariable boxplot.pdf", bbox_inches='tight')
plt.show()