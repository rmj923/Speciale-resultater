import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 0. Typografi: Brug LaTeX + Computer Modern
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

# 8. Definér farver pr. bankstørrelse
farver = {
    'Små banker': '#1f77b4',
    'Mellemstore banker': '#ff7f0e',
    'Store banker': '#2ca02c',
    'Meget store banker': '#d62728',
    'Mega banker': '#9467bd'
}

# 9. Variabler og titler
variabler = ['return_on_assets', 'net_interest_income_ratio', 'npl_ratio', 'leverage_ratio']
titler = [
    'Return on Assets',
    'Net Interest Income Ratio',
    'Non-Performing Loans Ratio',
    'Leverage Ratio'
]

# 10. Scatterplot layout og størrelser
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, var in enumerate(variabler):
    sns.scatterplot(
        data=pivot,
        x=var,
        y='andel_smv',
        hue='bank_stoerrelse',
        palette=farver,
        ax=axes[i],
        alpha=0.7,
        edgecolor='w',
        s=60
    )
    axes[i].set_title(titler[i], fontsize=14)
    axes[i].set_xlabel('')
    axes[i].set_ylabel(r"Andel SMV-lån", fontsize=14)
    axes[i].tick_params(axis='x', labelsize=12)
    axes[i].tick_params(axis='y', labelsize=12)

# 11. Samlet legend
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels_,
    loc='lower center',
    ncol=5,
    frameon=False,
    fontsize=13,
    bbox_to_anchor=(0.5, -0.02)
)

# 12. Fjern individuelle legends
for ax in axes:
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

# 13. Layout og gem
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("kontrolvariable_scatterplot.pdf", bbox_inches='tight')
plt.show()
