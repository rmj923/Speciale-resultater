import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
import warnings

# Undgå warnings for små samples eller ustabilitet
warnings.filterwarnings("ignore")

# 1. Indlæs data
file_path = '/Users/emilerbas/Desktop/data.2.csv'
data = pd.read_csv(file_path)

# 2. Filtrér relevante CB_ITEM og bankstørrelser
relevante_cb = ['E0035', 'E0038', 'I7000', 'I2513', 'KSV12']
relevante_størrelser = ['SL30', 'ST10', 'ST20', 'SM20', 'GSIB']
data = data[data['CB_ITEM'].isin(relevante_cb) & data['SBS_BREAKDOWN'].isin(relevante_størrelser)]

# 3. Pivotér data
pivot = data.pivot_table(index=['TIME_PERIOD', 'SBS_BREAKDOWN'], columns='CB_ITEM', values='OBS_VALUE').reset_index()
pivot = pivot.rename(columns={
    'E0035': 'total_loans',
    'E0038': 'sme_loans',
    'I7000': 'npl_ratio',
    'I2513': 'net_interest_income_ratio',
    'KSV12': 'leverage_ratio'
})

# 4. Beregn andel af SMV-lån
pivot['andel_smv'] = pivot['sme_loans'] / pivot['total_loans']
pivot = pivot.dropna(subset=['andel_smv', 'npl_ratio', 'net_interest_income_ratio', 'leverage_ratio'])

# 5. Banklabels og tidstrend
labels = {
    'SL30': 'Små banker',
    'ST10': 'Mellemstore banker',
    'ST20': 'Store banker',
    'SM20': 'Meget store banker',
    'GSIB': 'Mega banker'
}
pivot['bank_stoerrelse'] = pivot['SBS_BREAKDOWN'].map(labels)
pivot['time_trend'] = np.arange(len(pivot))
pivot['time_trend'] = pivot['time_trend'] / pivot['time_trend'].max()  # Skaler til [0,1]

# 6. Standardisér kontrolvariable
scaler = StandardScaler()
kontrolvars = ['npl_ratio', 'net_interest_income_ratio', 'leverage_ratio']
pivot_std = pivot.copy()
pivot_std[kontrolvars] = scaler.fit_transform(pivot[kontrolvars])

# 7. OLS-model uden ROA med robuste standardfejl
formula = 'andel_smv ~ C(bank_stoerrelse, Treatment(reference="Små banker")) + time_trend + npl_ratio + net_interest_income_ratio + leverage_ratio'
model3_std = smf.ols(formula=formula, data=pivot_std).fit(cov_type='HC1')
print(model3_std.summary())




# 8. Gem residualer fra modellen
pivot_std['residuals'] = model3_std.resid

# 9. Boxplot: residualer fordelt på bankstørrelse (potentielt endogen)
plt.figure(figsize=(10, 6))
sns.boxplot(data=pivot_std, x='bank_stoerrelse', y='residuals', palette='pastel')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Bankstørrelse', fontsize=12)
plt.ylabel('Residualer fra OLS-model', fontsize=12)
plt.xticks(fontsize=12)  # Gør x-ticks større
plt.tight_layout()
plt.savefig("Residualer.pdf", bbox_inches='tight')
plt.show()


# 10. Valgfrit: Opsummer residualstatistik pr. bankstørrelse
print("\nBeskrivende statistik for residualer pr. bankstørrelse:")
print(pivot_std.groupby('bank_stoerrelse')['residuals'].describe())