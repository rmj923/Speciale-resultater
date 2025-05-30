from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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



# 1. Definer din model formel (samme som i Model 3)
formula = 'andel_smv ~ C(bank_stoerrelse, Treatment(reference="Små banker")) + time_trend + ROA + NPL_ratio + NII_ratio + leverage_ratio'

# 2. Opret designmatrice (X) og afhængig variabel (y)
y, X = dmatrices(formula, data=pivot, return_type='dataframe')

# 3. Beregn VIF for hver forklarende variabel
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# 4. Udskriv resultater
print(vif_data)
