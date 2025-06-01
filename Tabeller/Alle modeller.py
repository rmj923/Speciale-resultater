import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from statsmodels.iolib.summary2 import summary_col
from linearmodels.panel import RandomEffects
import statsmodels.api as sm

# 1. Indlæs data
file_path = '/Users/emilerbas/Desktop/SUP_renset.csv'
data = pd.read_csv(file_path)

# 2. Filtrér relevante CB_ITEM og bankstørrelser
relevante_cb = ['E0035', 'E0038', 'I2004', 'KSV12', 'I7000', 'I2513', 'A0001', 'A0000']
relevante_størrelser = ['SL30', 'ST10', 'ST20', 'SM20', 'GSIB', 'NEEA', 'EEA', 'ROW', 'SSM', 'DOM']
data = data[data['CB_ITEM'].isin(relevante_cb) & data['SBS_BREAKDOWN'].isin(relevante_størrelser)]

# 3. Pivotér data
pivot = data.pivot_table(index=['TIME_PERIOD', 'SBS_BREAKDOWN'], columns='CB_ITEM', values='OBS_VALUE').reset_index()
pivot = data.pivot_table(index=['TIME_PERIOD', 'SBS_BREAKDOWN'], columns='CB_ITEM', values='OBS_VALUE').reset_index()

pivot = pivot.rename(columns={
    'E0035': 'total_loans',
    'E0038': 'sme_loans',
    'I2004': 'return_on_assets',
    'I7000': 'npl_ratio',
    'I2513': 'net_interest_income_ratio',
    'KSV12': 'leverage_ratio',
    'A0001': 'assets_per_bank',
    'A0000': 'total_assets'
})

# 4. Beregn andel af SMV-lån
pivot['andel_smv'] = pivot['sme_loans'] / pivot['total_loans']
pivot['log_assets'] = np.log(pivot['assets_per_bank'])
pivot['log_total_loans'] = np.log(pivot['total_loans'])
pivot['log_assets_geo'] = np.log(pivot['total_assets'])
pivot = pivot.dropna(subset=['andel_smv', 'npl_ratio', 'net_interest_income_ratio', 'leverage_ratio'])

# 7. Tilføj bankstørrelseslabels og tidstrend
labels_dansk = {
    'NEEA': 'Ikke-EØS Europa',
    'EEA': 'EØS (uden for SSM)',
    'ROW': 'Resten af verden (RoW)',
    'SSM': 'SSM-lande',
    'DOM': 'Hjemmemarkedseksponering'
}
pivot['geografisk_eksponering'] = pivot['SBS_BREAKDOWN'].map(labels_dansk)

geo_order = [
    'Hjemmemarkedseksponering',
    'SSM-lande',
    'EØS (uden for SSM)',
    'Ikke-EØS Europa',
    'Resten af verden (RoW)'
]

geo_dtype = CategoricalDtype(categories=geo_order, ordered=False)
pivot['geografisk_eksponering'] = pivot['geografisk_eksponering'].astype(geo_dtype)

# 5. Tilføj bankstørrelseslabels
labels = {
    'SL30': 'Små banker',
    'ST10': 'Mellemstore banker',
    'ST20': 'Store banker',
    'SM20': 'Meget store banker',
    'GSIB': 'Mega banker'
}
pivot['bank_stoerrelse'] = pivot['SBS_BREAKDOWN'].map(labels)

# 6. Sortér bankstørrelser i ønsket rækkefølge
bank_order = [
    "Små banker",
    "Mellemstore banker",
    "Store banker",
    "Meget store banker",
    "Mega banker"
]

pivot["bank_stoerrelse"] = pd.Categorical(
    pivot["bank_stoerrelse"],
    categories=bank_order,
    ordered=True
)
pivot['time_trend'] = np.arange(len(pivot))
pivot['time_trend'] = pivot['time_trend'] / pivot['time_trend'].max()

size_map = {
    'Små banker': 1,
    'Mellemstore banker': 2,
    'Store banker': 3,
    'Meget store banker': 4,
    'Mega banker': 5
}
pivot['bank_size_numeric'] = pivot['bank_stoerrelse'].map(size_map)
pivot['bank_size_numeric'] = pivot['bank_stoerrelse'].map(size_map).astype(float)

model1 = smf.ols('andel_smv ~ C(bank_stoerrelse, Treatment(reference="Små banker"))', data=pivot).fit(cov_type='HC1')
model2 = smf.ols('andel_smv ~ C(bank_stoerrelse, Treatment(reference="Små banker")) + time_trend', data=pivot).fit(cov_type='HC1')  
model3a = smf.ols('andel_smv ~ C(bank_stoerrelse, Treatment(reference="Små banker")) + time_trend + return_on_assets + npl_ratio + net_interest_income_ratio + leverage_ratio',data=pivot).fit(cov_type='HC1')
model4a = smf.ols('andel_smv ~ C(bank_stoerrelse, Treatment(reference="Små banker")) * leverage_ratio + time_trend + return_on_assets + npl_ratio + net_interest_income_ratio', data=pivot).fit(cov_type='HC1')
model4b = smf.ols('andel_smv ~ C(bank_stoerrelse, Treatment(reference="Små banker")) *  return_on_assets + leverage_ratio + npl_ratio + net_interest_income_ratio + time_trend', data=pivot).fit(cov_type='HC1')
model4c = smf.ols('andel_smv ~ C(bank_stoerrelse, Treatment(reference="Små banker")) * npl_ratio + return_on_assets + leverage_ratio + time_trend + net_interest_income_ratio', data=pivot).fit(cov_type='HC1')
model4d = smf.ols('andel_smv ~ C(bank_stoerrelse, Treatment(reference="Små banker")) * net_interest_income_ratio +  return_on_assets + npl_ratio + leverage_ratio + time_trend', data=pivot).fit(cov_type='HC1')
model3a_numerisk = smf.ols('andel_smv ~ bank_size_numeric + time_trend + return_on_assets + npl_ratio + net_interest_income_ratio + leverage_ratio',data=pivot).fit(cov_type='HC1')
model5= smf.ols('andel_smv ~ log_assets + time_trend + return_on_assets + npl_ratio + net_interest_income_ratio + leverage_ratio', data=pivot).fit(cov_type='HC1')
model6= smf.ols('andel_smv ~ log_total_loans + time_trend + return_on_assets + npl_ratio + net_interest_income_ratio + leverage_ratio', data=pivot).fit(cov_type='HC1')
model7a = smf.ols('log_assets_geo ~ C(geografisk_eksponering) + time_trend + return_on_assets + npl_ratio + net_interest_income_ratio + leverage_ratio', data=pivot).fit(cov_type='HC1')
pivot['log_assets_hat'] = model7a.fittedvalues
model7b = smf.ols('andel_smv ~ log_assets_hat + time_trend + return_on_assets + npl_ratio + net_interest_income_ratio + leverage_ratio',data=pivot).fit(cov_type='HC1')

# Indtast model for output
print(model7b.summary())

Model1_3 = summary_col(
    results=[model1, model2, model3a],
    model_names=["Model 1", "Model 2", "Model 3a"],
    stars=True,
    float_format="%.3f",
    info_dict={
        'R-squared': lambda x: f"{x.rsquared:.3f}",
        'Adj. R-squared': lambda x: f"{x.rsquared_adj:.3f}",
        'N': lambda x: f"{int(x.nobs)}"
    }
)
latex_code = Model1_3.as_latex()
print(latex_code)

Model4 = summary_col(
    results=[model4a, model4b, model4c, model4d],
    model_names=["Model 4a", "Model 4b", "Model 4c", 'Model 4d'],
    stars=True,
    float_format="%.3f",
    info_dict={
        'R-squared': lambda x: f"{x.rsquared:.3f}",
        'Adj. R-squared': lambda x: f"{x.rsquared_adj:.3f}",
        'N': lambda x: f"{int(x.nobs)}"
    }
)
latex_code = Model4.as_latex()
print(latex_code)


Model5_7 = summary_col(
    results=[model3_numerisk, model5, model6, model7b],
    model_names=["Model 3 numerisk", "Model 5", "Model 6", 'Model 7'],
    stars=True,
    float_format="%.3f",
    info_dict={
        'R-squared': lambda x: f"{x.rsquared:.3f}",
        'Adj. R-squared': lambda x: f"{x.rsquared_adj:.3f}",
        'N': lambda x: f"{int(x.nobs)}"
    }
)
latex_code = Model5_7.as_latex()
print(latex_code)


pivot['TIME_PERIOD'] = pd.to_datetime(pivot['TIME_PERIOD'])
data3b = pivot[pivot['TIME_PERIOD'] < '2021-07-01'].copy()
data3c = pivot[pivot['TIME_PERIOD'] >= '2021-07-01'].copy()
model3b = smf.ols('andel_smv ~ C(bank_stoerrelse, Treatment(reference="Små banker")) + time_trend + return_on_assets + npl_ratio + net_interest_income_ratio + leverage_ratio', data=data3b).fit(cov_type='HC1')
model3c = smf.ols('andel_smv ~ C(bank_stoerrelse, Treatment(reference="Små banker")) + time_trend + return_on_assets + npl_ratio + net_interest_income_ratio + leverage_ratio', data=data3c).fit(cov_type='HC1')
Model3 = summary_col(
    results=[model3a, model3b, model3c],
    model_names=["Model 3a", "Model 3b", "Model 3c"],
    stars=True,
    float_format="%.3f",
    info_dict={
        'R-squared': lambda x: f"{x.rsquared:.3f}",
        'Adj. R-squared': lambda x: f"{x.rsquared_adj:.3f}",
        'N': lambda x: f"{int(x.nobs)}"
    }
)
latex_code = Model3.as_latex()
print(latex_code)

with open('/Users/emilerbas/Desktop/Tabeller/model1.txt', 'w') as f:
    f.write(model1.summary().as_text())
with open('/Users/emilerbas/Desktop/Tabeller/model2.txt', 'w') as f:
    f.write(model2.summary().as_text())
with open('/Users/emilerbas/Desktop/Tabeller/model3a.txt', 'w') as f:
    f.write(model3a.summary().as_text())
with open('/Users/emilerbas/Desktop/Tabeller/model3a_numerisk.txt', 'w') as f:
    f.write(model3a_numerisk.summary().as_text())
with open('/Users/emilerbas/Desktop/Tabeller/model3b.txt', 'w') as f:
    f.write(model3b.summary().as_text())
with open('/Users/emilerbas/Desktop/Tabeller/model3c.txt', 'w') as f:
    f.write(model3c.summary().as_text())
with open('/Users/emilerbas/Desktop/Tabeller/model4a.txt', 'w') as f:
    f.write(model4a.summary().as_text())
with open('/Users/emilerbas/Desktop/Tabeller/model4b.txt', 'w') as f:
    f.write(model4b.summary().as_text())
with open('/Users/emilerbas/Desktop/Tabeller/model4c.txt', 'w') as f:
    f.write(model4c.summary().as_text())
with open('/Users/emilerbas/Desktop/Tabeller/model4d.txt', 'w') as f:
    f.write(model4d.summary().as_text())
with open('/Users/emilerbas/Desktop/Tabeller/model5.txt', 'w') as f:
    f.write(model5.summary().as_text())
with open('/Users/emilerbas/Desktop/Tabeller/model6.txt', 'w') as f:
    f.write(model6.summary().as_text())
with open('/Users/emilerbas/Desktop/Tabeller/model7a.txt', 'w') as f:
    f.write(model7a.summary().as_text())
with open('/Users/emilerbas/Desktop/Tabeller/model7b.txt', 'w') as f:
    f.write(model7b.summary().as_text())

