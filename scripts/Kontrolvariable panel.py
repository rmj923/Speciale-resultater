import pandas as pd
import numpy as np

# Indlæs data
file_path = '/Users/emilerbas/Desktop/data.2.csv'
data = pd.read_csv(file_path)

# Filtrér relevante variable og bankstørrelser
relevante_cb = ['I2004', 'KSV12', 'I7000', 'I2513']  # ROA, leverage, NPL, net interest income
relevante_størrelser = ['SL30', 'ST10', 'ST20', 'SM20', 'GSIB']
data = data[data['CB_ITEM'].isin(relevante_cb) & data['SBS_BREAKDOWN'].isin(relevante_størrelser)]

# Pivotér data
pivot = data.pivot_table(index=['TIME_PERIOD', 'SBS_BREAKDOWN'], columns='CB_ITEM', values='OBS_VALUE').reset_index()
pivot = pivot.rename(columns={
    'I2004': 'return_on_assets',
    'KSV12': 'leverage_ratio',
    'I7000': 'npl_ratio',
    'I2513': 'net_interest_income_ratio'
})

# Tilføj labels og definer korrekt rækkefølge
labels = {
    'SL30': 'Små banker',
    'ST10': 'Mellemstore banker',
    'ST20': 'Store banker',
    'SM20': 'Meget store banker',
    'GSIB': 'Mega banker'
}
pivot['bank_stoerrelse'] = pivot['SBS_BREAKDOWN'].map(labels)

# Fjern rækker med NaNs
pivot = pivot.dropna(subset=[
    'return_on_assets', 'leverage_ratio', 'npl_ratio', 'net_interest_income_ratio'
])

# Definér ønsket rækkefølge
bank_rækkefølge = ['Små banker', 'Mellemstore banker', 'Store banker', 'Meget store banker', 'Mega banker']

# Beregn deskriptiv statistik for hver kontrolvariabel i korrekt rækkefølge
nøgletal = ['return_on_assets', 'leverage_ratio', 'npl_ratio', 'net_interest_income_ratio']
for variable in nøgletal:
    print(f"\nDeskriptiv statistik for: {variable.replace('_', ' ').title()}")
    stats = pivot.groupby('bank_stoerrelse')[variable].agg(['mean', 'median', 'std', 'min', 'max']).round(3)
    stats = stats.loc[bank_rækkefølge]  # sorter rækkerne i rigtig rækkefølge
    print(stats)
