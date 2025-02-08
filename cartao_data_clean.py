# %%
import pandas as pd
# %%
df = pd.read_csv("../CARTON/Chapter_1_cleaned_data.csv")
df
# %%
df['default payment next month'].value_counts(normalize=True) * 100
# %%regressao logistica
# inicializar o algoritimo
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
# %% random state controla e aleotoriedade do  algoritimo
lr.get_params(C = 1000, random_stage = 42)
# %% treinar algoritimo
X =df[['EDUCATION']].values.reshape(-1, 1)
y = df['default payment next month'].values
lr.fit(X, y)
# realizar previsoes
lr.predict(df[['EDUCATION']][10:20].values.reshape(-1, 1))
