# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
np.random.seed(1)
X = np.random.uniform(low=0.0, high=10.0, size=(1000,))

slope = 0.25
intercept = -1.25
noise = np.random.normal(loc= 0.0, scale=1.0, size=(1000,))

y = slope  *X + intercept + noise

plt.scatter(X , y , s=1)
# %% regressao linear vem do sklearn

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.get_params()

# %% treinando o modelo com base x e y
lr.fit(X.reshape(-1, 1), y)

# %% fezendo previsoes
lr.coef_, lr.intercept_
# %%
y_pred = lr.predict(X.reshape(-1, 1))
y_pred
# %%
lr.predict(np.array([0.01]).reshape(-1,1))
# %% 
plt.scatter(X , y , s=1)
plt.plot(X, y_pred, 'r')
# %%
import pandas as pd
# %%
URL = "https://raw.githubusercontent.com/TrainingByPackt/Data-Science-Projects-with-Python/refs/heads/master/Data/Chapter_1_cleaned_data.csv"
df = pd.read_csv(URL)
df
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(
    df['EDUCATION'].values.reshape(-1, 1),
    df['default payment next month'].values,
    test_size= 0.2,
    random_state=24)

print(X_test.shape)
print(X_train.shape)
print(y_test.shape)
print(y_train.shape)

# %%
np.mean(y_train), np.mean(y_test)

# %% criando o modelo
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


# %% acuracia quantidade de valores de predição que vc acertou
#pela quantidade do vetor
#como verificamos:
# sum(y_pred == y_test) / len(y_pred)

acc= np.mean(y_pred == y_test)
acc
# %% MATRIZ DE CONFUSAO -> E a contagem de combinaçoes de resultados organizados em uma matriz
# modelo = 1 -> real = 1 tp - true positive
# modelo = 0 -> real = 1 fn - false negative
# modelo = 1 -> real = 0 fp - false positive
# modelo = 0 -> real = 0 tn - true negative

TP = sum((y_pred == 1 ) & (y_test == 1))
TN = sum((y_pred == 0 ) & (y_test == 0))
FP = sum((y_pred == 1 ) & (y_test == 0))
FN = sum((y_pred == 0 ) & (y_test == 1))

TP, TN, FP, FN
# %%
#
confusion_matrix = np.array([[TN, FP], [FN, TP]])
# %%
sum(confusion_matrix.diagonal()) / confusion_matrix.sum()
# %%s predict-proba
lr.predict_proba(X_test)
