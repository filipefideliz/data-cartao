# %% TRATAMENTO DE DADOS
import pandas as pd
# %%
df = pd.read_excel("../CARTAO/default_of_credit_card_clients__courseware_version_1_21_19.xls")
df.head()
# %% qtds de linhas e colunas
df.shape

# %% verificando os dados
df.columns
# %% qtd de colunas
len(df.columns)
# %% VERIFICANDO ID UNICOS
df["ID"].unique()
# %%
id_counts = df["ID"].value_counts()
# %% formar de calcular os id unicos e nao unicos
id_counts.value_counts()
# %% verificar quais sao ids duplos 
duple_mask = id_counts == 2
duple_mask
# %% selecionar os ids duplicados
dupe_idIs = id_counts.index[duple_mask]
# %% transforma em lista ids duplos
dupe_ids = list(dupe_idIs)
len(dupe_ids)
# %% condiçao para mostrar os ids repetidos no dataframe
df.loc[df["ID"].isin(dupe_ids[0:3]), :].head(10)
# %% variavel que armazena os valores zeros dentro do dataframe
df_zero_mask = df == 0
# %%
## dfzeromask pego os val iguais a 0 
# no iloc pego pela numeração do indice sequencial
#: todas as linhas
# 1: todas as colunas menos a primeira
# pego todos os valores no .all
# axis = 1 dentro das colunas  

feature_zero_mask = df_zero_mask.iloc[:, 1:].all(axis=1)
sum(feature_zero_mask)
# %% o ~ inverte para mostrar o valor false deveio o true
# assim elminando os linhas 0
df_clean_1 = df.loc[~feature_zero_mask, :].copy()

# %% verificando o tamanho
df_clean_1.shape
# %% VERIFICANDO SE OS VALORES ESTAO BATENDO COMO VALORES UNICOS
df_clean_1["ID"].nunique()
# %%
df_clean_1.reset_index(drop=True).info()
# %% explorando dados
df_clean_1.info()
# %% informações sobre o data set
df_clean_1["PAY_1"].head(5)
# %%
df_clean_1["PAY_1"].value_counts()
# %%
valid_pay_1_mask = df_clean_1["PAY_1"] != "Not available"
valid_pay_1_mask
# %%
sum(valid_pay_1_mask)
# %% 
df_clean_2 = df_clean_1.loc[valid_pay_1_mask, :].copy()

# %%
df_clean_2.shape 
# %%
df_clean_2.info()
# %%
df_clean_2["PAY_1"] = df_clean_2["PAY_1"].astype(int)
# %%
df_clean_2.info()
# %% EXPLICAÇAÕ POR QUR TIRA ESSES VALORES
#-1 USOU O VALOR QUE FOI TOTALMENTE PAGO
#-2 CREDITO NAO USADO
# 0 PAGAMENTO MINIMO FOI FEITO
import matplotlib as mpl
import matplotlib.pyplot as plt
# %%

df_clean_2[["AGE",'LIMIT_BAL']].hist()
# %%
df_clean_2[["AGE","LIMIT_BAL"]].describe()
# %% VALORES 0 5 6 NAO ENCONTRASE SIGINIFICADO NA DESCRIÇAO DO NOSSO DATASET
df_clean_2["EDUCATION"].value_counts()
# %%
df_clean_2["EDUCATION"].replace(to_replace=[0,5,6], value=4, inplace=True)
df_clean_2["EDUCATION"].value_counts()
# %%
df_clean_2["MARRIAGE"].value_counts()
# %%
df_clean_2["MARRIAGE"].replace(to_replace=0, value=3, inplace=True)
df_clean_2["MARRIAGE"].value_counts()
# %% CARACTERISTICAS CATEGORICAS - O ML NORMALMENTE TRANSFORMA CATEGORIA EM NUMEROS PARA AJUDA NO MODELO
df_clean_2.groupby('EDUCATION').agg({'default payment next month': 'mean'}).plot.bar(legend=False)
plt.ylabel('Default Rate')
plt.xlabel('Education Level: ordinal enconding')
# %% implementando one-hot encoding
#criando coluna vazia
df_clean_2['EDUCATION_CAT'] = 'none'
# %% examinando as primeiras linhas
df_clean_2[['EDUCATION','EDUCATION_CAT']].head(10)
# %% criando um dicionario de mapeamento
cat_mapping = {
    1:'graduate school',
    2:'university',
    3:"high school",
    4:"others"
}
cat_mapping
# %%Aplicando o mapeamento de categoria
df_clean_2["EDUCATION_CAT"]= df_clean_2["EDUCATION"].map(cat_mapping)
df_clean_2[['EDUCATION','EDUCATION_CAT']].head(10)
# %% codificação de caracteristicas do OHE
edu_ohe = pd.get_dummies(df_clean_2["EDUCATION_CAT"])
edu_ohe.head(10)
# %% Concatenando o dataframe original com OHE
df_with_one = pd.concat([df_clean_2, edu_ohe], axis=1)
df_with_one[['EDUCATION_CAT','graduate school', 'high school', 'others', 'university']].head(10)
# %%
df_with_one.to_csv('chapter1_data_cleaner.csv', index=False)