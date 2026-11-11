
# ============================================================
# PROJETO: PREVISÃO DE PREÇOS DE PRODUTOS - OLIST
# Objetivo: Prever o preço dos produtos utilizando Machine Learning
# ============================================================

# 1. IMPORTAÇÃO DAS BIBLIOTECAS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import warnings
warnings.filterwarnings("ignore")


# 2. CARREGAMENTO DO DATASET

pasta = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/"

itens = pd.read_csv(pasta + "olist_order_items_dataset.csv")
produtos = pd.read_csv(pasta + "olist_products_dataset.csv")

# União dos datasets pelo product_id
olist = itens.merge(produtos, on="product_id")

print("Dimensões do dataset:", olist.shape)

print("\nPrimeiras linhas:")
print(olist.head())


# 3. ANÁLISE EXPLORATÓRIA

print("\nInformações do dataset:")
olist.info()

print("\nEstatísticas descritivas:")
print(olist.describe())

print("\nValores ausentes:")
print(olist.isnull().sum())

# Distribuição dos preços
plt.figure(figsize=(10, 5))
sns.histplot(olist["price"], bins=50, kde=True)
plt.title("Distribuição dos preços dos produtos")
plt.xlabel("Preço (R$)")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()


# 4. PRÉ-PROCESSAMENTO

colunas = [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "price"
]

# Seleção das colunas e remoção de valores ausentes
dados = olist[colunas].dropna().copy()

# Remoção de valores inválidos
dados = dados[
    (dados["price"] > 0) &
    (dados["product_weight_g"] > 0) &
    (dados["product_length_cm"] > 0) &
    (dados["product_height_cm"] > 0) &
    (dados["product_width_cm"] > 0)
]

# Separação entre features e target
X = dados.drop(columns=["price"])
y = dados["price"]

print("\nTotal de registros:", len(dados))
print("Variáveis preditoras:", X.columns.tolist())
print("Variável alvo: price")


# 5. SEPARAÇÃO ENTRE TREINO E TESTE

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDados de treinamento:", X_train.shape)
print("Dados de teste:", X_test.shape)


# 6. CRIAÇÃO DOS MODELOS

# Modelo de referência
dummy = DummyRegressor(strategy="mean")

# Regressão Linear
linear = LinearRegression()

# Random Forest
forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# 7. TREINAMENTO DOS MODELOS

dummy.fit(X_train, y_train)
linear.fit(X_train, y_train)
forest.fit(X_train, y_train)

print("\nTodos os modelos foram treinados!")


# 8. REALIZAÇÃO DAS PREVISÕES

pred_dummy = dummy.predict(X_test)
pred_linear = linear.predict(X_test)
pred_forest = forest.predict(X_test)

print("Previsões realizadas com sucesso!")


# 9. AVALIAÇÃO DOS MODELOS

def avaliar(nome, y_true, y_pred):
    return {
        "Modelo": nome,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }


placar = pd.DataFrame([
    avaliar("Dummy Regressor", y_test, pred_dummy),
    avaliar("Regressão Linear", y_test, pred_linear),
    avaliar("Random Forest", y_test, pred_forest)
])

placar = placar.sort_values(
    by="R2",
    ascending=False
)

print("\nCOMPARAÇÃO DOS MODELOS:")
print(placar.to_string(index=False))


# 10. GRÁFICO DE COMPARAÇÃO

plt.figure(figsize=(10, 5))

sns.barplot(
    data=placar,
    x="Modelo",
    y="R2"
)

plt.title("Comparação do desempenho dos modelos")
plt.xlabel("Modelos")
plt.ylabel("R²")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# 11. VALORES REAIS X PREVISTOS

previsoes_modelos = {
    "Dummy Regressor": pred_dummy,
    "Regressão Linear": pred_linear,
    "Random Forest": pred_forest
}

melhor_modelo = placar.iloc[0]["Modelo"]
previsoes = previsoes_modelos[melhor_modelo]

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    previsoes,
    alpha=0.4
)

plt.xlabel("Preço real (R$)")
plt.ylabel("Preço previsto (R$)")
plt.title(f"Valores reais x previstos - {melhor_modelo}")

plt.tight_layout()
plt.show()


# 12. IMPORTÂNCIA DAS VARIÁVEIS

importancias = pd.DataFrame({
    "Variável": X.columns,
    "Importância": forest.feature_importances_
})

importancias = importancias.sort_values(
    by="Importância",
    ascending=False
)

print("\nIMPORTÂNCIA DAS VARIÁVEIS:")
print(importancias.to_string(index=False))

plt.figure(figsize=(9, 5))

sns.barplot(
    data=importancias,
    x="Importância",
    y="Variável"
)

plt.title("Importância das variáveis - Random Forest")
plt.tight_layout()
plt.show()


# 13. RESULTADO FINAL

melhor = placar.iloc[0]

print("\n====================================")
print("           RESULTADO FINAL")
print("====================================")

print("Melhor modelo:", melhor["Modelo"])
print("MAE:", round(melhor["MAE"], 2))
print("RMSE:", round(melhor["RMSE"], 2))
print("R²:", round(melhor["R2"], 4))

print("====================================")
print("Projeto finalizado com sucesso!")
