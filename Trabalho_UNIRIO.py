# ============================================================
# TRABALHO 18 - ALUNOS DA UNIRIO: QUEM TRABALHA?
# ============================================================

# 1. IMPORTAÇÃO DAS BIBLIOTECAS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

sns.set_theme(style="whitegrid")

# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================

url = "https://raw.githubusercontent.com/DATAUNIRIO/Base_de_dados/master/QE.csv"

df = pd.read_csv(url)

alvo = "Trabalha"

print("Dimensão da base:", df.shape)
print(df.head())

print("\nValores ausentes:")
print(df.isnull().sum())

print("\nDistribuição das classes:")
print(df[alvo].value_counts())

# ============================================================
# 3. ANÁLISE EXPLORATÓRIA
# ============================================================

numericas = [
    "Desempenho",
    "Estresse",
    "Creditos",
    "Horas_estudo"
]

# Distribuição das classes

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Trabalha",
    hue="Trabalha",
    palette="Set2",
    legend=False
)

plt.title("Distribuição dos alunos que trabalham")
plt.show()

# Histogramas

df[numericas].hist(
    figsize=(12, 8),
    bins=10,
    color="skyblue",
    edgecolor="black"
)

plt.tight_layout()
plt.show()

# Comparação por situação de trabalho

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for coluna, ax in zip(numericas, axes.flatten()):
    sns.boxplot(
        data=df,
        x="Trabalha",
        y=coluna,
        hue="Trabalha",
        palette="Set2",
        legend=False,
        ax=ax
    )
    ax.set_title(coluna)

plt.tight_layout()
plt.show()

# Correlação

plt.figure(figsize=(9, 6))

sns.heatmap(
    df[numericas].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlação entre variáveis numéricas")
plt.show()

# ============================================================
# 4. PRÉ-PROCESSAMENTO
# ============================================================

caracteristicas = [
    "Desempenho",
    "Estresse",
    "Creditos",
    "Horas_estudo",
    "Mora_pais",
    "RJ",
    "Namorado_a"
]

X = df[caracteristicas].copy()
y = df[alvo].copy()

categoricas = [
    "Mora_pais",
    "RJ",
    "Namorado_a"
]

processamento_numerico = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

processamento_categorico = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessador = ColumnTransformer([
    ("num", processamento_numerico, numericas),
    ("cat", processamento_categorico, categoricas)
])

# ============================================================
# 5. DIVISÃO DOS DADOS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Treinamento:", X_train.shape)
print("Teste:", X_test.shape)

# ============================================================
# 6. CRIAÇÃO DOS MODELOS
# ============================================================

modelo_lr = Pipeline([
    ("preprocessamento", preprocessador),
    ("modelo", LogisticRegression(max_iter=1000))
])

modelo_arvore = Pipeline([
    ("preprocessamento", preprocessador),
    ("modelo", DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    ))
])

# ============================================================
# 7. TREINAMENTO
# ============================================================

modelo_lr.fit(X_train, y_train)
modelo_arvore.fit(X_train, y_train)

print("Modelos treinados com sucesso!")

# ============================================================
# 8. AVALIAÇÃO
# ============================================================

pred_lr = modelo_lr.predict(X_test)
pred_arvore = modelo_arvore.predict(X_test)

acc_lr = accuracy_score(y_test, pred_lr)
acc_arvore = accuracy_score(y_test, pred_arvore)

print("\nAcurácia Regressão Logística:",
      round(acc_lr * 100, 2), "%")

print("Acurácia Árvore de Decisão:",
      round(acc_arvore * 100, 2), "%")

print("\nREGRESSÃO LOGÍSTICA")

print(classification_report(
    y_test,
    pred_lr,
    zero_division=0
))

print("\nÁRVORE DE DECISÃO")

print(classification_report(
    y_test,
    pred_arvore,
    zero_division=0
))

# ============================================================
# 9. MATRIZES DE CONFUSÃO
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    pred_lr,
    ax=axes[0],
    cmap="Blues"
)

axes[0].set_title("Regressão Logística")

ConfusionMatrixDisplay.from_predictions(
    y_test,
    pred_arvore,
    ax=axes[1],
    cmap="Greens"
)

axes[1].set_title("Árvore de Decisão")

plt.tight_layout()
plt.show()

# ============================================================
# 10. TESTE DE DIFERENTES RANDOM_STATE
# ============================================================

estados = [1, 10, 20, 42, 50, 100, 123, 200]

resultados = []

for estado in estados:

    X_train_exp, X_test_exp, y_train_exp, y_test_exp = (
        train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=estado,
            stratify=y
        )
    )

    modelo = Pipeline([
        ("preprocessamento", preprocessador),
        ("modelo", LogisticRegression(max_iter=1000))
    ])

    modelo.fit(X_train_exp, y_train_exp)

    previsoes = modelo.predict(X_test_exp)

    acuracia = accuracy_score(
        y_test_exp,
        previsoes
    )

    resultados.append({
        "Random State": estado,
        "Acurácia": acuracia
    })

resultado_df = pd.DataFrame(resultados)

print("\nResultados dos random_state:")
print(resultado_df.to_string(index=False))

# Gráfico

plt.figure(figsize=(10, 5))

sns.barplot(
    data=resultado_df,
    x="Random State",
    y="Acurácia",
    color="cornflowerblue"
)

plt.title("Acurácia por random_state")
plt.ylim(0, 1)
plt.show()

# ============================================================
# 11. CONCLUSÃO
# ============================================================

melhor_modelo = (
    "Regressão Logística"
    if acc_lr >= acc_arvore
    else "Árvore de Decisão"
)

print("\n========== CONCLUSÃO ==========")

print("Melhor modelo neste teste:", melhor_modelo)

print("Média das acurácias:",
      round(resultado_df["Acurácia"].mean() * 100, 2), "%")

print("Maior acurácia:",
      round(resultado_df["Acurácia"].max() * 100, 2), "%")

print("Menor acurácia:",
      round(resultado_df["Acurácia"].min() * 100, 2), "%")

print("""
O trabalho utilizou Machine Learning para identificar
padrões relacionados à situação de trabalho dos alunos
da UNIRIO.

Foram realizadas análise exploratória, preparação dos
dados, treinamento e avaliação de dois modelos:
Regressão Logística e Árvore de Decisão.

Também foram testados diferentes valores de
random_state para verificar a variação dos resultados.

Como a base possui apenas 95 alunos, os resultados
devem ser interpretados com cautela.
""")

print("========== FIM DO TRABALHO ==========")