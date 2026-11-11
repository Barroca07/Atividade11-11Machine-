Projeto Olist — Previsão de Preços com Machine Learning
1. Descrição do projeto

Este projeto utiliza dados da empresa Olist para desenvolver modelos de Machine Learning capazes de prever o preço de produtos com base em suas características físicas.

O objetivo é comparar diferentes algoritmos de regressão e identificar qual apresenta o melhor desempenho.

2. Dataset utilizado

Foi utilizado o dataset público da Olist, disponível no GitHub:

https://github.com/olist/work-at-olist-data

Foram utilizados dois arquivos:

olist_order_items_dataset.csv
olist_products_dataset.csv

Os arquivos foram unidos por meio da coluna product_id.

3. Variáveis utilizadas
Features (X)
product_weight_g: peso do produto em gramas.
product_length_cm: comprimento do produto em centímetros.
product_height_cm: altura do produto em centímetros.
product_width_cm: largura do produto em centímetros.
Target (y)
price: preço do produto.
4. Pré-processamento

O pré-processamento incluiu:

Seleção das variáveis necessárias.
Remoção de registros com valores ausentes.
Remoção de preços e dimensões inválidos.
Separação entre variáveis preditoras e variável alvo.
Divisão dos dados em treinamento (80%) e teste (20%).
5. Modelos utilizados

Foram desenvolvidos três modelos:

Dummy Regressor: utilizado como referência, realizando previsões com base na média dos preços.
Regressão Linear: utilizada para identificar relações lineares entre as características físicas e o preço.
Random Forest Regressor: utiliza várias árvores de decisão para realizar previsões e identificar relações não lineares.
6. Métricas de avaliação

Foram utilizadas três métricas:

MAE: erro absoluto médio. Quanto menor, melhor.
RMSE: raiz do erro quadrático médio. Quanto menor, melhor.
R²: coeficiente de determinação. Quanto maior, melhor.
7. Resultados

O projeto apresenta uma tabela comparativa com os resultados dos três modelos.

Também são gerados gráficos para:

Distribuição dos preços.
Comparação do desempenho dos modelos.
Valores reais versus valores previstos.
Importância das variáveis no Random Forest.

O modelo com melhor desempenho será identificado automaticamente com base no maior valor de R².

8. Como executar

Instale as bibliotecas necessárias:

bash
pip install pandas numpy matplotlib seaborn scikit-learn


Execute o arquivo:

bash
python projeto_olist.py


É necessário ter conexão com a internet para carregar os arquivos CSV do GitHub.

9. Conclusão

O projeto permite analisar a aplicação de técnicas de Machine Learning na previsão de preços de produtos.

A comparação entre Dummy Regressor, Regressão Linear e Random Forest permite avaliar o desempenho dos algoritmos e verificar se os modelos conseguem superar uma previsão baseada na média.

É importante considerar que o preço de um produto depende de diversos fatores além de suas dimensões físicas, como categoria, marca e características comerciais. Portanto, as variáveis utilizadas representam apenas parte das informações que podem explicar seu preço.