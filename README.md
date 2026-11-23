# TSPE v2 · Classificação de risco de mortalidade hospitalar no SUS em Pernambuco

Disciplina WYD6399 Tópicos Avançados em Big Data com Python · UniFavip Wyden · Prof. Melquezedeque Lima

Equipe: Julio César Batista Silva, Teves Gabriel, Luciano Filho, Jonathas Costa e Anderson Davi

**Simulador:** https://juliobatista04.github.io/TSPE-v2/
**Análise completa (4 partes):** https://juliobatista04.github.io/TSPE-v2/frontend/index.html

## Resumo

Com dados públicos do SIH/SUS (DATASUS, TABNET) de Pernambuco, montamos uma base de 239 grupos de internação. Cada grupo é um capítulo da CID-10 combinado com uma faixa etária. O modelo classifica cada grupo como **alto risco** (taxa de mortalidade hospitalar acima da mediana do estado, 0,90%) ou **baixo risco** (igual ou abaixo). O objetivo é apoiar o planejamento de leitos, equipes e protocolos. O modelo trabalha com grupos, não com pacientes.

## Aprendizado de máquina utilizado

### Tipo de problema
Classificação binária supervisionada. Cada registro tem as variáveis de entrada e o rótulo conhecido (`alto_risco` = 0 ou 1), e o modelo aprende a relação entre os dois.

### Variáveis de entrada
Começamos com 7 candidatas e ficamos com 4, escolhidas por três critérios calculados só no treino: informação mútua, correlação de Spearman com o alvo e importância por permutação. Entrou quem passou em pelo menos dois.

| Variável | O que é |
|---|---|
| `tx_mort_melhor_hosp` | taxa de mortalidade do hospital com mais internações naquele capítulo da CID |
| `media_permanencia` | média de dias de internação do grupo |
| `tx_mort_macro` | taxa de mortalidade do estado naquela faixa etária |
| `faixa_etaria_ord` | faixa etária como número de 0 a 11 (a ordem das idades é mantida) |

A taxa de mortalidade do próprio grupo ficou de fora, porque é dela que sai o alvo.

### Algoritmos comparados

| Algoritmo | Como funciona, em uma linha | Hiperparâmetros testados |
|---|---|---|
| Regressão Logística | combinação linear das variáveis transformada em probabilidade | C (regularização) |
| KNN | olha os k grupos mais parecidos e vota | k, peso por distância, tipo de distância |
| Árvore de Decisão | sequência de perguntas do tipo "permanência > 6 dias?" | profundidade, mínimo por folha, critério |
| Random Forest | várias árvores treinadas em amostras diferentes, decisão por maioria | nº de árvores, profundidade, mínimo por folha, nº de variáveis por divisão |
| SVM (kernel RBF) | fronteira curva que separa as classes com a maior margem possível | C, gamma |
| Gradient Boosting | árvores pequenas em sequência, cada uma corrigindo o erro da anterior | nº de árvores, taxa de aprendizado, profundidade, subamostra |

### Como treinamos e avaliamos
- Divisão estratificada: 75% para treino (179 grupos) e 25% para teste (60 grupos), semente 42.
- Pré-processamento dentro de um `Pipeline` do scikit-learn: a padronização (`StandardScaler`) é ajustada só nos dados de treino de cada dobra.
- `GridSearchCV` com `StratifiedKFold` de 5 dobras, otimizando AUROC. Foram 157 combinações de hiperparâmetros no total.
- O modelo final foi escolhido pela maior AUROC na validação cruzada. O teste só foi usado uma vez, no fim.
- Métricas no teste: AUROC, F1, sensibilidade, especificidade, precisão, acurácia e matriz de confusão.

### Resultados

| Modelo | AUROC CV | AUROC teste | F1 | Acurácia | Sensibilidade | Especificidade |
|---|---|---|---|---|---|---|
| **Random Forest (escolhido)** | 0,977 | 0,951 | 0,900 | 0,900 | 0,900 | 0,900 |
| Gradient Boosting | 0,975 | 0,956 | 0,951 | 0,950 | 0,967 | 0,933 |
| SVM (RBF) | 0,958 | 0,953 | 0,903 | 0,900 | 0,933 | 0,867 |
| KNN | 0,953 | 0,929 | 0,839 | 0,833 | 0,867 | 0,800 |
| Regressão Logística | 0,935 | 0,888 | 0,820 | 0,817 | 0,833 | 0,800 |
| Árvore de Decisão | 0,914 | 0,883 | 0,788 | 0,767 | 0,867 | 0,667 |

Matriz de confusão do Random Forest no teste: 27 verdadeiros negativos, 3 falsos positivos, 3 falsos negativos e 27 verdadeiros positivos.

Configuração final do Random Forest: 200 árvores, `max_features="sqrt"`, sem limite de profundidade, `class_weight="balanced"`.

### Teste sem as variáveis de contexto
As duas taxas de contexto vêm da mesma fonte que o alvo, só que em outro nível de agregação. Para checar se isso inflava o resultado, treinamos de novo sem elas (`src/ablacao.py`):

| Entradas | RF AUROC CV | RF AUROC teste |
|---|---|---|
| As 4 selecionadas | 0,976 | 0,953 |
| Permanência e idade | 0,909 | 0,843 |
| Permanência, idade e capítulo da CID (sem nenhuma taxa de mortalidade) | 0,956 | 0,943 |

Sem nenhuma taxa de mortalidade na entrada, o modelo continua com AUROC acima de 0,94. O sinal vem da combinação doença, idade e tempo de internação.

### Limitações
- 239 grupos é pouco: no teste, cada erro a mais muda a acurácia em cerca de 1,7 ponto percentual.
- O alvo é um corte estatístico (mediana), não um desfecho clínico individual.
- As taxas de contexto foram calculadas no mesmo período do alvo.

## Como rodar

```bash
pip install -r requirements.txt
python src/rodar_tudo.py   # tratamento, análise, seleção, treino, teste de ablação, interface e documentação (cerca de 2 min)
python api.py              # API em http://localhost:5000 (opcional: o simulador da página não precisa dela)
```

Sem Python instalado, use `notebooks/TSPE_v2_pipeline_colab.ipynb` no Google Colab.

## O que tem em cada pasta

| Caminho | Conteúdo |
|---|---|
| `index.html` | página inicial: resumo, simulador e lista de materiais |
| `frontend/index.html` | análise completa em 4 partes (problema, metodologia, gráficos, resultados) |
| `src/` | um script por etapa: `tratamento.py`, `analise.py`, `selecao_variaveis.py`, `preprocessamento.py`, `treinar.py`, `ablacao.py` e geradores da interface e da documentação |
| `data/raw/` | onde ficam os 12 CSVs exportados do TABNET (ver o README da pasta) |
| `data/processado/` | base tratada (`dataset_ml.csv`) |
| `modelos/` | Random Forest treinado e metadados |
| `resultados/` | números em JSON e gráficos em PNG |
| `docs/` | documentação, guia passo a passo e figura da metodologia (`.drawio` e `.png`) |
| `api.py` | API Flask com a rota `/prever` |

## API

```bash
curl -X POST http://localhost:5000/prever -H "Content-Type: application/json" \
  -d '{"doenca":"Neoplasias (tumores)","faixa_etaria":"70 a 79 anos","media_permanencia":8}'
```

## Fonte dos dados

Ministério da Saúde. DATASUS. SIH/SUS, Morbidade Hospitalar do SUS por local de internação, Pernambuco. TABNET: https://datasus.saude.gov.br/informacoes-de-saude-tabnet/
