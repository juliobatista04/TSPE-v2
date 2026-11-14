# Classificação de risco de mortalidade hospitalar em Pernambuco com Machine Learning

**Disciplina:** WYD6399 · Tópicos em Big Data em Python · UniFavip Wyden
**Professor:** Me. Flávio Morais
**Equipe:** Julio César Batista Silva · Teves Gabriel · Luciano Filho · Jonathas Costa · Anderson Davi
**Data:** 14/11/2026 · Apresentação: 30/11/2026

---

## 1. Introdução

### 1.1 Contexto e relevância
O Sistema Único de Saúde registra todas as internações financiadas com recurso público no Sistema de Informações Hospitalares (SIH/SUS). Em Pernambuco, essas internações se distribuem por 164 municípios e por 20 capítulos da CID-10. A taxa de mortalidade hospitalar (óbitos a cada 100 internações) varia de 0% em grupos de baixo risco até 40,5% nos grupos mais críticos. Para quem planeja a rede de atenção, saber com antecedência quais combinações de doença e idade concentram o risco ajuda a alocar leitos, equipes e protocolos clínicos.

### 1.2 Problema de classificação
O projeto trata o problema como uma **tarefa de classificação binária supervisionada**. Cada registro é um grupo formado por um capítulo da CID-10 e uma faixa etária. A variável-alvo é:

- **alto_risco = 1 (Alto risco):** taxa de mortalidade hospitalar do grupo acima da mediana estadual (0,90%);
- **alto_risco = 0 (Baixo risco):** taxa igual ou abaixo da mediana.

O corte pela mediana foi escolhido por três razões: (i) é um critério reprodutível e sem arbitrariedade clínica; (ii) produz classes equilibradas (120 × 119), o que dispensa técnicas de balanceamento artificial; (iii) permite que o resultado seja lido como "este grupo está na metade de maior risco do estado".

### 1.3 Objetivos
- **Geral:** construir e comparar modelos de Inteligência Artificial capazes de classificar grupos de internação como de alto ou baixo risco de mortalidade hospitalar.
- **Específicos:** tratar os dados brutos do TABNET; analisar o comportamento das variáveis; selecionar atributos com critério metodológico; comparar seis algoritmos com validação cruzada e busca de hiperparâmetros; disponibilizar uma interface pública e interativa com os resultados.

---

## 2. Metodologia

A figura `docs/metodologia.drawio` (também em `docs/metodologia.svg`) resume as sete etapas descritas a seguir. Todo o pipeline é reprodutível com `python src/rodar_tudo.py`.

### 2.1 Base de dados
- **Fonte:** DATASUS, TABNET, SIH/SUS, "Morbidade Hospitalar do SUS por local de internação", Pernambuco.
- **Formato original:** 12 arquivos CSV exportados do TABNET, codificação latin1, separador `;`, três linhas de cabeçalho, rodapé com fonte e notas, números no formato brasileiro (`1.234,5`), valores ausentes como `-`. Nenhum arquivo estava pronto para Machine Learning.
- **Cruzamentos:** município × faixa etária, município × sexo, capítulo CID × faixa etária e estabelecimento × capítulo CID, cada um com três medidas (internações, média de permanência e taxa de mortalidade).
- **Dataset final:** 239 registros (capítulo CID × faixa etária) e 10 atributos: `doenca_cid`, `faixa_etaria`, `internacoes`, `media_permanencia`, `tx_mortalidade`, `media_perm_macro`, `tx_mort_macro`, `melhor_hospital`, `tx_mort_melhor_hosp` e o alvo `alto_risco`.

**Tratamento (`src/tratamento.py`):** leitura com remoção de cabeçalho e rodapé; conversão numérica; remoção do código IBGE dos municípios e do numeral romano dos capítulos; transformação de tabelas largas em formato longo (`melt`); união das três medidas por chave; agregação do contexto estadual por faixa etária (`media_perm_macro`, `tx_mort_macro`); identificação do hospital de referência (maior volume) por capítulo e sua taxa de mortalidade (`tx_mort_melhor_hosp`).

### 2.2 Análise dos dados (`src/analise.py`)
- **Ausentes:** no TABNET a taxa de mortalidade aparece como `-` quando não houve óbito; foi imputada com 0. Contexto hospitalar ausente recebeu a mediana da coluna. Após o tratamento não restaram ausentes. Duplicados: 0.
- **Estatísticas descritivas e outliers (regra 1,5×IQR):**

| Variável | Média | Desvio | Mín | Mediana | Máx | Assimetria | Outliers (1,5×IQR) |
|---|---|---|---|---|---|---|---|
| internacoes | 3017,00 | 5281,32 | 1,00 | 1246,00 | 55601,00 | 5,31 | 26 (10.88%) |
| media_permanencia | 4,84 | 2,82 | 0,10 | 4,30 | 16,00 | 1,06 | 7 (2.93%) |
| media_perm_macro | 4,46 | 1,16 | 3,05 | 4,58 | 6,50 | 0,26 | 0 (0.0%) |
| tx_mort_macro | 3,67 | 4,59 | 0,47 | 1,77 | 15,62 | 1,56 | 20 (8.37%) |
| tx_mort_melhor_hosp | 3,63 | 4,15 | 0,00 | 2,41 | 15,22 | 1,20 | 0 (0.0%) |
| tx_mortalidade | 2,93 | 5,41 | 0,00 | 0,90 | 40,49 | 3,66 | 25 (10.46%) |

- **Decisão sobre outliers:** mantidos. Os valores extremos (ex.: 55601 internações em um único grupo; mortalidade de 40,5%) são grupos reais e representam exatamente o fenômeno que se quer classificar. Removê-los esvaziaria a classe de alto risco. A assimetria de `internacoes` foi tratada com transformação logarítmica.
- **Relação com o alvo:** a proporção de alto risco cresce com a idade (de 10,5% na faixa de menor risco a 85,0% na de maior risco), com um pico também em menores de 1 ano. Por capítulo CID, o alto risco concentra-se em algumas doenças infecciosas e parasitárias, doenças do aparelho circulatório e neoplasias (tumores) (100,0%, 100,0% e 91,7%). Os gráficos de cada variável contra o alvo estão em `resultados/graficos/eda_*.png` e na Parte 3 da interface.

### 2.3 Seleção de variáveis (`src/selecao_variaveis.py`)
Três critérios independentes, calculados apenas no conjunto de treino: informação mútua com o alvo, correlação de Spearman (valor absoluto) com o alvo e importância por permutação em uma Random Forest de referência (queda de AUROC em 30 repetições). Uma variável entra no modelo se passar em **pelo menos dois dos três** critérios. Entre pares com colinearidade acima de 0,85, fica a de maior importância.

| Variável | Informação mútua | \|ρ\| Spearman | Importância (permutação) | Votos | Decisão |
|---|---|---|---|---|---|
| tx_mort_melhor_hosp | 0,1840 | 0,5676 | 0,0806 | 3/3 | selecionada |
| media_permanencia | 0,2179 | 0,5760 | 0,0408 | 3/3 | selecionada |
| tx_mort_macro | 0,1104 | 0,5370 | 0,0078 | 2/3 | selecionada |
| faixa_etaria_ord | 0,1834 | 0,4233 | 0,0070 | 2/3 | selecionada |
| media_perm_macro | 0,1232 | 0,4619 | 0,0045 | 2/3 | descartada |
| doenca_cod | 0,2268 | 0,0547 | 0,0019 | 1/3 | descartada |
| log_internacoes | 0,0198 | 0,2864 | 0,0019 | 1/3 | descartada |

Decisões adicionais: `melhor_hospital` foi descartada antes da análise (texto livre com um único valor por capítulo CID, redundante com `doenca_cid`); `tx_mortalidade` foi excluída por ser a origem do alvo (vazamento). `media_perm_macro` foi descartada por colinearidade com `tx_mort_macro` (ρ = 0,92); ambas derivam da faixa etária. `doenca_cod` e `log_internacoes` passaram em apenas um critério.

**Variáveis finais:** `tx_mort_melhor_hosp`, `media_permanencia`, `tx_mort_macro`, `faixa_etaria_ord`.

### 2.4 Pré-processamento (`src/preprocessamento.py`)
| Procedimento | O que foi feito | Justificativa |
|---|---|---|
| Limpeza | Remoção de cabeçalho/rodapé, totais e prefixos | Artefatos de exportação do TABNET |
| Ausentes | 0 para mortalidade (zero óbitos); mediana para contexto | Semântica do TABNET; mediana é robusta a outliers |
| Outliers | Mantidos | São os grupos de alto risco reais |
| Transformação | `log1p(internacoes)` | Cauda longa; reduz o peso dos extremos |
| Codificação | Faixa etária ordinal (0 a 11); capítulo CID one-hot | Idade tem ordem natural; doença é nominal |
| Padronização | `StandardScaler` dentro do Pipeline | KNN, SVM e Regressão Logística dependem de escala; ajustado só no treino de cada fold |
| Balanceamento | Não necessário (alvo pela mediana); `class_weight="balanced"` onde disponível | Classes 120 × 119 |
| Divisão | 75% treino (179) / 25% teste (60), estratificada, semente 42 | Teste nunca é usado para escolher nada |

### 2.5 Modelagem, validação cruzada e Grid Search (`src/treinar.py`)
Foram comparados seis algoritmos: Regressão Logística, K-Nearest Neighbors, Árvore de Decisão, Random Forest, SVM com kernel RBF e Gradient Boosting. Cada um foi encapsulado em um `Pipeline` do scikit-learn (pré-processamento + classificador) e otimizado com `GridSearchCV` usando `StratifiedKFold` com 5 dobras, otimizando **AUROC**. Ao todo, 157 combinações de hiperparâmetros foram avaliadas.

| Modelo | Grade testada (combinações) | Melhores hiperparâmetros |
|---|---|---|
| Regressão Logística | 5 | C=0.01, penalty=l2, solver=lbfgs |
| KNN | 24 | n_neighbors=15, p=1, weights=distance |
| Árvore de Decisão | 48 | criterion=gini, max_depth=6, min_samples_leaf=5 |
| Random Forest | 48 | max_depth=None, max_features=sqrt, min_samples_leaf=1, n_estimators=200 |
| SVM (RBF) | 16 | C=100, gamma=0.01, kernel=rbf |
| Gradient Boosting | 16 | learning_rate=0.03, max_depth=2, n_estimators=100, subsample=0.8 |

**Por que AUROC como métrica do Grid Search:** é independente do limiar de decisão e mede a capacidade de ordenar grupos por risco, que é o uso prático do modelo. **Por que validação estratificada:** garante a mesma proporção de classes em cada dobra, com 179 registros de treino isso evita dobras desbalanceadas por acaso. **Critério de escolha do modelo final:** maior AUROC médio na validação cruzada do treino (e não no teste), para não escolher o modelo "olhando a prova".

### 2.6 Avaliação
No conjunto de teste (60 registros, nunca vistos), cada modelo foi avaliado com AUROC, F1-score, sensibilidade (recall da classe Alto risco), especificidade, precisão, acurácia e matriz de confusão. Curvas ROC e Precisão-Recall foram geradas para todos.

---

## 3. Resultados

### 3.1 Desempenho dos modelos

| Modelo | AUROC CV (média ± dp) | AUROC | F1 | Acurácia | Sensibilidade | Especificidade | Precisão |
|---|---|---|---|---|---|---|---|
| Regressão Logística | 0,935 ± 0,04 | 0,888 | 0,820 | 0,817 | 0,833 | 0,800 | 0,806 |
| KNN | 0,953 ± 0,03 | 0,929 | 0,839 | 0,833 | 0,867 | 0,800 | 0,812 |
| Árvore de Decisão | 0,914 ± 0,04 | 0,883 | 0,788 | 0,767 | 0,867 | 0,667 | 0,722 |
| Random Forest **(melhor)** | 0,977 ± 0,02 | 0,951 | 0,900 | 0,900 | 0,900 | 0,900 | 0,900 |
| SVM (RBF) | 0,958 ± 0,03 | 0,953 | 0,903 | 0,900 | 0,933 | 0,867 | 0,875 |
| Gradient Boosting | 0,975 ± 0,03 | 0,956 | 0,951 | 0,950 | 0,967 | 0,933 | 0,935 |

**Matrizes de confusão (teste):**

| Modelo | VN | FP | FN | VP |
|---|---|---|---|---|
| Regressão Logística | 24 | 6 | 5 | 25 |
| KNN | 24 | 6 | 4 | 26 |
| Árvore de Decisão | 20 | 10 | 4 | 26 |
| Random Forest | 27 | 3 | 3 | 27 |
| SVM (RBF) | 26 | 4 | 2 | 28 |
| Gradient Boosting | 28 | 2 | 1 | 29 |

### 3.2 Discussão
- O **Random Forest** foi o modelo escolhido, com AUROC de 0,977 ± 0,02 na validação cruzada e 0,951 no teste. Classificou corretamente 90,0% dos grupos, com sensibilidade de 90,0% e especificidade de 90,0%: erra pouco nos dois sentidos.
- Os modelos de conjunto (Random Forest e Gradient Boosting) e o SVM ficaram próximos entre si e acima dos modelos mais simples. A Árvore de Decisão isolada teve o pior desempenho (0,883 de AUROC), o que confirma o ganho de variância dos métodos de ensemble.
- A Regressão Logística, mesmo sendo linear, alcançou AUROC de 0,888: a relação entre as variáveis e o alvo é em boa parte monotônica.
- AUROC de validação cruzada e de teste ficaram próximas em todos os modelos: não há sinal de sobreajuste relevante.
- **Importância das variáveis no Random Forest** (queda de AUROC ao permutar): `tx_mort_melhor_hosp` 0,196; `media_permanencia` 0,076; `tx_mort_macro` 0,035; `faixa_etaria_ord` 0,013. A taxa de mortalidade do hospital de referência da doença é o fator mais informativo, seguida do tempo médio de permanência do próprio grupo.

### 3.3 Limitações
- A base tem 239 grupos; o conjunto de teste tem 60. Cada erro a mais muda a acurácia em cerca de 1,7 pontos percentuais, então as diferenças entre os três melhores modelos não são estatisticamente conclusivas.
- O alvo é um corte estatístico (mediana) sobre grupos agregados, não um desfecho clínico individual. O modelo classifica grupos, não pacientes, e serve como apoio à gestão.
- As variáveis de contexto (`tx_mort_macro`, `tx_mort_melhor_hosp`) são agregações do próprio período; em produção precisariam ser recalculadas periodicamente.

### 3.4 Interface
A interface (`frontend/index.html`) é um único arquivo HTML, organizado nas quatro partes pedidas: (1) problema, contexto, objetivo, base e variável-alvo; (2) metodologia com fluxo visual, tabela de seleção e justificativas de pré-processamento; (3) análise gráfica interativa de cada variável contra o alvo; (4) resultados dos modelos com tabela comparativa, métricas, curvas ROC, matrizes de confusão, importâncias e hiperparâmetros. Inclui ainda um simulador que consulta a API Flask (`api.py`) para classificar novos grupos.

---

## 4. Referências

BRASIL. Ministério da Saúde. DATASUS. **Sistema de Informações Hospitalares do SUS (SIH/SUS)**: Morbidade Hospitalar do SUS por local de internação, Pernambuco. TABNET. Disponível em: https://datasus.saude.gov.br/informacoes-de-saude-tabnet/. Acesso em: nov. 2026.

BREIMAN, L. Random Forests. **Machine Learning**, v. 45, n. 1, p. 5-32, 2001.

CORTES, C.; VAPNIK, V. Support-vector networks. **Machine Learning**, v. 20, p. 273-297, 1995.

FRIEDMAN, J. H. Greedy function approximation: a gradient boosting machine. **Annals of Statistics**, v. 29, n. 5, p. 1189-1232, 2001.

HAN, J.; KAMBER, M.; PEI, J. **Data Mining: Concepts and Techniques**. 3. ed. Waltham: Morgan Kaufmann, 2011.

HOSMER, D. W.; LEMESHOW, S.; STURDIVANT, R. X. **Applied Logistic Regression**. 3. ed. Hoboken: Wiley, 2013.

KOHAVI, R. A study of cross-validation and bootstrap for accuracy estimation and model selection. In: **IJCAI**, 1995. p. 1137-1143.

ORGANIZAÇÃO MUNDIAL DA SAÚDE. **CID-10: Classificação Estatística Internacional de Doenças e Problemas Relacionados à Saúde**. 10. rev. São Paulo: EDUSP, 2008.

PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. **Journal of Machine Learning Research**, v. 12, p. 2825-2830, 2011.

TUKEY, J. W. **Exploratory Data Analysis**. Reading: Addison-Wesley, 1977.
