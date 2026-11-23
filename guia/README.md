# Guia das pastas

O que cada pasta do projeto faz, na ordem em que o trabalho acontece.

## 1. `data/`: os dados

| Pasta | Responsável por |
|---|---|
| `data/raw/` | Guardar os 12 arquivos CSV baixados do TABNET (DATASUS), do jeito que vieram, sem mexer. O `README.md` dela lista o nome de cada arquivo. |
| `data/processado/` | Guardar a base já tratada. O arquivo principal é `dataset_ml.csv`: 239 linhas (doença × faixa etária) e 10 colunas. |

## 2. `src/`: o código

Cada arquivo faz uma etapa. Rodando `python src/rodar_tudo.py`, todas rodam em sequência.

| Arquivo | Responsável por |
|---|---|
| `config.py` | Nomes de arquivos, pastas, faixas etárias, definição do alvo e semente aleatória. Tudo que é configuração fica aqui. |
| `tratamento.py` | Ler os CSVs do TABNET, limpar (cabeçalho, rodapé, números com vírgula, códigos), juntar as tabelas e criar o alvo `alto_risco`. |
| `analise.py` | Análise exploratória: dados ausentes, outliers, correlação e os gráficos de cada variável contra o alvo. |
| `selecao_variaveis.py` | Escolher as variáveis que entram no modelo (informação mútua, Spearman e importância por permutação). |
| `preprocessamento.py` | Preparar as variáveis para o modelo: faixa etária em número, padronização da escala. |
| `treinar.py` | Treinar os 6 modelos com validação cruzada e Grid Search, medir as métricas e salvar o melhor. |
| `ablacao.py` | Treinar de novo sem as variáveis de contexto, para mostrar que o resultado não depende delas. |
| `build_frontend.py` | Montar a página de análise em 4 partes (`frontend/index.html`). |
| `gerar_home.py` | Montar a página da apresentação (`index.html` na raiz): consulta da taxa de mortalidade e principais doenças. |
| `gerar_drawio.py` | Gerar a figura da metodologia (`docs/metodologia.drawio`). |
| `gerar_documentacao.py` e `gerar_guia_html.py` | Gerar a documentação e o guia em HTML com os números atualizados. |
| `rodar_tudo.py` | Rodar todas as etapas acima, na ordem. |

## 3. `modelos/`: o modelo pronto

| Arquivo | Responsável por |
|---|---|
| `melhor_modelo.joblib` | O Random Forest já treinado. É ele que classifica os grupos. |
| `metadados.json` | Informações que o modelo precisa para receber uma consulta: lista de doenças, faixas etárias, corte do alvo. |

## 4. `resultados/`: os números

| Pasta ou arquivo | Responsável por |
|---|---|
| `resultados_modelos.json` | Métricas de cada modelo (AUROC, F1, acurácia, sensibilidade, especificidade, matriz de confusão). |
| `selecao_variaveis.json` | Notas de cada variável e o motivo de ter entrado ou saído. |
| `analise.json` e `log_tratamento.json` | Números da análise exploratória e o registro do tratamento. |
| `ablacao.json` | Resultado do teste sem as variáveis de contexto. |
| `graficos/` | Os gráficos em PNG usados na documentação e nos slides. |

## 5. Páginas

| Arquivo | Responsável por |
|---|---|
| `index.html` (raiz) | Página da apresentação: consulta da taxa de mortalidade e principais doenças com tratamentos. É o que abre em https://juliobatista04.github.io/TSPE-v2/ |
| `frontend/index.html` | Análise completa em 4 partes, exigida na diretriz (problema, metodologia, gráficos, resultados). |
| `frontend/template.html` | Modelo da página de análise, sem os dados. Para mudar texto ou cor, edite aqui. |

## 6. `docs/`: documentação

| Arquivo | Responsável por |
|---|---|
| `Documentacao.md` | O texto do trabalho: introdução, metodologia, resultados e referências. |
| `GUIA_PASSO_A_PASSO.md` e `Guia_TSPE_v2.html` | Como instalar, rodar e publicar, e respostas para as perguntas da banca. |
| `metodologia.drawio` e `metodologia.png` | Figura das 7 etapas do trabalho. O `.drawio` abre no diagrams.net. O `.png` vai no slide. |

## 7. Outros

| Arquivo ou pasta | Responsável por |
|---|---|
| `api.py` | API Flask que recebe doença, faixa etária e dias de internação e devolve a classificação. |
| `notebooks/` | Versão do projeto para rodar no Google Colab. |
| `requirements.txt` | Lista das bibliotecas Python necessárias. |
| `Procfile` | Configuração para publicar a API no Railway ou no Render. |
| `guia/` | Esta pasta. |
