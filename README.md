# TSPE v2 · Classificação de risco de mortalidade hospitalar em Pernambuco

> Projeto da disciplina WYD6399 · Tópicos em Big Data em Python · UniFavip Wyden · Prof. Me. Flávio Morais
> Equipe: Julio César Batista Silva · Teves Gabriel · Luciano Filho · Jonathas Costa · Anderson Davi

Um classificador de Machine Learning treinado com dados públicos do SIH/SUS (DATASUS, TABNET) que responde, para cada combinação de capítulo CID-10 e faixa etária, se a **taxa de mortalidade hospitalar** do grupo fica **acima (Alto risco)** ou **abaixo (Baixo risco)** da mediana do estado.

Evolução do projeto [TSPE](https://github.com/lucianot389-dotcom/TSPE): mesma base e mesma lógica de tratamento, agora com problema de classificação formal, seleção de variáveis justificada, seis modelos com validação cruzada e Grid Search, e uma interface nova em quatro partes.

## Como rodar em 3 comandos

```bash
pip install -r requirements.txt
python src/rodar_tudo.py        # tratamento → análise → seleção → modelos → interface → docs (~1,5 min)
python api.py                   # sobe a API em http://localhost:5000
```

Depois abra `frontend/index.html` no navegador (duplo clique). Tudo funciona offline; só o simulador precisa da API.

Sem Python instalado? Use o notebook `notebooks/TSPE_v2_pipeline_colab.ipynb` no Google Colab.

## Estrutura

```
tspe-v2/
├── api.py                      # API Flask: carrega o modelo treinado e responde /prever
├── requirements.txt
├── Procfile                    # deploy da API (Railway / Render / Heroku)
├── data/
│   ├── raw/                    # 12 CSVs do TABNET (ver data/raw/README.md)
│   └── processado/             # tabelas longas + dataset_ml.csv (gerados)
├── src/
│   ├── config.py               # caminhos, nomes de arquivo, constantes, alvo, semente
│   ├── tratamento.py           # etapa 1: limpeza do TABNET e dataset final (lógica do notebook original)
│   ├── analise.py              # etapa 2: EDA, ausentes, outliers, gráficos variável × alvo
│   ├── selecao_variaveis.py    # etapa 3: info. mútua + Spearman + permutação, 2 de 3
│   ├── preprocessamento.py     # codificação, log, escala (dentro do Pipeline)
│   ├── treinar.py              # etapa 4: 6 modelos, StratifiedKFold(5), GridSearchCV, métricas
│   ├── build_frontend.py       # etapa 5: injeta os JSONs no template → frontend/index.html
│   ├── gerar_drawio.py         # figura da metodologia (.drawio + .svg)
│   ├── gerar_documentacao.py   # docs/Documentacao.md com os números reais
│   └── rodar_tudo.py           # roda tudo na ordem
├── frontend/
│   ├── template.html           # interface (4 partes + simulador), sem dados
│   ├── chart.umd.js            # Chart.js embutido (funciona offline)
│   └── index.html              # interface final gerada
├── resultados/                 # JSONs e gráficos PNG gerados
├── modelos/                    # melhor_modelo.joblib + metadados.json
├── docs/                       # Documentacao.md, metodologia.drawio/.svg, GUIA_PASSO_A_PASSO.md
└── notebooks/                  # versão para Google Colab
```

## Resultado atual

| Modelo | AUROC CV | AUROC teste | F1 | Acurácia |
|---|---|---|---|---|
| Random Forest **(escolhido)** | 0,977 | 0,951 | 0,900 | 0,900 |
| Gradient Boosting | 0,975 | 0,956 | 0,951 | 0,950 |
| SVM (RBF) | 0,958 | 0,953 | 0,903 | 0,900 |
| KNN | 0,953 | 0,929 | 0,839 | 0,833 |
| Regressão Logística | 0,935 | 0,888 | 0,820 | 0,817 |
| Árvore de Decisão | 0,914 | 0,883 | 0,788 | 0,767 |

O modelo final é escolhido pelo AUROC da validação cruzada, nunca pelo teste. Detalhes, justificativas e limitações em `docs/Documentacao.md`.

## API

| Rota | Método | Descrição |
|---|---|---|
| `/saude` | GET | status e nome do modelo |
| `/opcoes` | GET | listas de doenças e faixas etárias |
| `/prever` | POST | `{"doenca": "...", "faixa_etaria": "...", "media_permanencia": 8}` |
| `/resultados`, `/analise`, `/selecao` | GET | JSONs do pipeline |

```bash
curl -X POST http://localhost:5000/prever -H "Content-Type: application/json" \
  -d '{"doenca":"Neoplasias (tumores)","faixa_etaria":"70 a 79 anos","media_permanencia":8}'
```

## Deploy

- **API:** Railway/Render com o `Procfile` incluído. Depois gere a interface apontando para a URL: `python src/build_frontend.py https://sua-api.up.railway.app`.
- **Interface:** `frontend/index.html` é estático; serve na Vercel, GitHub Pages ou Netlify.

## Fonte dos dados

BRASIL. Ministério da Saúde. DATASUS. SIH/SUS, Morbidade Hospitalar do SUS por local de internação, Pernambuco. TABNET: https://datasus.saude.gov.br/informacoes-de-saude-tabnet/
