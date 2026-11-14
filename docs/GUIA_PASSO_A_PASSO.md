# Guia passo a passo · TSPE v2

Como colocar o projeto para rodar, do zero, e o que cada peça faz. Pensado para qualquer integrante da equipe conseguir reproduzir e defender o trabalho na apresentação de 30/11/2026.

## 0. O que você vai ter no fim

| Entrega da diretriz | Onde está |
|---|---|
| Base pública, não preparada para ML (item 3) | `data/raw/` (TABNET) → `data/processado/dataset_ml.csv` |
| Análise dos dados (item 5) | `src/analise.py` → `resultados/analise.json` + `resultados/graficos/eda_*.png` |
| Seleção de variáveis justificada (item 6) | `src/selecao_variaveis.py` → `resultados/selecao_variaveis.json` |
| Gráfico de cada variável × alvo (item 7) | Parte 3 da interface + `eda_02..07.png` |
| Pré-processamento justificado (item 8) | `src/preprocessamento.py` + tabela na Documentação §2.4 |
| CV + Grid Search + métrica (item 9) | `src/treinar.py` → `resultados/resultados_modelos.json` |
| Métricas e matriz de confusão (item 10) | Parte 4 da interface + `modelo_*.png` |
| Interface em 4 partes (item 11) | `frontend/index.html` |
| Figura da metodologia em Draw.io (item 11) | `docs/metodologia.drawio` |
| Documentação (item 12) | `docs/Documentacao.md` |

## 1. Pré-requisitos

- Python 3.10 ou superior (`python --version`)
- Git
- Um navegador

Sem Python na máquina? Pule para a seção 7 (Colab).

## 2. Baixar e instalar

```bash
git clone https://github.com/juliobatista04/TSPE-v2.git
cd TSPE-v2
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## 3. (Opcional, recomendado) Colocar os CSVs brutos do TABNET

O pipeline funciona sem eles (usa `data/processado/dataset_ml_legado.csv`, o mesmo dataset do projeto anterior). Mas com os 12 CSVs originais em `data/raw/` o tratamento roda do zero e a interface mostra a contagem real de dados ausentes, o que fortalece a defesa do item 3 da diretriz ("base não preparada").

Os nomes esperados estão em `data/raw/README.md`. São os mesmos do notebook `tratamento_dados_ML_2.ipynb`; basta copiar a pasta `data/raw` do Colab antigo.

## 4. Rodar o pipeline

```bash
python src/rodar_tudo.py
```

Leva cerca de 1 a 2 minutos. O que acontece, na ordem:

1. **Tratamento** (`tratamento.py`): lê os CSVs do TABNET, limpa cabeçalho/rodapé, converte números, derrete tabelas largas, une internações + permanência + mortalidade, cria o alvo `alto_risco` (1 se taxa de mortalidade > mediana). Grava `data/processado/dataset_ml.csv` e `resultados/log_tratamento.json`.
2. **Análise** (`analise.py`): estatísticas descritivas, ausentes, duplicados, outliers (1,5×IQR), correlação de Spearman, séries para os gráficos. Grava `resultados/analise.json` e 7 PNGs.
3. **Seleção de variáveis** (`selecao_variaveis.py`): informação mútua, Spearman e importância por permutação, só no treino. Entra quem passa em 2 de 3; entre colineares fica a mais importante. Grava `resultados/selecao_variaveis.json` com a justificativa de cada variável.
4. **Treino** (`treinar.py`): seis modelos em Pipeline, `GridSearchCV` com `StratifiedKFold(5)` otimizando AUROC, avaliação no teste (25%). Grava `resultados/resultados_modelos.json`, 5 PNGs, `modelos/melhor_modelo.joblib` e `modelos/metadados.json`.
5. **Interface** (`build_frontend.py`): injeta os JSONs no `frontend/template.html` e gera `frontend/index.html` (um arquivo só, com Chart.js embutido).
6. **Figura** (`gerar_drawio.py`): `docs/metodologia.drawio` e `.svg` com os números reais.
7. **Documentação** (`gerar_documentacao.py`): `docs/Documentacao.md`.

Quer rodar só uma etapa? `python src/analise.py`, por exemplo. Cada script é independente desde que os anteriores já tenham rodado.

## 5. Subir a API e abrir a interface

```bash
python api.py
```

Abra `frontend/index.html` no navegador. Vá até **Simulador**, escolha uma doença, uma faixa etária e um tempo de permanência e clique em **Classificar**. O campo "Endereço da API" vem com `http://localhost:5000`; se a API estiver publicada, troque pela URL.

Teste rápido pelo terminal:

```bash
curl -X POST http://localhost:5000/prever -H "Content-Type: application/json" \
  -d '{"doenca":"Neoplasias (tumores)","faixa_etaria":"70 a 79 anos","media_permanencia":8}'
```

## 6. Publicar (para o público acessar na apresentação)

**API (Railway, igual ao projeto anterior):**
1. Novo projeto → Deploy from GitHub → selecione o repositório.
2. O `Procfile` já está pronto (`gunicorn api:app`). Railway detecta o Python e instala o `requirements.txt`.
3. Copie a URL pública (ex.: `https://tspe-v2.up.railway.app`) e teste `/saude`.

**Interface:**
1. Gere o HTML apontando para a API publicada: `python src/build_frontend.py https://tspe-v2.up.railway.app`
2. Faça commit do `frontend/index.html` e publique a pasta `frontend/` na Vercel, Netlify ou GitHub Pages (Settings → Pages → pasta `/frontend`).

Importante: `modelos/melhor_modelo.joblib` precisa estar no repositório (está, por padrão). A API não treina nada ao subir, só carrega o modelo. Por isso ela sobe em menos de 1 segundo.

## 7. Alternativa: tudo no Google Colab

Abra `notebooks/TSPE_v2_pipeline_colab.ipynb` no Colab (Arquivo → Abrir notebook → GitHub) e rode as células em ordem. Ele clona o repositório, instala as dependências, roda o pipeline, mostra os gráficos, abre a interface dentro do Colab e deixa baixar um zip com tudo.

## 8. Abrir e editar a figura da metodologia

1. Acesse https://app.diagrams.net
2. Arquivo → Abrir de → Dispositivo → `docs/metodologia.drawio`
3. Edite o que quiser e exporte em PNG (Arquivo → Exportar como → PNG) para colar na documentação e na apresentação.

A figura já sai com os números reais (registros, atributos, combinações do Grid Search, melhor modelo). Se rodar o pipeline de novo, ela é regenerada.

## 9. Como defender cada decisão (para as perguntas do professor)

| Pergunta provável | Resposta curta |
|---|---|
| Por que esse alvo? | Mediana da taxa de mortalidade: reprodutível, sem arbitrariedade clínica, classes 50/50. |
| A base estava pronta para ML? | Não. 12 CSVs do TABNET com cabeçalho, rodapé, números no formato brasileiro, "-" como ausente, tabelas largas. Tudo tratado em `tratamento.py`. |
| Como escolheram as variáveis? | Três critérios independentes (informação mútua, Spearman, permutação); entra com 2 de 3. `media_perm_macro` caiu por colinearidade (ρ 0,92) com `tx_mort_macro`. |
| Por que não removeram outliers? | São grupos reais de alto risco (neoplasias em idosos, por exemplo). Removê-los apagaria a classe que queremos prever. Tratamos a assimetria com log. |
| Houve vazamento? | Não: `tx_mortalidade` (origem do alvo) ficou fora; escala e one-hot são ajustados dentro do Pipeline, só no treino de cada fold; teste é tocado uma vez. |
| Por que AUROC no Grid Search? | Independe do limiar e mede a capacidade de ordenar grupos por risco, que é o uso prático. |
| Por que Random Forest e não Gradient Boosting, que teve teste melhor? | Escolhemos pela validação cruzada (0,977 vs 0,975), não pelo teste. Com 60 registros de teste, a diferença de 1 acerto não é conclusiva. |
| Qual a limitação? | 239 grupos; classifica grupos, não pacientes; apoio à gestão, não decisão clínica. |

## 10. Mudar coisas sem quebrar

- **Trocar o corte do alvo** (ex.: tercil superior): edite `finalizar_dataset()` em `src/tratamento.py` e rode tudo de novo.
- **Adicionar um modelo:** inclua uma entrada em `catalogo_modelos()` em `src/treinar.py` com pipeline, grade e justificativa. A interface e a documentação se adaptam sozinhas.
- **Mudar cores/textos da interface:** edite `frontend/template.html` (tokens de cor no topo do CSS) e rode `python src/build_frontend.py`.
- **Renomear os CSVs do TABNET:** edite `ARQUIVOS_RAW` em `src/config.py`.
