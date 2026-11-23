"""
Gera docs/Documentacao.md (diretriz 12) a partir dos JSONs do pipeline.
Assim o texto nunca fica desatualizado em relação aos números.

Uso: python src/gerar_documentacao.py
"""
import json
from datetime import date

import config as C

R = json.load(open(C.DIR_RESULTADOS / "resultados_modelos.json", encoding="utf-8"))
A = json.load(open(C.DIR_RESULTADOS / "analise.json", encoding="utf-8"))
S = json.load(open(C.DIR_RESULTADOS / "selecao_variaveis.json", encoding="utf-8"))
T = json.load(open(C.DIR_RESULTADOS / "log_tratamento.json", encoding="utf-8"))

M = R["modelos"]; melhor = R["melhor_modelo"]; mb = M[melhor]["teste"]
res = A["resumo"]; out = A["outliers"]
f = lambda v, d=3: f"{v:.{d}f}".replace(".", ",")
pc = lambda v: f"{100*v:.1f}%".replace(".", ",")

def tabela_modelos():
    linhas = ["| Modelo | AUROC CV (média ± dp) | AUROC | F1 | Acurácia | Sensibilidade | Especificidade | Precisão |", "|---|---|---|---|---|---|---|---|"]
    for n, m in M.items():
        t = m["teste"]; tag = " **(melhor)**" if n == melhor else ""
        linhas.append(f"| {n}{tag} | {f(m['cv_auroc_media'])} ± {f(m['cv_auroc_dp'],2)} | {f(t['auroc'])} | {f(t['f1'])} | {f(t['acuracia'])} | {f(t['sensibilidade'])} | {f(t['especificidade'])} | {f(t['precisao'])} |")
    return "\n".join(linhas)

def tabela_cm():
    linhas = ["| Modelo | VN | FP | FN | VP |", "|---|---|---|---|---|"]
    for n, m in M.items():
        c = m["teste"]["matriz_confusao"]
        linhas.append(f"| {n} | {c['tn']} | {c['fp']} | {c['fn']} | {c['tp']} |")
    return "\n".join(linhas)

def tabela_hp():
    linhas = ["| Modelo | Grade testada (combinações) | Melhores hiperparâmetros |", "|---|---|---|"]
    for n, m in M.items():
        hp = ", ".join(f"{k}={v}" for k, v in m["melhores_hiperparametros"].items())
        linhas.append(f"| {n} | {m['combinacoes_testadas']} | {hp} |")
    return "\n".join(linhas)

def tabela_sel():
    linhas = ["| Variável | Informação mútua | \\|ρ\\| Spearman | Importância (permutação) | Votos | Decisão |", "|---|---|---|---|---|---|"]
    for r in S["tabela"]:
        linhas.append(f"| {r['variavel']} | {f(r['info_mutua'],4)} | {f(r['spearman_abs'],4)} | {f(r['importancia_permutacao'],4)} | {r['votos']}/3 | {'selecionada' if r['selecionada'] else 'descartada'} |")
    return "\n".join(linhas)

def tabela_desc():
    linhas = ["| Variável | Média | Desvio | Mín | Mediana | Máx | Assimetria | Outliers (1,5×IQR) |", "|---|---|---|---|---|---|---|---|"]
    for v, d in res["descritivas"].items():
        linhas.append(f"| {v} | {f(d['mean'],2)} | {f(d['std'],2)} | {f(d['min'],2)} | {f(d['50%'],2)} | {f(d['max'],2)} | {f(d['assimetria'],2)} | {out[v]['quantidade']} ({str(out[v]['percentual']).replace('.', ',')}%) |")
    return "\n".join(linhas)

def bloco_ablacao():
    arq = C.DIR_RESULTADOS / "ablacao.json"
    if not arq.exists():
        return ""
    ab = json.load(open(arq, encoding="utf-8"))
    linhas = ["", "**Teste de sensibilidade (ablação).** Para medir quanto o desempenho depende das variáveis de contexto, os modelos foram treinados de novo, com o mesmo split, a mesma validação e Grid Search, em quatro cenários (`src/ablacao.py`):", "",
              "| Cenário | Variáveis | RF AUROC CV | RF AUROC teste | GB AUROC CV | GB AUROC teste |", "|---|---|---|---|---|---|"]
    for c in ab:
        m = c["modelos"]; neg = "**" if c["cenario"].startswith("Sem contexto +") else ""
        linhas.append(f"| {neg}{c['cenario']}{neg} | {neg}{c['descricao']}{neg} | {neg}{f(m['RF']['auroc_cv'])}{neg} | {neg}{f(m['RF']['auroc_teste'])}{neg} | {neg}{f(m['GB']['auroc_cv'])}{neg} | {neg}{f(m['GB']['auroc_teste'])}{neg} |")
    c = next(x for x in ab if x["cenario"].startswith("Sem contexto +"))["modelos"]
    linhas += ["", f"Leitura: retirar as variáveis de contexto derruba o desempenho só quando a informação da doença some junto. Com o capítulo CID codificado diretamente, sem nenhuma taxa de mortalidade como entrada, o Random Forest volta a AUROC de {f(c['RF']['auroc_cv'])} na validação cruzada e {f(c['RF']['auroc_teste'])} no teste. Ou seja, `tx_mort_melhor_hosp` funciona como um resumo da doença (um valor por capítulo), e o poder preditivo vem da combinação doença × idade × permanência, não de vazamento do alvo.", ""]
    return "\n".join(linhas)


faixa = A["series"]["alto_risco_por_faixa"]; doen = A["series"]["alto_risco_por_doenca"]
imp = R["importancias_melhor_modelo"]
comb = sum(m["combinacoes_testadas"] for m in M.values())

doc = f"""# Classificação de risco de mortalidade hospitalar em Pernambuco com Machine Learning

**Disciplina:** WYD6399 · Tópicos Avançados em Big Data com Python · UniFavip Wyden
**Professor:** Melquezedeque Lima
**Equipe:** Julio César Batista Silva · Teves Gabriel · Luciano Filho · Jonathas Costa · Anderson Davi
**Data:** {date.today().strftime('%d/%m/%Y')} · Apresentação: 30/11/2026

---

## 1. Introdução

### 1.1 Contexto e relevância
O Sistema Único de Saúde registra todas as internações financiadas com recurso público no Sistema de Informações Hospitalares (SIH/SUS). Em Pernambuco, essas internações se distribuem por 164 municípios e por 20 capítulos da CID-10. A taxa de mortalidade hospitalar (óbitos a cada 100 internações) varia de 0% em grupos de baixo risco até {f(res['descritivas']['tx_mortalidade']['max'],1)}% nos grupos mais críticos. Para quem planeja a rede de atenção, saber com antecedência quais combinações de doença e idade concentram o risco ajuda a alocar leitos, equipes e protocolos clínicos.

### 1.2 Problema de classificação
O projeto trata o problema como uma **tarefa de classificação binária supervisionada**. Cada registro é um grupo formado por um capítulo da CID-10 e uma faixa etária. A variável-alvo é:

- **alto_risco = 1 (Alto risco):** taxa de mortalidade hospitalar do grupo acima da mediana estadual ({f(T['corte_alvo'],2)}%);
- **alto_risco = 0 (Baixo risco):** taxa igual ou abaixo da mediana.

O corte pela mediana foi escolhido por três razões: (i) é um critério reprodutível e sem arbitrariedade clínica; (ii) produz classes equilibradas ({res['distribuicao_alvo']['Baixo risco']} × {res['distribuicao_alvo']['Alto risco']}), o que dispensa técnicas de balanceamento artificial; (iii) permite que o resultado seja lido como "este grupo está na metade de maior risco do estado".

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
- **Dataset final:** {res['registros']} registros (capítulo CID × faixa etária) e {res['atributos']} atributos: `doenca_cid`, `faixa_etaria`, `internacoes`, `media_permanencia`, `tx_mortalidade`, `media_perm_macro`, `tx_mort_macro`, `melhor_hospital`, `tx_mort_melhor_hosp` e o alvo `alto_risco`.

**Tratamento (`src/tratamento.py`):** leitura com remoção de cabeçalho e rodapé; conversão numérica; remoção do código IBGE dos municípios e do numeral romano dos capítulos; transformação de tabelas largas em formato longo (`melt`); união das três medidas por chave; agregação do contexto estadual por faixa etária (`media_perm_macro`, `tx_mort_macro`); identificação do hospital de referência (maior volume) por capítulo e sua taxa de mortalidade (`tx_mort_melhor_hosp`).

### 2.2 Análise dos dados (`src/analise.py`)
- **Ausentes:** no TABNET a taxa de mortalidade aparece como `-` quando não houve óbito; foi imputada com 0. Contexto hospitalar ausente recebeu a mediana da coluna. Após o tratamento não restaram ausentes. Duplicados: {res['duplicados']}.
- **Estatísticas descritivas e outliers (regra 1,5×IQR):**

{tabela_desc()}

- **Decisão sobre outliers:** mantidos. Os valores extremos (ex.: {f(res['descritivas']['internacoes']['max'],0)} internações em um único grupo; mortalidade de {f(res['descritivas']['tx_mortalidade']['max'],1)}%) são grupos reais e representam exatamente o fenômeno que se quer classificar. Removê-los esvaziaria a classe de alto risco. A assimetria de `internacoes` foi tratada com transformação logarítmica.
- **Relação com o alvo:** a proporção de alto risco cresce com a idade (de {pc(min(faixa['proporcao']))} na faixa de menor risco a {pc(max(faixa['proporcao']))} na de maior risco), com um pico também em menores de 1 ano. Por capítulo CID, o alto risco concentra-se em {doen['labels'][0].lower()}, {doen['labels'][1].lower()} e {doen['labels'][2].lower()} ({pc(doen['proporcao'][0])}, {pc(doen['proporcao'][1])} e {pc(doen['proporcao'][2])}). Os gráficos de cada variável contra o alvo estão em `resultados/graficos/eda_*.png` e na Parte 3 da interface.

### 2.3 Seleção de variáveis (`src/selecao_variaveis.py`)
Três critérios independentes, calculados apenas no conjunto de treino: informação mútua com o alvo, correlação de Spearman (valor absoluto) com o alvo e importância por permutação em uma Random Forest de referência (queda de AUROC em 30 repetições). Uma variável entra no modelo se passar em **pelo menos dois dos três** critérios. Entre pares com colinearidade acima de 0,85, fica a de maior importância.

{tabela_sel()}

Decisões adicionais: `melhor_hospital` foi descartada antes da análise (texto livre com um único valor por capítulo CID, redundante com `doenca_cid`); `tx_mortalidade` foi excluída por ser a origem do alvo (vazamento). `media_perm_macro` foi descartada por colinearidade com `tx_mort_macro` (ρ = 0,92); ambas derivam da faixa etária. `doenca_cod` e `log_internacoes` passaram em apenas um critério.

**Variáveis finais:** {", ".join(f"`{v}`" for v in S["selecionadas"])}.

**Variáveis de contexto e risco de vazamento:** `tx_mort_macro` e `tx_mort_melhor_hosp` vêm da mesma fonte de mortalidade que origina o alvo, mas em outro nível de agregação. A primeira é a taxa estadual da faixa etária, somando todas as doenças. A segunda é a taxa do hospital de maior volume no capítulo CID, somando todas as idades. Nenhuma delas é a taxa do próprio grupo (`tx_mortalidade`), que ficou fora do modelo. Elas representam informação que o gestor já tem antes de avaliar um grupo: o risco típico daquela idade e o desempenho do hospital de referência daquela doença. Como foram calculadas no mesmo período do alvo, essa limitação é discutida e testada na seção 3.3.

### 2.4 Pré-processamento (`src/preprocessamento.py`)
| Procedimento | O que foi feito | Justificativa |
|---|---|---|
| Limpeza | Remoção de cabeçalho/rodapé, totais e prefixos | Artefatos de exportação do TABNET |
| Ausentes | 0 para mortalidade (zero óbitos); mediana para contexto | Semântica do TABNET; mediana é robusta a outliers |
| Outliers | Mantidos | São os grupos de alto risco reais |
| Transformação | `log1p(internacoes)`, gerando a candidata `log_internacoes` | Cauda longa; reduz o peso dos extremos. A variável foi avaliada na seleção e descartada (1/3 critérios) |
| Codificação | Faixa etária ordinal (0 a 11); capítulo CID codificado (`doenca_cod`) para a etapa de seleção | Idade tem ordem natural. O capítulo CID foi descartado na seleção (1/3 critérios); a doença entra no modelo de forma indireta, pela `tx_mort_melhor_hosp` |
| Padronização | `StandardScaler` dentro do Pipeline | KNN, SVM e Regressão Logística dependem de escala; ajustado só no treino de cada fold |
| Balanceamento | Não necessário (alvo pela mediana); `class_weight="balanced"` onde disponível | Classes {res['distribuicao_alvo']['Baixo risco']} × {res['distribuicao_alvo']['Alto risco']} |
| Divisão | {int((1-R['problema']['proporcao_teste'])*100)}% treino ({R['problema']['n_treino']}) / {int(R['problema']['proporcao_teste']*100)}% teste ({R['problema']['n_teste']}), estratificada, semente {R['problema']['semente']} | Teste nunca é usado para escolher nada |

### 2.5 Modelagem, validação cruzada e Grid Search (`src/treinar.py`)
Foram comparados seis algoritmos: Regressão Logística, K-Nearest Neighbors, Árvore de Decisão, Random Forest, SVM com kernel RBF e Gradient Boosting. Cada um foi encapsulado em um `Pipeline` do scikit-learn (pré-processamento + classificador) e otimizado com `GridSearchCV` usando `StratifiedKFold` com {R['problema']['folds']} dobras, otimizando **AUROC**. Ao todo, {comb} combinações de hiperparâmetros foram avaliadas.

{tabela_hp()}

**Por que AUROC como métrica do Grid Search:** é independente do limiar de decisão e mede a capacidade de ordenar grupos por risco, que é o uso prático do modelo. **Por que validação estratificada:** garante a mesma proporção de classes em cada dobra, com {R['problema']['n_treino']} registros de treino isso evita dobras desbalanceadas por acaso. **Critério de escolha do modelo final:** maior AUROC médio na validação cruzada do treino (e não no teste), para não escolher o modelo "olhando a prova".

### 2.6 Avaliação
No conjunto de teste ({R['problema']['n_teste']} registros, nunca vistos), cada modelo foi avaliado com AUROC, F1-score, sensibilidade (recall da classe Alto risco), especificidade, precisão, acurácia e matriz de confusão. Curvas ROC e Precisão-Recall foram geradas para todos.

---

## 3. Resultados

### 3.1 Desempenho dos modelos

{tabela_modelos()}

**Matrizes de confusão (teste):**

{tabela_cm()}

### 3.2 Discussão
- O **{melhor}** foi o modelo escolhido, com AUROC de {f(M[melhor]['cv_auroc_media'])} ± {f(M[melhor]['cv_auroc_dp'],2)} na validação cruzada e {f(mb['auroc'])} no teste. Classificou corretamente {pc(mb['acuracia'])} dos grupos, com sensibilidade de {pc(mb['sensibilidade'])} e especificidade de {pc(mb['especificidade'])}: erra pouco nos dois sentidos.
- Os modelos de conjunto (Random Forest e Gradient Boosting) e o SVM ficaram próximos entre si e acima dos modelos mais simples. A Árvore de Decisão isolada teve o pior desempenho ({f(M['Árvore de Decisão']['teste']['auroc'])} de AUROC), o que confirma a redução de variância obtida pelos métodos de ensemble.
- A Regressão Logística, mesmo sendo linear, alcançou AUROC de {f(M['Regressão Logística']['teste']['auroc'])}: a relação entre as variáveis e o alvo é em boa parte monotônica.
- AUROC de validação cruzada e de teste ficaram próximas em todos os modelos: não há sinal de sobreajuste relevante.
- **Importância das variáveis no {melhor}** (queda de AUROC ao permutar): {"; ".join(f"`{i['variavel']}` {f(i['importancia'])}" for i in imp)}. A taxa de mortalidade do hospital de referência da doença é o fator mais informativo, seguida do tempo médio de permanência do próprio grupo.

### 3.3 Limitações
- A base tem {res['registros']} grupos; o conjunto de teste tem {R['problema']['n_teste']}. Cada erro a mais muda a acurácia em cerca de {f(100/R['problema']['n_teste'],1)} pontos percentuais, então as diferenças entre os três melhores modelos não são estatisticamente conclusivas.
- O alvo é um corte estatístico (mediana) sobre grupos agregados, não um desfecho clínico individual. O modelo classifica grupos, não pacientes, e serve como apoio à gestão.
- As variáveis de contexto (`tx_mort_macro`, `tx_mort_melhor_hosp`) são agregações do mesmo período e da mesma fonte que originam o alvo. Mesmo estando em outro nível de agregação, carregam parte da informação de mortalidade do grupo. Para medir esse efeito, treinamos uma versão do modelo sem elas (tabela abaixo). Uma validação ainda mais rigorosa seria calculá-las com dados de um período anterior ao do alvo. Em produção, precisariam ser recalculadas periodicamente.
{bloco_ablacao()}

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
"""
(C.RAIZ / "docs" / "Documentacao.md").write_text(doc, encoding="utf-8")
print("[docs] docs/Documentacao.md gerado")
