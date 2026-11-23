"""
Etapa 4: Modelagem, otimização e avaliação (diretrizes 9 e 10).

Para cada modelo:
  - Pipeline(pré-processamento -> classificador), para que escala e one-hot
    sejam ajustados só no treino de cada fold (sem vazamento)
  - GridSearchCV com StratifiedKFold(5) otimizando AUROC
  - Avaliação no conjunto de TESTE (25%, nunca visto): AUROC, F1,
    sensibilidade, especificidade, precisão, acurácia e matriz de confusão
  - Curva ROC e curva Precisão-Recall salvas como séries para a interface

Saídas:
  resultados/resultados_modelos.json  -> tudo que a interface e a documentação usam
  resultados/graficos/modelo_*.png    -> figuras para a documentação
  modelos/melhor_modelo.joblib        -> pipeline completo do melhor modelo (usado pela API)
  modelos/metadados.json              -> colunas, corte do alvo, nome do modelo

Uso: python src/treinar.py
"""
from __future__ import annotations

import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config as C
from preprocessamento import construir_preprocessador, construir_preprocessador_arvore, montar_X

ROXO, LILAS, VINHO = "#40005F", "#B57BDB", "#A8324E"


def catalogo_modelos(colunas: list[str]) -> dict:
    """Modelos vistos em sala + grade de hiperparâmetros de cada um."""
    pre = lambda: construir_preprocessador(colunas)
    pre_arv = lambda: construir_preprocessador_arvore(colunas)
    return {
        "Regressão Logística": {
            "pipeline": Pipeline([("pre", pre()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=C.SEMENTE))]),
            "grade": {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__penalty": ["l2"], "clf__solver": ["lbfgs"]},
            "justificativa": "Modelo linear de referência, interpretável (coeficientes). C controla a regularização.",
        },
        "KNN": {
            "pipeline": Pipeline([("pre", pre()), ("clf", KNeighborsClassifier())]),
            "grade": {"clf__n_neighbors": [3, 5, 7, 9, 11, 15], "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]},
            "justificativa": "Baseado em distância; depende da padronização. Testa k ímpar para evitar empate e pesos por distância.",
        },
        "Árvore de Decisão": {
            "pipeline": Pipeline([("pre", pre_arv()), ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=C.SEMENTE))]),
            "grade": {"clf__max_depth": [2, 3, 4, 5, 6, None], "clf__min_samples_leaf": [1, 3, 5, 10], "clf__criterion": ["gini", "entropy"]},
            "justificativa": "Regras legíveis para o público. max_depth e min_samples_leaf limitam o sobreajuste em base pequena.",
        },
        "Random Forest": {
            "pipeline": Pipeline([("pre", pre_arv()), ("clf", RandomForestClassifier(class_weight="balanced", random_state=C.SEMENTE, n_jobs=-1))]),
            "grade": {"clf__n_estimators": [200, 500], "clf__max_depth": [3, 5, 8, None], "clf__min_samples_leaf": [1, 2, 4], "clf__max_features": ["sqrt", 0.5]},
            "justificativa": "Conjunto de árvores: reduz variância da árvore única e lida bem com interações não lineares.",
        },
        "SVM (RBF)": {
            "pipeline": Pipeline([("pre", pre()), ("clf", SVC(probability=True, class_weight="balanced", random_state=C.SEMENTE))]),
            "grade": {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1, 1], "clf__kernel": ["rbf"]},
            "justificativa": "Fronteira não linear com margem máxima; C e gamma controlam complexidade.",
        },
        "Gradient Boosting": {
            "pipeline": Pipeline([("pre", pre_arv()), ("clf", GradientBoostingClassifier(random_state=C.SEMENTE))]),
            "grade": {"clf__n_estimators": [100, 300], "clf__learning_rate": [0.03, 0.1], "clf__max_depth": [2, 3], "clf__subsample": [0.8, 1.0]},
            "justificativa": "Boosting sequencial de árvores rasas; costuma liderar em dados tabulares.",
        },
    }


def metricas(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "acuracia": round(float(accuracy_score(y_true, y_pred)), 4),
        "precisao": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "sensibilidade": round(float(recall_score(y_true, y_pred)), 4),             # recall da classe 1
        "especificidade": round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "auroc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "matriz_confusao": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> dict:
    df = pd.read_csv(C.ARQ_DATASET, sep=";")
    sel = json.load(open(C.DIR_RESULTADOS / "selecao_variaveis.json", encoding="utf-8"))["selecionadas"]
    X = montar_X(df, sel)
    y = df[C.COL_ALVO].astype(int)
    colunas = list(X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=C.TAMANHO_TESTE, stratify=y, random_state=C.SEMENTE)
    cv = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEMENTE)
    print(f"[treino] treino={len(X_tr)} teste={len(X_te)} | colunas={colunas}")

    resultados, curvas = {}, {}
    melhor_nome, melhor_auc, melhor_pipe = None, -1, None

    for nome, spec in catalogo_modelos(colunas).items():
        t0 = time.time()
        gs = GridSearchCV(spec["pipeline"], spec["grade"], cv=cv, scoring=C.METRICA_GRID, n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr)
        pipe = gs.best_estimator_

        # desempenho em validação cruzada (treino) com o melhor conjunto de hiperparâmetros
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
        cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1")

        y_prob = pipe.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        m = metricas(y_te, y_pred, y_prob)

        fpr, tpr, _ = roc_curve(y_te, y_prob)
        prec, rec, _ = precision_recall_curve(y_te, y_prob)
        curvas[nome] = {
            "roc": {"fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]},
            "pr": {"recall": [round(float(v), 4) for v in rec], "precisao": [round(float(v), 4) for v in prec]},
        }

        resultados[nome] = {
            "melhores_hiperparametros": {k.replace("clf__", ""): (v if isinstance(v, (int, float, str)) or v is None else str(v)) for k, v in gs.best_params_.items()},
            "combinacoes_testadas": int(len(gs.cv_results_["params"])),
            "cv_auroc_media": round(float(cv_auc.mean()), 4), "cv_auroc_dp": round(float(cv_auc.std()), 4),
            "cv_f1_media": round(float(cv_f1.mean()), 4), "cv_f1_dp": round(float(cv_f1.std()), 4),
            "teste": m,
            "tempo_s": round(time.time() - t0, 1),
            "justificativa": spec["justificativa"],
        }
        print(f"[treino] {nome:20s} CV AUROC={cv_auc.mean():.3f}±{cv_auc.std():.3f} | teste AUROC={m['auroc']:.3f} F1={m['f1']:.3f} "
              f"acc={m['acuracia']:.3f} sens={m['sensibilidade']:.3f} esp={m['especificidade']:.3f} ({resultados[nome]['tempo_s']}s)")

        # critério de escolha: maior AUROC em validação cruzada (não no teste, para não "escolher olhando a prova")
        if cv_auc.mean() > melhor_auc:
            melhor_nome, melhor_auc, melhor_pipe = nome, cv_auc.mean(), pipe

    # ── Importância das variáveis no melhor modelo (permutação) ──────────
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(melhor_pipe, X_te, y_te, n_repeats=30, random_state=C.SEMENTE, scoring="roc_auc")
    importancias = sorted(
        [{"variavel": c, "importancia": round(float(v), 4), "dp": round(float(s), 4)}
         for c, v, s in zip(colunas, perm.importances_mean, perm.importances_std)],
        key=lambda r: -r["importancia"])

    # ── Figuras ───────────────────────────────────────────────────────────
    figuras = _figuras(resultados, curvas, melhor_nome, importancias)

    saida = {
        "problema": {
            "alvo": C.COL_ALVO,
            "descricao_alvo": "1 = taxa de mortalidade hospitalar do grupo acima da mediana estadual; 0 = igual ou abaixo",
            "classes": C.NOMES_CLASSES,
            "n_treino": int(len(X_tr)), "n_teste": int(len(X_te)),
            "proporcao_teste": C.TAMANHO_TESTE, "folds": C.N_FOLDS, "metrica_grid": C.METRICA_GRID,
            "variaveis": colunas, "semente": C.SEMENTE,
        },
        "modelos": resultados,
        "curvas": curvas,
        "melhor_modelo": melhor_nome,
        "criterio_escolha": "maior AUROC médio na validação cruzada estratificada (5 folds) do treino",
        "importancias_melhor_modelo": importancias,
        "figuras": figuras,
    }
    with open(C.DIR_RESULTADOS / "resultados_modelos.json", "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=1)

    # ── Modelo final: re-treina o melhor pipeline com TODOS os dados ─────
    melhor_pipe.fit(X, y)
    joblib.dump(melhor_pipe, C.DIR_MODELOS / "melhor_modelo.joblib")
    meta = {
        "modelo": melhor_nome,
        "colunas": colunas,
        "corte_alvo": json.load(open(C.DIR_RESULTADOS / "log_tratamento.json", encoding="utf-8"))["corte_alvo"],
        "faixas_etarias": C.FAIXAS_ETARIAS,
        "doencas": sorted(df["doenca_cid"].unique().tolist()),
        "medianas_contexto": {c: float(df[c].median()) for c in ("media_perm_macro", "tx_mort_macro", "tx_mort_melhor_hosp")},
        "contexto_por_faixa": df.groupby("faixa_etaria")[["media_perm_macro", "tx_mort_macro"]].mean().round(4).to_dict(orient="index"),
        "contexto_por_doenca": df.groupby("doenca_cid")[["tx_mort_melhor_hosp"]].mean().round(4).to_dict(orient="index"),
        "hospital_por_doenca": df.groupby("doenca_cid")["melhor_hospital"].first().to_dict(),
    }
    json.dump(meta, open(C.DIR_MODELOS / "metadados.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[treino] melhor modelo: {melhor_nome} (CV AUROC {melhor_auc:.3f}). Salvo em modelos/melhor_modelo.joblib")
    return saida


def _figuras(res, curvas, melhor, importancias) -> list[str]:
    salvos = []
    nomes = list(res.keys())

    # Comparativo de métricas
    mets = ["auroc", "f1", "acuracia", "sensibilidade", "especificidade", "precisao"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    largura = 0.13
    xs = np.arange(len(nomes))
    cores = plt.cm.Purples(np.linspace(0.45, 0.95, len(mets)))
    for i, m in enumerate(mets):
        ax.bar(xs + i * largura, [res[n]["teste"][m] for n in nomes], largura, label=m, color=cores[i])
    ax.set_xticks(xs + largura * 2.5); ax.set_xticklabels(nomes, rotation=15, ha="right")
    ax.set_ylim(0, 1.05); ax.set_title("Métricas no conjunto de teste"); ax.legend(ncol=6, fontsize=8, loc="lower center")
    salvos.append(_salvar(fig, "modelo_01_metricas"))

    # Curvas ROC
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for n, cor in zip(nomes, plt.cm.Purples(np.linspace(0.4, 1, len(nomes)))):
        ax.plot(curvas[n]["roc"]["fpr"], curvas[n]["roc"]["tpr"], label=f"{n} (AUC {res[n]['teste']['auroc']:.2f})", color=cor, lw=2 if n == melhor else 1.2)
    ax.plot([0, 1], [0, 1], "--", color="#999"); ax.set_xlabel("1 - especificidade"); ax.set_ylabel("sensibilidade")
    ax.set_title("Curvas ROC (teste)"); ax.legend(fontsize=7)
    salvos.append(_salvar(fig, "modelo_02_roc"))

    # Matrizes de confusão
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    for ax, n in zip(axes.ravel(), nomes):
        mc = res[n]["teste"]["matriz_confusao"]
        cm = np.array([[mc["tn"], mc["fp"]], [mc["fn"], mc["tp"]]])
        ConfusionMatrixDisplay(cm, display_labels=["Baixo", "Alto"]).plot(ax=ax, cmap="Purples", colorbar=False)
        ax.set_title(n, fontsize=10)
    fig.suptitle("Matrizes de confusão (teste)")
    salvos.append(_salvar(fig, "modelo_03_matrizes"))

    # CV vs teste
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(xs - 0.2, [res[n]["cv_auroc_media"] for n in nomes], 0.4, yerr=[res[n]["cv_auroc_dp"] for n in nomes], label="AUROC CV (treino)", color=LILAS, capsize=3)
    ax.bar(xs + 0.2, [res[n]["teste"]["auroc"] for n in nomes], 0.4, label="AUROC teste", color=ROXO)
    ax.set_xticks(xs); ax.set_xticklabels(nomes, rotation=15, ha="right"); ax.set_ylim(0.5, 1.02); ax.legend(); ax.set_title("Validação cruzada × teste")
    salvos.append(_salvar(fig, "modelo_04_cv_vs_teste"))

    # Importâncias
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh([i["variavel"] for i in importancias][::-1], [i["importancia"] for i in importancias][::-1],
            xerr=[i["dp"] for i in importancias][::-1], color=ROXO, capsize=3)
    ax.set_xlabel("queda de AUROC ao permutar"); ax.set_title(f"Importância das variáveis: {melhor}")
    salvos.append(_salvar(fig, "modelo_05_importancias"))
    return salvos


def _salvar(fig, nome):
    p = C.DIR_GRAFICOS / f"{nome}.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return str(p.relative_to(C.RAIZ))


if __name__ == "__main__":
    main()
