"""
Etapa 4b — Teste de sensibilidade (ablação) das variáveis de contexto.

Pergunta: tx_mort_macro e tx_mort_melhor_hosp vêm da mesma fonte de
mortalidade que origina o alvo. O desempenho depende delas (vazamento)?

Retreina Random Forest, Gradient Boosting e Regressão Logística com o mesmo
split, a mesma validação cruzada e Grid Search, em quatro cenários, e grava
resultados/ablacao.json (usado pela documentação).

Uso: python src/ablacao.py
"""
from __future__ import annotations

import json
import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C
from preprocessamento import montar_X

warnings.filterwarnings("ignore")

CENARIOS = [
    ("Completo", "as 4 selecionadas",
     ["tx_mort_melhor_hosp", "media_permanencia", "tx_mort_macro", "faixa_etaria_ord"]),
    ("Sem tx_mort_melhor_hosp", "permanência, idade, tx_mort_macro",
     ["media_permanencia", "tx_mort_macro", "faixa_etaria_ord"]),
    ("Sem as duas de contexto", "permanência, idade",
     ["media_permanencia", "faixa_etaria_ord"]),
    ("Sem contexto + capítulo CID (one-hot)", "permanência, idade, doença",
     ["media_permanencia", "faixa_etaria_ord", "doenca_cod"]),
]

MODELOS = [
    ("RF", lambda: RandomForestClassifier(class_weight="balanced", random_state=C.SEMENTE, n_jobs=-1),
     {"clf__n_estimators": [300], "clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 2, 4]}),
    ("GB", lambda: GradientBoostingClassifier(random_state=C.SEMENTE),
     {"clf__n_estimators": [100, 300], "clf__learning_rate": [0.03, 0.1], "clf__max_depth": [2, 3]}),
    ("LR", lambda: LogisticRegression(max_iter=3000, class_weight="balanced"),
     {"clf__C": [0.01, 0.1, 1, 10]}),
]


def main() -> dict:
    df = pd.read_csv(C.ARQ_DATASET, sep=";")
    y = df[C.COL_ALVO].astype(int)
    cv = StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=C.SEMENTE)
    saida = []
    for nome, desc, sel in CENARIOS:
        X = montar_X(df, sel)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=C.TAMANHO_TESTE, stratify=y, random_state=C.SEMENTE)
        num = [c for c in X.columns if c != "doenca_cid"]
        etapas = [("num", StandardScaler(), num)]
        if "doenca_cid" in X.columns:
            etapas.append(("cat", OneHotEncoder(handle_unknown="ignore"), ["doenca_cid"]))
        linha = {"cenario": nome, "descricao": desc, "variaveis": list(X.columns), "modelos": {}}
        for sigla, fabrica, grade in MODELOS:
            pipe = Pipeline([("pre", ColumnTransformer(etapas)), ("clf", fabrica())])
            gs = GridSearchCV(pipe, grade, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
            p = gs.best_estimator_.predict_proba(X_te)[:, 1]
            linha["modelos"][sigla] = {
                "auroc_cv": round(float(gs.best_score_), 3),
                "auroc_teste": round(float(roc_auc_score(y_te, p)), 3),
                "acuracia_teste": round(float(accuracy_score(y_te, p >= 0.5)), 3),
            }
        saida.append(linha)
        m = linha["modelos"]
        print(f"[ablacao] {nome:40s} RF CV {m['RF']['auroc_cv']:.3f} teste {m['RF']['auroc_teste']:.3f} | "
              f"GB CV {m['GB']['auroc_cv']:.3f} teste {m['GB']['auroc_teste']:.3f}")
    with open(C.DIR_RESULTADOS / "ablacao.json", "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)
    return {"cenarios": saida}


if __name__ == "__main__":
    main()
