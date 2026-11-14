"""
Etapa 3 — Seleção de variáveis (diretriz 6).

Três critérios independentes, todos calculados só no conjunto de TREINO
para não vazar informação do teste:
  1. Informação mútua (captura relação não linear com o alvo)
  2. Correlação de Spearman com o alvo e colinearidade entre preditoras
  3. Importância por permutação em uma Random Forest de referência

Regra de decisão: a variável entra se passar em pelo menos dois dos três
critérios. Tudo fica registrado em resultados/selecao_variaveis.json com
a justificativa de cada decisão.

Uso: python src/selecao_variaveis.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import config as C
from preprocessamento import codificar_basico


def main() -> dict:
    df = pd.read_csv(C.ARQ_DATASET, sep=";")
    X, y = codificar_basico(df)

    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=C.TAMANHO_TESTE,
                                        stratify=y, random_state=C.SEMENTE)
    cols = list(X_tr.columns)

    # 1. Informação mútua
    discreta = [c in ("faixa_etaria_ord", "doenca_cod") for c in cols]
    mi = mutual_info_classif(X_tr, y_tr, discrete_features=discreta, random_state=C.SEMENTE)
    mi = pd.Series(mi, index=cols)

    # 2. Correlação de Spearman (|rho| com o alvo) e colinearidade
    rho = X_tr.corrwith(y_tr, method="spearman").abs()
    colin = X_tr.corr(method="spearman").abs()
    pares_colineares = [
        {"a": a, "b": b, "rho": round(float(colin.loc[a, b]), 3)}
        for i, a in enumerate(cols) for b in cols[i + 1:] if colin.loc[a, b] > 0.85
    ]

    # 3. Importância por permutação (RF de referência)
    rf = RandomForestClassifier(n_estimators=400, random_state=C.SEMENTE, min_samples_leaf=2)
    rf.fit(X_tr, y_tr)
    perm = permutation_importance(rf, X_tr, y_tr, n_repeats=30, random_state=C.SEMENTE, scoring="roc_auc")
    imp = pd.Series(perm.importances_mean, index=cols)

    # Limiares: metade da média de cada critério (corte relativo, não absoluto)
    lim_mi, lim_rho, lim_imp = mi.mean() * 0.5, 0.10, max(imp.mean() * 0.5, 0.0)

    tabela = []
    for c in cols:
        votos = int(mi[c] >= lim_mi) + int(rho[c] >= lim_rho) + int(imp[c] >= lim_imp)
        tabela.append({
            "variavel": c,
            "info_mutua": round(float(mi[c]), 4),
            "spearman_abs": round(float(rho[c]), 4),
            "importancia_permutacao": round(float(imp[c]), 4),
            "votos": votos,
            "selecionada": votos >= 2,
        })
    tabela.sort(key=lambda r: (-r["votos"], -r["importancia_permutacao"]))

    # Colinearidade: entre duas selecionadas com |rho| > 0,85 fica a de maior importância
    removidas_colin = []
    por_nome = {r["variavel"]: r for r in tabela}
    for p in pares_colineares:
        a, b = por_nome[p["a"]], por_nome[p["b"]]
        if a["selecionada"] and b["selecionada"]:
            perde = a if a["importancia_permutacao"] < b["importancia_permutacao"] else b
            perde["selecionada"] = False
            perde["motivo_extra"] = f"colinear com {p['a'] if perde is b else p['b']} (rho={p['rho']})"
            removidas_colin.append(perde["variavel"])

    selecionadas = [r["variavel"] for r in tabela if r["selecionada"]]
    descartadas = [r["variavel"] for r in tabela if not r["selecionada"]]

    justificativas = {
        "melhor_hospital": "Descartada antes da análise: é um nome (texto livre) com 1 valor por capítulo CID, "
                           "o que a torna redundante com doenca_cid e sem poder de generalização.",
        "tx_mortalidade": "Descartada: é a origem da variável-alvo. Usá-la seria vazamento de informação.",
    }
    for r in tabela:
        if r["selecionada"]:
            justificativas[r["variavel"]] = (
                f"Selecionada com {r['votos']}/3 critérios: IM={r['info_mutua']}, "
                f"|rho|={r['spearman_abs']}, importância={r['importancia_permutacao']}.")
        elif r.get("motivo_extra"):
            justificativas[r["variavel"]] = (
                f"Descartada por colinearidade: {r['motivo_extra']}. Carrega a mesma informação "
                f"(ambas derivam da faixa etária) e tinha menor importância.")
        else:
            justificativas[r["variavel"]] = (
                f"Descartada: passou em apenas {r['votos']}/3 critérios (IM={r['info_mutua']}, "
                f"|rho|={r['spearman_abs']}, importância={r['importancia_permutacao']}).")

    resultado = {
        "criterios": {
            "info_mutua": f"mutual_info_classif; limiar = 50% da média ({lim_mi:.4f})",
            "spearman": f"|rho| com o alvo >= {lim_rho}",
            "permutacao": f"queda média de AUROC ao embaralhar a variável (30 repetições); limiar = 50% da média ({lim_imp:.4f})",
            "regra": "entra quem passa em pelo menos 2 dos 3 critérios; entre colineares (|rho|>0,85) fica a de maior importância",
        },
        "tabela": tabela,
        "pares_colineares": pares_colineares,
        "selecionadas": selecionadas,
        "descartadas": descartadas,
        "justificativas": justificativas,
    }
    with open(C.DIR_RESULTADOS / "selecao_variaveis.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    print("[selecao] " + pd.DataFrame(tabela).to_string(index=False))
    print(f"[selecao] selecionadas: {selecionadas}")
    print(f"[selecao] descartadas : {descartadas}")
    if pares_colineares:
        print(f"[selecao] pares colineares (>0,85): {pares_colineares}")
    return resultado


if __name__ == "__main__":
    main()
