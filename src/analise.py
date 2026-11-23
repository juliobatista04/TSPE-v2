"""
Etapa 2: Análise exploratória dos dados (EDA).

Responde às diretrizes 5 e 7:
  - comportamento das variáveis, dados ausentes, outliers, atributos sem valor
  - um gráfico de cada variável contra a variável-alvo

Saídas:
  resultados/analise.json          -> números e séries prontas para a interface
  resultados/graficos/eda_*.png    -> figuras para a documentação

Uso: python src/analise.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as C

ROXO, LILAS, VINHO, VERDE = "#40005F", "#B57BDB", "#A8324E", "#1F7A5A"
sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams.update({"axes.titleweight": "bold", "figure.dpi": 130})


def resumo_geral(df: pd.DataFrame) -> dict:
    num = df[C.VARS_NUMERICAS + [C.COL_ORIGEM_ALVO]]
    desc = num.describe().T
    desc["cv"] = desc["std"] / desc["mean"]          # coeficiente de variação
    desc["assimetria"] = num.skew()
    return {
        "registros": int(len(df)),
        "atributos": int(df.shape[1]),
        "ausentes_por_coluna": {k: int(v) for k, v in df.isna().sum().items()},
        "duplicados": int(df.duplicated().sum()),
        "descritivas": {c: {k: round(float(v), 4) for k, v in desc.loc[c].items()} for c in desc.index},
        "distribuicao_alvo": {C.NOMES_CLASSES[int(k)]: int(v) for k, v in df[C.COL_ALVO].value_counts().sort_index().items()},
        "categorias": {
            "doenca_cid": int(df["doenca_cid"].nunique()),
            "faixa_etaria": int(df["faixa_etaria"].nunique()),
            "melhor_hospital": int(df["melhor_hospital"].nunique()),
        },
    }


def outliers_iqr(df: pd.DataFrame) -> dict:
    """Conta outliers pela regra 1,5×IQR em cada variável numérica."""
    out = {}
    for c in C.VARS_NUMERICAS + [C.COL_ORIGEM_ALVO]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[c] < lim_inf) | (df[c] > lim_sup)
        out[c] = {
            "quantidade": int(mask.sum()),
            "percentual": round(100 * mask.mean(), 2),
            "limite_inferior": round(float(lim_inf), 3),
            "limite_superior": round(float(lim_sup), 3),
            "maximo": round(float(df[c].max()), 3),
        }
    return out


def series_para_interface(df: pd.DataFrame) -> dict:
    """Séries agregadas que a interface desenha com Chart.js."""
    s = {}

    # Alvo × faixa etária (proporção de alto risco)
    g = df.groupby("faixa_etaria", observed=True)[C.COL_ALVO].agg(["mean", "count"])
    s["alto_risco_por_faixa"] = {
        "labels": list(g.index.astype(str)),
        "proporcao": [round(float(v), 3) for v in g["mean"]],
        "n": [int(v) for v in g["count"]],
    }

    # Alvo × capítulo CID (ordenado)
    g = df.groupby("doenca_cid")[C.COL_ALVO].agg(["mean", "count"]).sort_values("mean", ascending=False)
    s["alto_risco_por_doenca"] = {
        "labels": list(g.index),
        "proporcao": [round(float(v), 3) for v in g["mean"]],
        "n": [int(v) for v in g["count"]],
    }

    # Mortalidade média por faixa etária (contexto)
    g = df.groupby("faixa_etaria", observed=True)[C.COL_ORIGEM_ALVO].mean()
    s["mortalidade_media_por_faixa"] = {"labels": list(g.index.astype(str)), "valores": [round(float(v), 2) for v in g]}

    # Boxplot de cada numérica por classe (quartis)
    s["boxplots"] = {}
    for c in C.VARS_NUMERICAS:
        s["boxplots"][c] = {}
        for k, nome in C.NOMES_CLASSES.items():
            v = df.loc[df[C.COL_ALVO] == k, c]
            q = v.quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
            s["boxplots"][c][nome] = {"min": q[0], "q1": q[1], "mediana": q[2], "q3": q[3], "max": q[4], "media": float(v.mean())}

    # Dispersão internações × permanência colorida pela classe
    s["dispersao"] = [
        {"x": float(r.internacoes), "y": float(r.media_permanencia), "classe": int(r.alto_risco),
         "doenca": r.doenca_cid, "faixa": str(r.faixa_etaria)}
        for r in df.itertuples()
    ]

    # Heatmap doença × faixa (taxa de mortalidade)
    piv = df.pivot_table(index="doenca_cid", columns="faixa_etaria", values=C.COL_ORIGEM_ALVO, observed=True)
    s["heatmap_mortalidade"] = {
        "linhas": list(piv.index),
        "colunas": [str(c) for c in piv.columns],
        "valores": [[None if pd.isna(v) else round(float(v), 2) for v in linha] for linha in piv.values],
    }

    # Matriz de correlação (Spearman, por causa da assimetria)
    corr = df[C.VARS_NUMERICAS + [C.COL_ORIGEM_ALVO, C.COL_ALVO]].corr(method="spearman")
    s["correlacao"] = {"labels": list(corr.columns), "valores": [[round(float(v), 3) for v in l] for l in corr.values]}

    # Histogramas
    s["histogramas"] = {}
    for c in C.VARS_NUMERICAS + [C.COL_ORIGEM_ALVO]:
        vals = df[c].values
        contagem, bordas = np.histogram(np.log1p(vals) if c == "internacoes" else vals, bins=20)
        s["histogramas"][c] = {"log": c == "internacoes", "bordas": [round(float(b), 3) for b in bordas], "contagem": [int(x) for x in contagem]}
    return s


# ── Figuras para a documentação ───────────────────────────────────────────
def figuras(df: pd.DataFrame) -> list[str]:
    salvos = []
    pal = {0: LILAS, 1: ROXO}
    rot = {0: C.NOMES_CLASSES[0], 1: C.NOMES_CLASSES[1]}
    d = df.copy()
    d["classe"] = d[C.COL_ALVO].map(rot)

    # 1. Distribuição do alvo
    fig, ax = plt.subplots(figsize=(5, 3.5))
    d["classe"].value_counts().plot.bar(ax=ax, color=[LILAS, ROXO])
    ax.set_title("Distribuição da variável-alvo")
    ax.set_xlabel(""); ax.set_ylabel("registros"); plt.xticks(rotation=0)
    salvos.append(_salvar(fig, "eda_01_alvo"))

    # 2. Boxplot de cada numérica por classe
    fig, axes = plt.subplots(1, len(C.VARS_NUMERICAS), figsize=(3.2 * len(C.VARS_NUMERICAS), 3.8))
    for ax, c in zip(axes, C.VARS_NUMERICAS):
        sns.boxplot(data=d, x="classe", y=c, hue="classe", ax=ax, legend=False,
                    order=[rot[0], rot[1]], palette={rot[0]: LILAS, rot[1]: ROXO})
        if c == "internacoes":
            ax.set_yscale("log")
        ax.set_title(c); ax.set_xlabel("")
    fig.suptitle("Variáveis numéricas × alvo")
    salvos.append(_salvar(fig, "eda_02_boxplots"))

    # 3. Faixa etária × alvo
    fig, ax = plt.subplots(figsize=(8, 3.8))
    g = d.groupby("faixa_etaria", observed=True)[C.COL_ALVO].mean()
    ax.bar(g.index.astype(str), g.values, color=ROXO)
    ax.set_ylim(0, 1); ax.set_ylabel("proporção de alto risco"); ax.set_title("Faixa etária × alvo")
    plt.xticks(rotation=35, ha="right")
    salvos.append(_salvar(fig, "eda_03_faixa_alvo"))

    # 4. Capítulo CID × alvo
    fig, ax = plt.subplots(figsize=(8, 6))
    g = d.groupby("doenca_cid")[C.COL_ALVO].mean().sort_values()
    ax.barh(g.index, g.values, color=ROXO)
    ax.set_xlim(0, 1); ax.set_xlabel("proporção de alto risco"); ax.set_title("Capítulo CID × alvo")
    salvos.append(_salvar(fig, "eda_04_doenca_alvo"))

    # 5. Heatmap doença × faixa
    piv = d.pivot_table(index="doenca_cid", columns="faixa_etaria", values=C.COL_ORIGEM_ALVO, observed=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.heatmap(piv, cmap="Purples", ax=ax, cbar_kws={"label": "tx. mortalidade (%)"})
    ax.set_title("Taxa de mortalidade: capítulo CID × faixa etária"); ax.set_xlabel(""); ax.set_ylabel("")
    salvos.append(_salvar(fig, "eda_05_heatmap"))

    # 6. Correlação
    corr = d[C.VARS_NUMERICAS + [C.COL_ORIGEM_ALVO, C.COL_ALVO]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="PuOr", center=0, ax=ax)
    ax.set_title("Correlação de Spearman")
    salvos.append(_salvar(fig, "eda_06_correlacao"))

    # 7. Dispersão
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k in (0, 1):
        sub = d[d[C.COL_ALVO] == k]
        ax.scatter(sub["internacoes"], sub["media_permanencia"], s=22, alpha=.75, color=pal[k], label=rot[k])
    ax.set_xscale("log"); ax.set_xlabel("internações (log)"); ax.set_ylabel("média de permanência (dias)")
    ax.legend(); ax.set_title("Internações × permanência, por classe")
    salvos.append(_salvar(fig, "eda_07_dispersao"))
    return salvos


def _salvar(fig, nome: str) -> str:
    caminho = C.DIR_GRAFICOS / f"{nome}.png"
    fig.tight_layout(); fig.savefig(caminho); plt.close(fig)
    return str(caminho.relative_to(C.RAIZ))


def main() -> dict:
    df = pd.read_csv(C.ARQ_DATASET, sep=";")
    df["faixa_etaria"] = pd.Categorical(df["faixa_etaria"], categories=C.FAIXAS_ETARIAS, ordered=True)

    resultado = {
        "resumo": resumo_geral(df),
        "outliers": outliers_iqr(df),
        "series": series_para_interface(df),
        "figuras": figuras(df),
    }
    with open(C.DIR_RESULTADOS / "analise.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=1)

    r = resultado["resumo"]
    print(f"[analise] {r['registros']} registros, {r['atributos']} atributos, "
          f"alvo = {r['distribuicao_alvo']}")
    print("[analise] outliers (1,5×IQR):", {k: v["quantidade"] for k, v in resultado["outliers"].items()})
    print(f"[analise] {len(resultado['figuras'])} figuras salvas em resultados/graficos/")
    return resultado


if __name__ == "__main__":
    main()
