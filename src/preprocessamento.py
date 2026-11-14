"""
Pré-processamento (diretriz 8).

Duas camadas:
  1. codificar_basico(): transformações determinísticas que não dependem
     dos dados de treino (codificação ordinal da faixa etária, código inteiro
     do capítulo CID, log das internações). Usada na seleção de variáveis.
  2. construir_preprocessador(): ColumnTransformer que entra DENTRO do
     Pipeline do scikit-learn, para que padronização e one-hot sejam
     ajustados só no treino em cada fold da validação cruzada (sem vazamento).

Justificativas:
  - faixa_etaria é ORDINAL (idade cresce): codificação 0..11 preserva a ordem.
  - doenca_cid é NOMINAL: one-hot para modelos lineares/distância; o código
    inteiro fica disponível para árvores.
  - internacoes tem assimetria forte (cauda longa): log1p aproxima da normal
    e reduz o peso dos outliers sem removê-los (são municípios/doenças reais).
  - StandardScaler: KNN, SVM e Regressão Logística são sensíveis à escala.
  - Balanceamento: o alvo é definido pela mediana, logo já sai balanceado
    (~50/50). Não é necessário SMOTE/undersampling; usamos class_weight
    onde o modelo suporta, por segurança.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import config as C

MAPA_FAIXA = {f: i for i, f in enumerate(C.FAIXAS_ETARIAS)}


def codificar_basico(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Devolve X (numérico, sem vazamento) e y."""
    X = pd.DataFrame(index=df.index)
    X["faixa_etaria_ord"] = df["faixa_etaria"].map(MAPA_FAIXA).astype(int)
    doencas = sorted(df["doenca_cid"].unique())
    X["doenca_cod"] = df["doenca_cid"].map({d: i for i, d in enumerate(doencas)}).astype(int)
    X["log_internacoes"] = np.log1p(df["internacoes"])
    for c in ("media_permanencia", "media_perm_macro", "tx_mort_macro", "tx_mort_melhor_hosp"):
        X[c] = df[c].astype(float)
    y = df[C.COL_ALVO].astype(int)
    return X, y


def montar_X(df: pd.DataFrame, selecionadas: list[str]) -> pd.DataFrame:
    """Monta a matriz de atributos (colunas brutas + derivadas) para o Pipeline."""
    X = pd.DataFrame(index=df.index)
    X["faixa_etaria_ord"] = df["faixa_etaria"].map(MAPA_FAIXA).astype(int)
    X["doenca_cid"] = df["doenca_cid"].astype(str)
    X["log_internacoes"] = np.log1p(df["internacoes"])
    for c in ("media_permanencia", "media_perm_macro", "tx_mort_macro", "tx_mort_melhor_hosp"):
        X[c] = df[c].astype(float)
    # mantém só as selecionadas (doenca_cod da seleção vira doenca_cid aqui)
    manter = [c for c in X.columns if c in selecionadas or (c == "doenca_cid" and "doenca_cod" in selecionadas)]
    return X[manter]


def construir_preprocessador(colunas: list[str]) -> ColumnTransformer:
    numericas = [c for c in colunas if c != "doenca_cid"]
    etapas = [("num", StandardScaler(), numericas)]
    if "doenca_cid" in colunas:
        etapas.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["doenca_cid"]))
    return ColumnTransformer(etapas, remainder="drop")


def construir_preprocessador_arvore(colunas: list[str]) -> ColumnTransformer:
    """Árvores não precisam de escala; usam one-hot da doença mas sem padronizar."""
    numericas = [c for c in colunas if c != "doenca_cid"]
    etapas = [("num", FunctionTransformer(validate=False), numericas)]
    if "doenca_cid" in colunas:
        etapas.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["doenca_cid"]))
    return ColumnTransformer(etapas, remainder="drop")
