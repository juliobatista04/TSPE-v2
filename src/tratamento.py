"""
Etapa 1: Tratamento dos dados brutos do TABNET (SIH/SUS).

Reproduz a lógica do notebook original (tratamento_dados_ML_2.ipynb),
agora em funções reutilizáveis e com log do que foi feito em cada passo.

Entrada : CSVs exportados do TABNET em data/raw/ (latin1, separador ';')
Saída   : data/processado/*.csv (formato longo) e data/processado/dataset_ml.csv

Uso:
    python src/tratamento.py
Se data/raw/ estiver vazio, o script usa o dataset herdado do projeto
anterior (dataset_ml_legado.csv) e avisa.
"""
from __future__ import annotations

import codecs
import json
import re
from io import StringIO

import numpy as np
import pandas as pd

import config as C

LOG: dict = {"etapas": [], "ausentes_antes": {}, "linhas": {}}


def _log(msg: str) -> None:
    print(f"[tratamento] {msg}")
    LOG["etapas"].append(msg)


# ── Leitura ───────────────────────────────────────────────────────────────
def ler_csv_tabnet(chave: str) -> pd.DataFrame:
    """Lê um CSV do TABNET pulando cabeçalho (3 linhas) e rodapé (Fonte/Notas)."""
    caminho = C.DIR_RAW / C.ARQUIVOS_RAW[chave]
    linhas = []
    with codecs.open(caminho, "r", encoding="latin1") as f:
        for i, linha in enumerate(f):
            if i < 3:
                continue
            linha = linha.strip()
            if not linha:
                continue
            baixo = linha.lower()
            if any(p in baixo for p in ("fonte", "notas", "dados", "consulte")):
                break
            linhas.append(linha)
    df = pd.read_csv(StringIO("\n".join(linhas)), sep=";", quotechar='"')
    _log(f"{chave}: {df.shape[0]} linhas × {df.shape[1]} colunas lidas")
    return df


# ── Limpezas ──────────────────────────────────────────────────────────────
def limpar_numero(val):
    """'1.234,5' -> 1234.5 ; '-' / '...' -> NaN"""
    if pd.isna(val) or str(val).strip() in ("-", "...", ""):
        return np.nan
    s = str(val).strip().replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def limpar_prefixo_numerico(val) -> str:
    """'260010 AFOGADOS DA INGAZEIRA' -> 'AFOGADOS DA INGAZEIRA'"""
    return re.sub(r"^\d+\s+", "", str(val).strip()).strip()


def limpar_doenca(val) -> str:
    """'IX.  Doenças do aparelho circulatório' -> 'Doenças do aparelho circulatório'"""
    return re.sub(r"^[IVXLC]+\.\s*", "", str(val).strip()).strip()


def derreter(df: pd.DataFrame, col_chave: str, nome_dim: str, nome_valor: str) -> pd.DataFrame:
    """Tabela larga (uma coluna por categoria) -> formato longo."""
    cols = [c for c in df.columns if c not in (col_chave, "Total")]
    return df.melt(id_vars=[col_chave], value_vars=cols, var_name=nome_dim, value_name=nome_valor)


def _preparar(df: pd.DataFrame, limpador) -> pd.DataFrame:
    df = df.copy()
    df.iloc[:, 0] = df.iloc[:, 0].apply(limpador)
    df = df.drop(columns=["Total"], errors="ignore")
    df = df.dropna(subset=[df.columns[0]])
    df = df[df.iloc[:, 0] != "Total"]
    return df


def montar_tabela(prefixo: str, limpador, nome_id: str, nome_dim: str) -> pd.DataFrame:
    """Lê internações, permanência e mortalidade de um cruzamento e une tudo."""
    partes = {
        "internacoes":       ler_csv_tabnet(f"{prefixo}_internacoes"),
        "media_permanencia": ler_csv_tabnet(f"{prefixo}_mediaPermanencia"),
        "tx_mortalidade":    ler_csv_tabnet(f"{prefixo}_taxaMortalidade"),
    }
    longos = []
    for nome_valor, df in partes.items():
        df = _preparar(df, limpador)
        longo = derreter(df, df.columns[0], nome_dim, nome_valor)
        longo = longo.rename(columns={longo.columns[0]: nome_id})
        longos.append(longo)

    base = longos[0]
    for outro in longos[1:]:
        base = base.merge(outro, on=[nome_id, nome_dim], how="outer")

    for col in ("internacoes", "media_permanencia", "tx_mortalidade"):
        base[col] = base[col].apply(limpar_numero)

    # registra ausentes ANTES de qualquer imputação (vai para a análise)
    LOG["ausentes_antes"][prefixo] = base.isna().sum().to_dict()
    LOG["linhas"][prefixo] = int(len(base))
    return base.reset_index(drop=True)


# ── Pipeline completo ─────────────────────────────────────────────────────
def construir_dataset_a_partir_do_raw() -> pd.DataFrame:
    _log("Lendo arquivos brutos do TABNET em data/raw/")

    municipio_idade = montar_tabela("municipio-idade", limpar_prefixo_numerico, "municipio", "faixa_etaria")
    municipio_idade.to_csv(C.DIR_PROC / "municipio_idade.csv", index=False, sep=";")

    municipio_sexo = montar_tabela("municipio-sexo", limpar_prefixo_numerico, "municipio", "sexo")
    municipio_sexo["sexo"] = municipio_sexo["sexo"].map({"Masc": "M", "Fem": "F"}).fillna(municipio_sexo["sexo"])
    municipio_sexo.to_csv(C.DIR_PROC / "municipio_sexo.csv", index=False, sep=";")

    doenca_idade = montar_tabela("doenca-faixaEtaria", limpar_doenca, "doenca_cid", "faixa_etaria")
    doenca_idade.to_csv(C.DIR_PROC / "doenca_idade.csv", index=False, sep=";")

    hosp_doenca = montar_tabela("hospital-doenca", limpar_prefixo_numerico, "hospital", "cap_cid")
    hosp_doenca["cap_cid"] = hosp_doenca["cap_cid"].map(C.CAPITULOS_CID).fillna(hosp_doenca["cap_cid"]).apply(limpar_doenca)
    hosp_doenca = hosp_doenca.rename(columns={"cap_cid": "doenca_cid"})
    hosp_doenca.to_csv(C.DIR_PROC / "hosp_doenca.csv", index=False, sep=";")

    # ── Dataset mestre: capítulo CID × faixa etária ────────────────────────
    df_ml = doenca_idade.copy()

    # contexto estadual por faixa etária (agregado da tabela de municípios)
    macro = municipio_idade.groupby("faixa_etaria").agg(
        internacoes_macro=("internacoes", "sum"),
        media_perm_macro=("media_permanencia", "mean"),
        tx_mort_macro=("tx_mortalidade", "mean"),
    ).reset_index()
    df_ml = df_ml.merge(macro, on="faixa_etaria", how="left")

    # hospital de referência (maior volume) por capítulo CID
    melhor = (hosp_doenca.dropna(subset=["internacoes"])
              .sort_values("internacoes", ascending=False)
              .groupby("doenca_cid").first().reset_index()
              [["doenca_cid", "hospital", "tx_mortalidade"]]
              .rename(columns={"hospital": "melhor_hospital", "tx_mortalidade": "tx_mort_melhor_hosp"}))
    df_ml = df_ml.merge(melhor, on="doenca_cid", how="left")

    df_ml = df_ml[["doenca_cid", "faixa_etaria", "internacoes", "media_permanencia",
                   "tx_mortalidade", "media_perm_macro", "tx_mort_macro",
                   "melhor_hospital", "tx_mort_melhor_hosp"]]
    return df_ml


def finalizar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Regras de negócio finais + criação da variável-alvo."""
    n0 = len(df)
    # Sem internações ou sem permanência não há o que classificar
    df = df[(df["internacoes"] > 0) & (df["media_permanencia"] > 0)].copy()
    _log(f"Removidas {n0 - len(df)} linhas sem internações/permanência")

    # Taxa de mortalidade ausente no TABNET significa zero óbitos no grupo
    n_na = int(df["tx_mortalidade"].isna().sum())
    df["tx_mortalidade"] = df["tx_mortalidade"].fillna(0.0)
    _log(f"tx_mortalidade ausente em {n_na} linhas -> imputado 0 (zero óbitos)")

    # Contexto do hospital de referência ausente -> mediana da coluna
    for col in ("tx_mort_melhor_hosp", "media_perm_macro", "tx_mort_macro"):
        if df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
            _log(f"{col}: ausentes imputados com a mediana ({med:.2f})")
    df["melhor_hospital"] = df["melhor_hospital"].fillna("NAO INFORMADO")

    # Faixa etária como categoria ordenada
    df["faixa_etaria"] = pd.Categorical(df["faixa_etaria"], categories=C.FAIXAS_ETARIAS, ordered=True)
    df = df.dropna(subset=["faixa_etaria"]).sort_values(["doenca_cid", "faixa_etaria"]).reset_index(drop=True)

    # ── Variável-alvo ──────────────────────────────────────────────────────
    corte = float(df[C.COL_ORIGEM_ALVO].median())
    df[C.COL_ALVO] = (df[C.COL_ORIGEM_ALVO] > corte).astype(int)
    _log(f"Alvo '{C.COL_ALVO}': 1 se tx_mortalidade > mediana ({corte:.2f}). "
         f"Distribuição: {df[C.COL_ALVO].value_counts().to_dict()}")
    LOG["corte_alvo"] = corte
    LOG["distribuicao_alvo"] = {str(k): int(v) for k, v in df[C.COL_ALVO].value_counts().items()}
    LOG["shape_final"] = list(df.shape)
    return df


def main() -> pd.DataFrame:
    faltando = [a for a in C.ARQUIVOS_RAW.values() if not (C.DIR_RAW / a).exists()]
    if faltando:
        _log(f"data/raw/ incompleto ({len(faltando)} arquivos faltando). "
             f"Usando dataset herdado: {C.ARQ_DATASET_LEGADO.name}")
        df = pd.read_csv(C.ARQ_DATASET_LEGADO, sep=";")
        LOG["origem"] = "legado"
    else:
        df = construir_dataset_a_partir_do_raw()
        LOG["origem"] = "raw"

    df = finalizar_dataset(df)
    df.to_csv(C.ARQ_DATASET, index=False, sep=";", encoding="utf-8")
    _log(f"Dataset salvo em {C.ARQ_DATASET} com {df.shape[0]} registros e {df.shape[1]} atributos")

    with open(C.DIR_RESULTADOS / "log_tratamento.json", "w", encoding="utf-8") as f:
        json.dump(LOG, f, ensure_ascii=False, indent=2)
    return df


if __name__ == "__main__":
    main()
