"""
Configuração central do projeto TSPE v2.

Tudo que é caminho, nome de arquivo, constante de domínio ou parâmetro
de modelagem fica aqui. Os outros scripts importam deste módulo, então
mudar algo aqui muda em todo o pipeline.
"""
from pathlib import Path

# ── Caminhos ──────────────────────────────────────────────────────────────
RAIZ = Path(__file__).resolve().parent.parent
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROC = RAIZ / "data" / "processado"
DIR_RESULTADOS = RAIZ / "resultados"
DIR_GRAFICOS = DIR_RESULTADOS / "graficos"
DIR_MODELOS = RAIZ / "modelos"
DIR_FRONTEND = RAIZ / "frontend"

for d in (DIR_RAW, DIR_PROC, DIR_RESULTADOS, DIR_GRAFICOS, DIR_MODELOS):
    d.mkdir(parents=True, exist_ok=True)

# Dataset final usado na modelagem
ARQ_DATASET = DIR_PROC / "dataset_ml.csv"
# Dataset herdado do projeto anterior (fallback quando data/raw está vazio)
ARQ_DATASET_LEGADO = DIR_PROC / "dataset_ml_legado.csv"

# ── Arquivos brutos do TABNET (SIH/SUS, Pernambuco) ──────────────────────
# Chave lógica -> nome do arquivo exportado do TABNET (CSV, latin1, ';')
ARQUIVOS_RAW = {
    # Município × Faixa etária
    "municipio-idade_internacoes":        "Municipio-idade-intern.csv",
    "municipio-idade_mediaPermanencia":   "municipio-idade-mediaPerm.csv",
    "municipio-idade_taxaMortalidade":    "municipio-idade-txMort.csv",
    # Município × Sexo
    "municipio-sexo_internacoes":         "municipio-sexo-intern.csv",
    "municipio-sexo_mediaPermanencia":    "municipo-sexo-mediaPerm.csv",
    "municipio-sexo_taxaMortalidade":     "municipio-sexo-txMort (2).csv",
    # Capítulo CID × Faixa etária
    "doenca-faixaEtaria_internacoes":     "doenca_idade_internacoes.csv",
    "doenca-faixaEtaria_taxaMortalidade": "doenca_idade_TXmortalidade.csv",
    "doenca-faixaEtaria_mediaPermanencia":"doenca_idade_mediaPermanencia.csv",
    # Hospital × Capítulo CID
    "hospital-doenca_internacoes":        "hospital_doenca_internacoes.csv",
    "hospital-doenca_taxaMortalidade":    "hospial_doenca_TXmortalidade.csv",
    "hospital-doenca_mediaPermanencia":   "hospital_doenca_mediaPermanencia.csv",
}

# Ordem natural das faixas etárias (usada para codificação ordinal)
FAIXAS_ETARIAS = [
    "Menor 1 ano", "1 a 4 anos", "5 a 9 anos", "10 a 14 anos",
    "15 a 19 anos", "20 a 29 anos", "30 a 39 anos", "40 a 49 anos",
    "50 a 59 anos", "60 a 69 anos", "70 a 79 anos", "80 anos e mais",
]

# Capítulos da CID-10 (como aparecem no TABNET)
CAPITULOS_CID = {
    "Cap 01": "I.   Algumas doenças infecciosas e parasitárias",
    "Cap 02": "II.  Neoplasias (tumores)",
    "Cap 03": "III. Doenças sangue órgãos hemat e transt imunitár",
    "Cap 04": "IV.  Doenças endócrinas nutricionais e metabólicas",
    "Cap 05": "V.   Transtornos mentais e comportamentais",
    "Cap 06": "VI.  Doenças do sistema nervoso",
    "Cap 07": "VII. Doenças do olho e anexos",
    "Cap 08": "VIII.Doenças do ouvido e da apófise mastóide",
    "Cap 09": "IX.  Doenças do aparelho circulatório",
    "Cap 10": "X.   Doenças do aparelho respiratório",
    "Cap 11": "XI.  Doenças do aparelho digestivo",
    "Cap 12": "XII. Doenças da pele e do tecido subcutâneo",
    "Cap 13": "XIII.Doenças sist osteomuscular e tec conjuntivo",
    "Cap 14": "XIV. Doenças do aparelho geniturinário",
    "Cap 15": "XV.  Gravidez parto e puerpério",
    "Cap 16": "XVI. Algumas afec originadas no período perinatal",
    "Cap 17": "XVII.Malf cong deformid e anomalias cromossômicas",
    "Cap 18": "XVIII.Sint sinais e achad anorm ex clín e laborat",
    "Cap 19": "XIX. Lesões enven e alg out conseq causas externas",
    "Cap 21": "XXI. Contatos com serviços de saúde",
}

# ── Problema de classificação ─────────────────────────────────────────────
# Variável-alvo: ALTO_RISCO = 1 quando a taxa de mortalidade hospitalar do
# grupo (capítulo CID × faixa etária) fica ACIMA da mediana estadual.
# A mediana é calculada no próprio dataset e gravada em resultados/
# para que a API use exatamente o mesmo corte.
COL_ALVO = "alto_risco"
COL_ORIGEM_ALVO = "tx_mortalidade"
NOMES_CLASSES = {0: "Baixo risco", 1: "Alto risco"}

# Variáveis candidatas (antes da seleção)
VARS_NUMERICAS = [
    "internacoes", "media_permanencia",
    "media_perm_macro", "tx_mort_macro", "tx_mort_melhor_hosp",
]
VARS_CATEGORICAS = ["doenca_cid", "faixa_etaria"]

# ── Modelagem ─────────────────────────────────────────────────────────────
SEMENTE = 42
TAMANHO_TESTE = 0.25
N_FOLDS = 5
METRICA_GRID = "roc_auc"   # métrica que o Grid Search otimiza
