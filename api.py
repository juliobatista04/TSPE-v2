"""
API Flask do TSPE v2.

Diferente da versão anterior, a API NÃO treina nada ao subir: ela carrega o
pipeline já treinado (modelos/melhor_modelo.joblib) e os resultados gravados
pelos scripts. Sobe em menos de um segundo e responde sempre o mesmo modelo
que foi avaliado na documentação.

Rotas:
  GET  /            -> status
  GET  /saude       -> status
  GET  /opcoes      -> listas de doenças e faixas etárias para os formulários
  GET  /resultados  -> resultados_modelos.json (parte 4 da interface)
  GET  /analise     -> analise.json (parte 3 da interface)
  GET  /selecao     -> selecao_variaveis.json
  POST /prever      -> {"doenca": ..., "faixa_etaria": ..., "media_permanencia": ...}

Uso local:  python api.py   (porta 5000)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

RAIZ = Path(__file__).resolve().parent
DIR_MODELOS = RAIZ / "modelos"
DIR_RESULTADOS = RAIZ / "resultados"
DIR_FRONTEND = RAIZ / "frontend"

app = Flask(__name__, static_folder=None)


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


# ── Carregamento (uma vez) ────────────────────────────────────────────────
MODELO = joblib.load(DIR_MODELOS / "melhor_modelo.joblib")
META = json.load(open(DIR_MODELOS / "metadados.json", encoding="utf-8"))


def _json(nome: str) -> dict:
    return json.load(open(DIR_RESULTADOS / nome, encoding="utf-8"))


def montar_entrada(doenca: str, faixa: str, media_permanencia: float) -> pd.DataFrame:
    """Reconstrói as colunas do modelo a partir das 3 informações do usuário."""
    ctx_faixa = META["contexto_por_faixa"].get(faixa, {})
    ctx_doenca = META["contexto_por_doenca"].get(doenca, {})
    med = META["medianas_contexto"]
    linha = {
        "faixa_etaria_ord": META["faixas_etarias"].index(faixa),
        "doenca_cid": doenca,
        "media_permanencia": float(media_permanencia),
        "media_perm_macro": ctx_faixa.get("media_perm_macro", med["media_perm_macro"]),
        "tx_mort_macro": ctx_faixa.get("tx_mort_macro", med["tx_mort_macro"]),
        "tx_mort_melhor_hosp": ctx_doenca.get("tx_mort_melhor_hosp", med["tx_mort_melhor_hosp"]),
    }
    return pd.DataFrame([{c: linha[c] for c in META["colunas"]}])


# ── Rotas ─────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return jsonify({"status": "ok", "servico": "API TSPE v2", "modelo": META["modelo"], "variaveis": META["colunas"]})


@app.get("/saude")
def saude():
    return jsonify({"status": "ok", "modelo": META["modelo"]})


@app.get("/opcoes")
def opcoes():
    return jsonify({"doencas": META["doencas"], "faixas": META["faixas_etarias"], "corte_alvo": META["corte_alvo"]})


@app.get("/resultados")
def resultados():
    return jsonify(_json("resultados_modelos.json"))


@app.get("/analise")
def analise():
    return jsonify(_json("analise.json"))


@app.get("/selecao")
def selecao():
    return jsonify(_json("selecao_variaveis.json"))


@app.route("/prever", methods=["POST", "OPTIONS"])
def prever():
    if request.method == "OPTIONS":
        return "", 204
    dados = request.get_json(silent=True) or {}
    for campo in ("doenca", "faixa_etaria", "media_permanencia"):
        if campo not in dados:
            return jsonify({"erro": f"Campo obrigatório ausente: {campo}"}), 400
    if dados["doenca"] not in META["doencas"]:
        return jsonify({"erro": "Doença não encontrada.", "opcoes": META["doencas"]}), 422
    if dados["faixa_etaria"] not in META["faixas_etarias"]:
        return jsonify({"erro": "Faixa etária não encontrada.", "opcoes": META["faixas_etarias"]}), 422
    try:
        perm = float(dados["media_permanencia"])
    except (TypeError, ValueError):
        return jsonify({"erro": "media_permanencia deve ser numérico."}), 400

    X = montar_entrada(dados["doenca"], dados["faixa_etaria"], perm)
    prob = float(MODELO.predict_proba(X)[0, 1])
    classe = int(prob >= 0.5)
    return jsonify({
        "classe": classe,
        "rotulo": "Alto risco" if classe else "Baixo risco",
        "probabilidade_alto_risco": round(prob, 3),
        "probabilidade_baixo_risco": round(1 - prob, 3),
        "modelo": META["modelo"],
        "corte_alvo_tx_mortalidade": META["corte_alvo"],
        "hospital_referencia": META["hospital_por_doenca"].get(dados["doenca"]),
        "entrada_usada": X.iloc[0].to_dict(),
    })


# ── Serve a interface (opcional: útil para rodar tudo num servidor só) ───
@app.get("/app")
def app_html():
    return send_from_directory(DIR_FRONTEND, "index.html")


if __name__ == "__main__":
    porta = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=porta, debug=False)
