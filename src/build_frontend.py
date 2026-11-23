"""
Etapa 5: Gera a interface final (frontend/index.html).

Pega frontend/template.html e injeta, em um único arquivo, todos os JSONs
produzidos pelo pipeline. Resultado: a interface abre com duplo clique,
offline, e só precisa da API para o simulador ao vivo.

Uso:
    python src/build_frontend.py                       # API em http://localhost:5000
    python src/build_frontend.py https://sua-api.app   # API publicada (Railway, Render...)
"""
from __future__ import annotations

import json
import sys

import config as C


def main(api_url: str = "http://localhost:5000") -> None:
    dados = {
        "analise": json.load(open(C.DIR_RESULTADOS / "analise.json", encoding="utf-8")),
        "resultados": json.load(open(C.DIR_RESULTADOS / "resultados_modelos.json", encoding="utf-8")),
        "selecao": json.load(open(C.DIR_RESULTADOS / "selecao_variaveis.json", encoding="utf-8")),
        "tratamento": json.load(open(C.DIR_RESULTADOS / "log_tratamento.json", encoding="utf-8")),
        "metadados": json.load(open(C.DIR_MODELOS / "metadados.json", encoding="utf-8")),
    }
    arq_ab = C.DIR_RESULTADOS / "ablacao.json"
    dados["ablacao"] = json.load(open(arq_ab, encoding="utf-8")) if arq_ab.exists() else None
    # as figuras PNG não entram no HTML (os gráficos são desenhados ao vivo)
    dados["analise"].pop("figuras", None)
    dados["resultados"].pop("figuras", None)

    template = (C.DIR_FRONTEND / "template.html").read_text(encoding="utf-8")
    blob = json.dumps(dados, ensure_ascii=False).replace("</", "<\\/")
    chartjs = (C.DIR_FRONTEND / "chart.umd.js").read_text(encoding="utf-8")  # embutido: funciona offline
    html = (template.replace("/*__DADOS__*/null/*__FIM__*/", blob)
                    .replace("<script>/*__CHARTJS__*/</script>", "<script>" + chartjs + "</script>")
                    .replace("__API_URL__", api_url.rstrip("/")))
    saida = C.DIR_FRONTEND / "index.html"
    saida.write_text(html, encoding="utf-8")
    print(f"[frontend] {saida} gerado ({saida.stat().st_size/1024:.0f} KB) · API: {api_url}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000")
