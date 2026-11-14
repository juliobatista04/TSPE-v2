"""
Roda o pipeline completo, na ordem, e gera interface, figura e documentação.

    python src/rodar_tudo.py                       # API local
    python src/rodar_tudo.py https://sua-api.app   # API publicada
"""
import runpy
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import tratamento, analise, selecao_variaveis, treinar, build_frontend  # noqa: E402

t0 = time.time()
for nome, fn in [("1/7 tratamento", tratamento.main), ("2/7 análise", analise.main),
                 ("3/7 seleção de variáveis", selecao_variaveis.main), ("4/7 treino", treinar.main)]:
    print(f"\n══════ {nome} ══════")
    fn()
print("\n══════ 5/7 interface ══════")
build_frontend.main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000")
print("\n══════ 6/7 figura da metodologia (draw.io) ══════")
runpy.run_path(str(SRC / "gerar_drawio.py"))
print("\n══════ 7/7 documentação ══════")
runpy.run_path(str(SRC / "gerar_documentacao.py"))
runpy.run_path(str(SRC / "gerar_guia_html.py"))
print(f"\n✔ Pipeline completo em {time.time() - t0:.0f}s. Abra frontend/index.html e suba a API com: python api.py")
