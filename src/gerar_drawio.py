"""
Gera docs/metodologia.drawio: a figura da metodologia (diretriz 11, parte 2)
em formato Draw.io, preenchida com os números reais do pipeline.

Abra em https://app.diagrams.net (Arquivo > Abrir) ou na extensão do VS Code.
Também exporta docs/metodologia.svg para colar na documentação.

Uso: python src/gerar_drawio.py
"""
import json
from xml.sax.saxutils import escape

import config as C

R = json.load(open(C.DIR_RESULTADOS / "resultados_modelos.json", encoding="utf-8"))
A = json.load(open(C.DIR_RESULTADOS / "analise.json", encoding="utf-8"))
S = json.load(open(C.DIR_RESULTADOS / "selecao_variaveis.json", encoding="utf-8"))
T = json.load(open(C.DIR_RESULTADOS / "log_tratamento.json", encoding="utf-8"))

melhor = R["melhor_modelo"]
m = R["modelos"][melhor]["teste"]
comb = sum(x["combinacoes_testadas"] for x in R["modelos"].values())

ETAPAS = [
    ("1. BASE DE DADOS", f"SIH/SUS · TABNET (DATASUS)<br>Pernambuco · 12 CSVs brutos<br><b>{A['resumo']['registros']} registros × {A['resumo']['atributos']} atributos</b><br>Alvo: alto_risco (tx. mortalidade &gt; mediana {T['corte_alvo']:.2f}%)"),
    ("2. ANÁLISE DOS DADOS", "Ausentes ('-' = zero óbitos)<br>Duplicados · Outliers 1,5×IQR<br>Assimetria · Spearman<br>Gráfico de cada variável × alvo"),
    ("3. SELEÇÃO DE VARIÁVEIS", "Informação mútua<br>|ρ| Spearman com o alvo<br>Importância por permutação (RF)<br>Entra com 2 de 3 critérios<br><b>Selecionadas:<br>" + "<br>".join(", ".join(S["selecionadas"][i:i+2]) for i in range(0, len(S["selecionadas"]), 2)) + "</b>"),
    ("4. PRÉ-PROCESSAMENTO", "Limpeza TABNET · imputação (0 / mediana)<br>Outliers mantidos (grupos reais)<br>log1p(internações) · ordinal(idade)<br>One-hot(doença) · StandardScaler<br>Classes 50/50 · split 75/25 estratificado"),
    ("5. MODELAGEM", "Regressão Logística · KNN<br>Árvore de Decisão · Random Forest<br>SVM (RBF) · Gradient Boosting<br>Pipeline sklearn (pré + modelo)"),
    ("6. OTIMIZAÇÃO", f"GridSearchCV<br>StratifiedKFold (k=5)<br>Métrica: AUROC<br><b>{comb} combinações testadas</b>"),
    ("7. AVALIAÇÃO", f"AUROC · F1 · Sensibilidade<br>Especificidade · Precisão · Acurácia<br>Matriz de confusão · Curva ROC<br><b>Melhor: {melhor} · AUROC {m['auroc']:.3f}</b>"),
]

W, H, GAP, X0, Y0 = 250, 150, 40, 40, 80
cells = []
cid = 2
for i, (titulo, corpo) in enumerate(ETAPAS):
    col, row = i % 4, i // 4
    x, y = X0 + col * (W + GAP), Y0 + row * (H + 90)
    fill = "#40005F" if i == len(ETAPAS) - 1 else "#FFFFFF"
    font = "#FFFFFF" if i == len(ETAPAS) - 1 else "#1A1226"
    valor = f"<b><font color='{'#B57BDB' if i == len(ETAPAS)-1 else '#40005F'}'>{titulo}</font></b><br><br>{corpo}"
    style = (f"rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor={fill};strokeColor=#8B3DB8;strokeWidth=2;"
             f"fontColor={font};fontSize=11;align=left;spacingLeft=10;spacingRight=8;verticalAlign=top;spacingTop=6;fontFamily=Helvetica;")
    cells.append(f'<mxCell id="n{cid}" value="{escape(valor)}" style="{style}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{W}" height="{H}" as="geometry"/></mxCell>')
    cid += 1

for i in range(len(ETAPAS) - 1):
    estilo = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#65148C;strokeWidth=2;endArrow=blockThin;endFill=1;"
    cells.append(f'<mxCell id="e{i}" style="{estilo}" edge="1" parent="1" source="n{2+i}" target="n{3+i}"><mxGeometry relative="1" as="geometry"/></mxCell>')

titulo = ("<b><font style='font-size:18px' color='#40005F'>METODOLOGIA · TSPE</font></b><br>"
          "<font color='#4A3D5C'>Classificação de risco de mortalidade hospitalar em Pernambuco (SIH/SUS)</font>")
cells.insert(0, f'<mxCell id="t1" value="{escape(titulo)}" style="text;html=1;align=left;verticalAlign=middle;fontFamily=Helvetica;" vertex="1" parent="1"><mxGeometry x="{X0}" y="15" width="800" height="50" as="geometry"/></mxCell>')
rodape = "Equipe: Julio César Batista Silva · Teves Gabriel · Luciano Filho · Jonathas Costa · Anderson Davi — WYD6399 · UniFavip Wyden · 2026"
cells.append(f'<mxCell id="t2" value="{escape(rodape)}" style="text;html=1;align=left;fontSize=10;fontColor=#4A3D5C;fontFamily=Helvetica;" vertex="1" parent="1"><mxGeometry x="{X0}" y="{Y0 + 2*(H+90) - 40}" width="1000" height="30" as="geometry"/></mxCell>')

xml = f'''<mxfile host="app.diagrams.net" agent="tspe-v2">
  <diagram id="metodologia" name="Metodologia">
    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" background="#F6F6F6">
      <root><mxCell id="0"/><mxCell id="1" parent="0"/>
      {''.join(cells)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''
out = C.RAIZ / "docs" / "metodologia.drawio"
out.parent.mkdir(exist_ok=True)
out.write_text(xml, encoding="utf-8")
print(f"[drawio] {out} gerado com {len(ETAPAS)} etapas. Abra em app.diagrams.net")

# ── SVG equivalente (para a documentação e para quem não abre o draw.io) ──
import html as _h
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1220" height="{Y0 + 2*(H+90)}" font-family="Helvetica, Arial, sans-serif">',
       '<rect width="100%" height="100%" fill="#F6F6F6"/>',
       f'<text x="{X0}" y="38" font-size="20" font-weight="bold" fill="#40005F">METODOLOGIA · TSPE</text>',
       f'<text x="{X0}" y="58" font-size="12" fill="#4A3D5C">Classificação de risco de mortalidade hospitalar em Pernambuco (SIH/SUS)</text>']
pos = []
for i, (titulo, corpo) in enumerate(ETAPAS):
    col, row = i % 4, i // 4
    x, y = X0 + col * (W + GAP), Y0 + row * (H + 90)
    pos.append((x, y))
    last = i == len(ETAPAS) - 1
    svg.append(f'<rect x="{x}" y="{y}" rx="14" width="{W}" height="{H}" fill="{"#40005F" if last else "#fff"}" stroke="#8B3DB8" stroke-width="2"/>')
    svg.append(f'<text x="{x+10}" y="{y+20}" font-size="11" font-weight="bold" fill="{"#B57BDB" if last else "#40005F"}">{_h.escape(titulo)}</text>')
    linhas = [l.replace("<b>", "").replace("</b>", "").replace("&gt;", ">") for l in corpo.split("<br>")]
    for j, l in enumerate(linhas):
        svg.append(f'<text x="{x+10}" y="{y+40+j*16}" font-size="10" fill="{"#fff" if last else "#1A1226"}">{_h.escape(l)[:46]}</text>')
svg.append('<defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#65148C"/></marker></defs>')
for i in range(len(ETAPAS) - 1):
    (x1, y1), (x2, y2) = pos[i], pos[i + 1]
    if y1 == y2:
        svg.append(f'<line x1="{x1+W}" y1="{y1+H/2}" x2="{x2-2}" y2="{y2+H/2}" stroke="#65148C" stroke-width="2" marker-end="url(#a)"/>')
    else:
        svg.append(f'<path d="M{x1+W/2},{y1+H} L{x1+W/2},{y1+H+45} L{x2+W/2},{y1+H+45} L{x2+W/2},{y2-2}" fill="none" stroke="#65148C" stroke-width="2" marker-end="url(#a)"/>')
svg.append(f'<text x="{X0}" y="{Y0 + 2*(H+90) - 15}" font-size="10" fill="#4A3D5C">{_h.escape(rodape)}</text></svg>')
(C.RAIZ / "docs" / "metodologia.svg").write_text("\n".join(svg), encoding="utf-8")
print("[drawio] docs/metodologia.svg gerado")
