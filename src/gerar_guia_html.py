"""
Gera docs/Guia_TSPE_v2.html: o guia passo a passo e a documentação em um
único HTML estilizado (mesma identidade visual da interface), com índice.

Uso: python src/gerar_guia_html.py
"""
import re

import markdown

import config as C

CSS = """
:root{--deep:#29043B;--purple:#40005F;--mid:#65148C;--light:#8B3DB8;--glow:#B57BDB;--paper:#F6F6F6;--ink:#1A1226;--ink-soft:#4A3D5C;--line:#D9C9E8;--code-bg:#1B0A28;
--display:'Archivo Black',Impact,sans-serif;--body:'DM Sans',system-ui,sans-serif}
*{box-sizing:border-box;margin:0;padding:0}html{scroll-behavior:smooth}
body{font-family:var(--body);background:var(--paper);color:var(--ink);line-height:1.6;font-size:16px}
.capa{background:linear-gradient(135deg,#33064A,var(--deep));color:#fff;padding:80px 0 60px;position:relative;overflow:hidden}
.capa::before{content:"";position:absolute;width:520px;height:520px;border-radius:50%;background:var(--mid);filter:blur(50px);opacity:.5;right:-150px;top:-200px}
.wrap{max-width:1040px;margin:0 auto;padding:0 clamp(16px,4vw,40px);position:relative}
.tag{display:inline-block;font-family:var(--display);text-transform:uppercase;font-size:.68rem;letter-spacing:.08em;padding:.35rem .8rem;border-radius:999px;border:1.5px solid rgba(255,255,255,.4)}
h1{font-family:var(--display);text-transform:uppercase;font-size:clamp(2rem,5vw,3.6rem);line-height:1.05;margin:1rem 0;font-weight:400}
h1 em{font-style:normal;color:var(--glow)}
.capa p{color:rgba(255,255,255,.82);max-width:60ch;font-size:1.05rem}
.layout{display:grid;grid-template-columns:240px 1fr;gap:2.5rem;padding:48px 0}
nav.idx{position:sticky;top:24px;align-self:start;background:#fff;border:1.5px solid var(--line);border-radius:18px;padding:1rem;font-size:.86rem;max-height:calc(100vh - 48px);overflow:auto}
nav.idx .t{font-family:var(--display);text-transform:uppercase;font-size:.68rem;letter-spacing:.08em;color:var(--mid);margin-bottom:.5rem}
nav.idx a{display:block;color:var(--ink-soft);text-decoration:none;padding:.3rem .4rem;border-radius:8px}
nav.idx a:hover{background:var(--paper);color:var(--purple)}
nav.idx a.h3{padding-left:1.1rem;font-size:.8rem}
main{min-width:0}
.doc h1{color:var(--purple);font-size:2rem;margin:3rem 0 1rem}
.doc h2{font-family:var(--display);text-transform:uppercase;color:var(--purple);font-size:1.5rem;margin:2.6rem 0 .8rem;padding-top:1.2rem;border-top:1px solid var(--line);font-weight:400}
.doc h3{font-family:var(--display);text-transform:uppercase;color:var(--mid);font-size:.95rem;margin:1.6rem 0 .5rem;font-weight:400}
.doc p,.doc li{color:var(--ink-soft)}.doc p{margin:.6rem 0}
.doc ul,.doc ol{margin:.5rem 0 .8rem 1.3rem}.doc li{margin:.3rem 0}
.doc strong{color:var(--purple)}
.doc a{color:var(--mid)}
.doc code{background:#EEE3F3;color:var(--purple);padding:.1rem .4rem;border-radius:6px;font-size:.88em}
.doc pre{background:var(--code-bg);color:#E9DDF5;padding:1rem 1.2rem;border-radius:16px;overflow:auto;margin:.8rem 0;font-size:.84rem;line-height:1.5}
.doc pre code{background:none;color:inherit;padding:0}
.doc table{width:100%;border-collapse:collapse;font-size:.88rem;margin:.8rem 0 1.2rem;background:#fff;border-radius:16px;overflow:hidden;box-shadow:0 16px 36px -22px rgba(41,4,59,.4)}
.doc th{background:var(--purple);color:#fff;font-family:var(--display);text-transform:uppercase;font-size:.64rem;letter-spacing:.06em;padding:.7rem .8rem;text-align:left;font-weight:400}
.doc td{padding:.55rem .8rem;border-top:1px solid var(--line);vertical-align:top}
.doc tr:nth-child(even) td{background:var(--paper)}
.doc blockquote{border-left:4px solid var(--light);padding:.6rem 1rem;background:#fff;border-radius:0 14px 14px 0;margin:.8rem 0}
.doc hr{border:0;border-top:1px solid var(--line);margin:2rem 0}
.doc img{max-width:100%;border-radius:14px;border:1.5px solid var(--line)}
.part{background:#fff;border:1.5px solid var(--line);border-radius:22px;padding:1.6rem 2rem;margin-bottom:2rem;box-shadow:0 16px 36px -22px rgba(41,4,59,.4)}
.part .kicker{font-family:var(--display);text-transform:uppercase;font-size:.7rem;letter-spacing:.08em;color:var(--mid)}
footer{background:linear-gradient(180deg,#1C1824,#0F0C14);color:rgba(255,255,255,.6);padding:40px 0;font-size:.85rem;margin-top:3rem}
@media(max-width:860px){.layout{grid-template-columns:1fr}nav.idx{position:static;max-height:none}}
"""


def md_html(texto: str) -> str:
    return markdown.markdown(texto, extensions=["tables", "fenced_code", "toc"])


def indice(html: str) -> str:
    itens = []
    for m in re.finditer(r'<h([23]) id="([^"]+)">(.*?)</h\1>', html):
        nivel, hid, titulo = m.groups()
        titulo = re.sub("<.*?>", "", titulo)
        itens.append(f'<a class="{"h3" if nivel == "3" else ""}" href="#{hid}">{titulo}</a>')
    return "".join(itens)


guia = (C.RAIZ / "docs" / "GUIA_PASSO_A_PASSO.md").read_text(encoding="utf-8")
doc = (C.RAIZ / "docs" / "Documentacao.md").read_text(encoding="utf-8")
guia_html = md_html(guia.split("\n", 1)[1])            # remove o H1 (vai na capa)
doc_html = md_html(doc.split("\n", 1)[1])
# prefixa ids da documentação para não colidir com os do guia
doc_html = re.sub(r'id="', 'id="doc-', doc_html).replace('href="#', 'href="#doc-')
svg = (C.RAIZ / "docs" / "metodologia.svg").read_text(encoding="utf-8")

html = f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TSPE v2 · Guia passo a passo e documentação</title>
<link href="https://fonts.googleapis.com/css2?family=Archivo+Black&family=DM+Sans:ital,wght@0,400;0,500;0,700;1,400&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
<header class="capa"><div class="wrap">
  <span class="tag">WYD6399 · Tópicos Avançados em Big Data com Python · UniFavip Wyden</span>
  <h1>TSPE v2: <em>guia passo a passo</em> e documentação do projeto</h1>
  <p>Como reproduzir, publicar e defender o classificador de risco de mortalidade hospitalar em Pernambuco. Equipe: Julio César Batista Silva, Teves Gabriel, Luciano Filho, Jonathas Costa e Anderson Davi.</p>
</div></header>
<div class="wrap layout">
  <nav class="idx"><div class="t">Guia</div>{indice(guia_html)}<div class="t" style="margin-top:1rem">Documentação</div>{indice(doc_html)}</nav>
  <main class="doc">
    <div class="part"><div class="kicker">Parte A</div><h2 id="guia" style="border:0;margin-top:.3rem">Guia passo a passo</h2>{guia_html}</div>
    <div class="part"><div class="kicker">Figura da metodologia</div><h2 id="figura" style="border:0;margin-top:.3rem">Fluxo em sete etapas</h2><p>Gerada automaticamente com os números do pipeline; a versão editável está em <code>docs/metodologia.drawio</code>.</p><div style="overflow:auto;border:1.5px solid var(--line);border-radius:16px;background:#F6F6F6">{svg}</div></div>
    <div class="part"><div class="kicker">Parte B</div><h2 id="documentacao" style="border:0;margin-top:.3rem">Documentação do projeto</h2>{doc_html}</div>
  </main>
</div>
<footer><div class="wrap">TSPE v2 · Disciplina WYD6399 · Prof. Melquezedeque Lima · UniFavip Wyden · Caruaru, PE · 2026</div></footer>
</body></html>"""
saida = C.RAIZ / "docs" / "Guia_TSPE_v2.html"
saida.write_text(html, encoding="utf-8")
print(f"[guia] {saida} gerado ({saida.stat().st_size/1024:.0f} KB)")
