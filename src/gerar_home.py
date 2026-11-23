"""
Gera index.html na raiz do repositório (página do GitHub Pages usada na apresentação).

Conteúdo:
  1. Consulta da taxa de mortalidade: o usuário escolhe o capítulo da CID-10 e a
     faixa etária e vê a taxa observada na base, a classificação do modelo e a
     tabela da doença em todas as idades. Funciona sem a API: as probabilidades
     do modelo final são pré-calculadas (média de permanência de 0,1 a 20 dias).
  2. Principais doenças da base (por número estimado de óbitos) e os principais
     tratamentos de cada grupo.

Cores: as da bandeira de Pernambuco (azul, branco, amarelo, verde e vermelho).

Uso: python src/gerar_home.py
"""
import json

import joblib
import numpy as np
import pandas as pd

import config as C

REPO = "https://github.com/juliobatista04/TSPE-v2"

modelo = joblib.load(C.DIR_MODELOS / "melhor_modelo.joblib")
meta = json.load(open(C.DIR_MODELOS / "metadados.json", encoding="utf-8"))
R = json.load(open(C.DIR_RESULTADOS / "resultados_modelos.json", encoding="utf-8"))
df = pd.read_csv(C.ARQ_DATASET, sep=";")
DOENCAS, FAIXAS = meta["doencas"], meta["faixas_etarias"]

br = lambda v, c=2: f"{v:.{c}f}".replace(".", ",")
mil = lambda v: f"{int(v):,}".replace(",", ".")

# Nomes por extenso (o TABNET abrevia alguns capítulos)
NOME = {
    "Algumas afec originadas no período perinatal": "Afecções originadas no período perinatal",
    "Algumas doenças infecciosas e parasitárias": "Doenças infecciosas e parasitárias",
    "Doenças sangue órgãos hemat e transt imunitár": "Doenças do sangue e transtornos imunitários",
    "Doenças sist osteomuscular e tec conjuntivo": "Doenças do sistema osteomuscular e do tecido conjuntivo",
    "Doenças endócrinas nutricionais e metabólicas": "Doenças endócrinas, nutricionais e metabólicas",
    "Gravidez parto e puerpério": "Gravidez, parto e puerpério",
    "Lesões enven e alg out conseq causas externas": "Lesões, envenenamentos e causas externas",
    "Malf cong deformid e anomalias cromossômicas": "Malformações congênitas e anomalias cromossômicas",
    "Sint sinais e achad anorm ex clín e laborat": "Sintomas e achados anormais de exames",
}
nome = lambda d: NOME.get(d, d)

# ── Probabilidades do modelo (grade de permanência 0,1 a 20,0 dias) ───────
PERM = np.round(np.arange(1, 201) / 10, 1)
med = meta["medianas_contexto"]
linhas = []
for d in DOENCAS:
    cd = meta["contexto_por_doenca"].get(d, {})
    for i, f in enumerate(FAIXAS):
        cf = meta["contexto_por_faixa"].get(f, {})
        for p in PERM:
            linhas.append({
                "faixa_etaria_ord": i, "doenca_cid": d, "media_permanencia": float(p),
                "media_perm_macro": cf.get("media_perm_macro", med["media_perm_macro"]),
                "tx_mort_macro": cf.get("tx_mort_macro", med["tx_mort_macro"]),
                "tx_mort_melhor_hosp": cd.get("tx_mort_melhor_hosp", med["tx_mort_melhor_hosp"]),
            })
prob = np.round(modelo.predict_proba(pd.DataFrame(linhas)[meta["colunas"]])[:, 1] * 100).astype(int)
grade, k = {}, 0
for d in range(len(DOENCAS)):
    for f in range(len(FAIXAS)):
        bloco, k = prob[k:k + len(PERM)], k + len(PERM)
        rle, ant = [], None
        for j, v in enumerate(bloco):
            if v != ant:
                rle += [j, int(v)]; ant = v
        grade[f"{d}-{f}"] = rle

# dados observados por combinação: [taxa, internações, permanência]
obs = {f"{DOENCAS.index(r.doenca_cid)}-{FAIXAS.index(r.faixa_etaria)}":
       [round(float(r.tx_mortalidade), 2), int(r.internacoes), round(float(r.media_permanencia), 1)]
       for r in df.itertuples()}

# ── Principais doenças: por número estimado de óbitos ────────────────────
df["obitos"] = df["internacoes"] * df["tx_mortalidade"] / 100
tot = df.groupby("doenca_cid").agg(intern=("internacoes", "sum"), obitos=("obitos", "sum"))
tot["tx"] = 100 * tot["obitos"] / tot["intern"]
tot = tot.sort_values("obitos", ascending=False)

TRAT = {
    "Algumas doenças infecciosas e parasitárias": (
        "Infecções graves, com destaque para a sepse (infecção generalizada), além de tuberculose, HIV/aids, dengue e diarreias infecciosas.",
        ["Antibiótico o mais cedo possível; nos protocolos de sepse, na primeira hora após o diagnóstico",
         "Hidratação venosa e coleta de hemoculturas antes do antibiótico",
         "Suporte em UTI quando há falência de órgãos (pressão baixa, falta de oxigênio, rins)",
         "Tratamento específico conforme o agente: antivirais, antiparasitários, esquema da tuberculose"]),
    "Doenças do aparelho circulatório": (
        "Infarto, AVC (derrame), insuficiência cardíaca e complicações da hipertensão.",
        ["Trombolítico para dissolver o coágulo no infarto e no AVC isquêmico, dentro da janela de tempo",
         "Cateterismo e angioplastia para desobstruir as artérias do coração",
         "Cirurgia cardíaca (ponte de safena, troca de válvula) nos casos indicados",
         "Remédios de uso contínuo para pressão, colesterol e para evitar novos coágulos, e reabilitação"]),
    "Doenças do aparelho respiratório": (
        "Pneumonias, DPOC (enfisema e bronquite crônica), asma e insuficiência respiratória.",
        ["Antibiótico na pneumonia bacteriana",
         "Oxigênio quando a saturação cai",
         "Broncodilatadores e corticoide nas crises de DPOC e asma",
         "Ventilação mecânica em UTI nos casos graves"]),
    "Neoplasias (tumores)": (
        "Cânceres de todos os tipos, como mama, próstata, pulmão, intestino e colo do útero.",
        ["Cirurgia para retirar o tumor",
         "Quimioterapia",
         "Radioterapia",
         "Transplante de medula óssea em alguns cânceres do sangue. Muitas vezes os tratamentos são combinados"]),
    "Doenças do aparelho digestivo": (
        "Cirrose, hemorragias digestivas, pancreatite, apendicite, pedra na vesícula e hérnias.",
        ["Cirurgia: retirada do apêndice ou da vesícula, correção de hérnias",
         "Endoscopia para localizar e conter sangramentos",
         "Tratamento clínico da cirrose e da pancreatite (dieta, remédios, hidratação)",
         "Transplante de fígado nos casos avançados de cirrose"]),
    "Lesões enven e alg out conseq causas externas": (
        "Acidentes de trânsito, quedas, agressões, queimaduras e envenenamentos.",
        ["Atendimento pré-hospitalar (SAMU) e estabilização na emergência",
         "Cirurgia geral e ortopédica (fraturas)",
         "UTI nos traumas graves",
         "Antídotos nos envenenamentos e reabilitação depois da alta"]),
}

principais = []
for d, r in tot.head(6).iterrows():
    desc, trat = TRAT[d]
    principais.append(
        f'<article class="doenca"><h3>{nome(d)}</h3>'
        f'<p class="num"><b>{mil(r.intern)}</b> internações · cerca de <b>{mil(round(r.obitos))}</b> óbitos · '
        f'taxa de mortalidade de <b>{br(r.tx, 1)}%</b></p>'
        f'<p>{desc}</p><p class="rot-trat">Principais tratamentos</p><ul>'
        + "".join(f"<li>{t}</li>" for t in trat) + "</ul></article>")

total_int = df["internacoes"].sum()
total_ob = round(df["obitos"].sum())
top3 = tot.head(3)["obitos"].sum() / df["obitos"].sum()

dados = {"doencas": DOENCAS, "faixas": FAIXAS, "grade": grade, "obs": obs, "corte": meta["corte_alvo"]}
ordem = sorted(range(len(DOENCAS)), key=lambda i: nome(DOENCAS[i]))
opt_d = "".join(f'<option value="{i}"{" selected" if DOENCAS[i] == "Doenças do aparelho circulatório" else ""}>{nome(DOENCAS[i])}</option>' for i in ordem)
opt_f = "".join(f'<option value="{i}"{" selected" if f == "60 a 69 anos" else ""}>{f}</option>' for i, f in enumerate(FAIXAS))

html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Taxa de mortalidade hospitalar no SUS · Pernambuco</title>
<style>
:root{{--azul:#0b3d91;--amarelo:#f6c90e;--verde:#1f7a3a;--vermelho:#c8102e;--tinta:#1f2328;--cinza:#59636e;--linha:#d8dee4;--fundo:#f6f8fa}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:Arial,Helvetica,sans-serif;color:var(--tinta);background:#fff;line-height:1.5;font-size:16px}}
.faixa{{height:6px;background:linear-gradient(90deg,var(--azul) 0 50%,var(--amarelo) 50% 67%,var(--verde) 67% 84%,var(--vermelho) 84%)}}
.wrap{{max-width:900px;margin:0 auto;padding:0 20px}}
header{{padding:22px 0 18px;border-bottom:1px solid var(--linha)}}
h1{{font-size:1.5rem;color:var(--azul)}}
header p{{color:var(--cinza);font-size:.9rem;margin-top:4px}}
section{{padding:26px 0;border-bottom:1px solid var(--linha)}}
h2{{font-size:1.2rem;color:var(--azul);margin-bottom:6px}}
.sub{{color:var(--cinza);font-size:.92rem;margin-bottom:14px}}
.filtros{{display:grid;grid-template-columns:2fr 1fr;gap:12px}}
label{{display:block;font-size:.85rem;font-weight:bold;margin-bottom:4px}}
select,input{{width:100%;padding:8px;border:1px solid #afb8c1;border-radius:4px;font:inherit;background:#fff}}
.resultado{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px}}
.caixa{{border:1px solid var(--linha);border-radius:4px;padding:12px;background:var(--fundo)}}
.caixa .r{{font-size:.8rem;color:var(--cinza)}}
.caixa .v{{font-size:1.6rem;font-weight:bold;margin-top:2px}}
.caixa .d{{font-size:.82rem;color:var(--cinza);margin-top:2px}}
.alto{{color:var(--vermelho)}} .baixo{{color:var(--verde)}}
.sim{{margin-top:14px;font-size:.9rem;background:#fff8d6;border-left:4px solid var(--amarelo);padding:10px 12px}}
.sim input{{width:90px;display:inline-block;margin:0 6px;padding:4px 6px}}
table{{width:100%;border-collapse:collapse;margin-top:16px;font-size:.9rem}}
th,td{{padding:7px 8px;border-bottom:1px solid var(--linha);text-align:left}}
th{{background:var(--azul);color:#fff;font-weight:normal}}
td.n,th.n{{text-align:right}}
tr.sel td{{background:#fff8d6;font-weight:bold}}
.doencas{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.doenca{{border:1px solid var(--linha);border-top:4px solid var(--azul);border-radius:4px;padding:14px}}
.doenca h3{{font-size:1rem;margin-bottom:4px}}
.doenca p{{font-size:.9rem;color:var(--cinza);margin-bottom:6px}}
.doenca p.num{{color:var(--tinta)}}
.rot-trat{{font-weight:bold;color:var(--tinta)!important;margin-top:8px}}
.doenca ul{{margin-left:18px;font-size:.9rem}}
.doenca li{{margin:2px 0}}
.aviso{{font-size:.85rem;color:var(--cinza);margin-top:14px}}
footer{{padding:20px 0 36px;font-size:.82rem;color:var(--cinza)}}
footer p+p{{margin-top:4px}}
footer a{{color:var(--azul)}}
@media(max-width:720px){{.filtros,.resultado,.doencas{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="faixa"></div>
<header><div class="wrap">
  <h1>Taxa de mortalidade hospitalar no SUS · Pernambuco</h1>
  <p>Trabalho de Tópicos Avançados em Big Data com Python · UniFavip Wyden · Prof. Melquezedeque Lima<br>
  Julio César Batista Silva, Teves Gabriel, Luciano Filho, Jonathas Costa e Anderson Davi</p>
</div></header>

<section><div class="wrap">
  <h2>Consultar</h2>
  <p class="sub">Escolha a doença e a faixa etária. Os números vêm das internações do SUS em Pernambuco (DATASUS). A classificação de risco é feita pelo modelo de aprendizado de máquina do projeto ({R['melhor_modelo']}).</p>
  <div class="filtros">
    <div><label for="d">Doença (capítulo da CID-10)</label><select id="d">{opt_d}</select></div>
    <div><label for="fx">Faixa etária</label><select id="fx">{opt_f}</select></div>
  </div>
  <div class="resultado">
    <div class="caixa"><div class="r">Taxa de mortalidade</div><div class="v" id="tx"></div><div class="d" id="txd"></div></div>
    <div class="caixa"><div class="r">Internações</div><div class="v" id="in"></div><div class="d" id="ind"></div></div>
    <div class="caixa"><div class="r">Classificação do modelo</div><div class="v" id="cl"></div><div class="d" id="cld"></div></div>
  </div>
  <div class="sim">E se a média de internação fosse de <input id="p" type="number" min="0.1" max="20" step="0.1"> dias? <span id="simr"></span></div>
  <table>
    <thead><tr><th>Faixa etária</th><th class="n">Internações</th><th class="n">Dias de internação (média)</th><th class="n">Taxa de mortalidade</th><th>Risco</th></tr></thead>
    <tbody id="tab"></tbody>
  </table>
</div></section>

<section><div class="wrap">
  <h2>Principais doenças</h2>
  <p class="sub">Na base são {mil(total_int)} internações e cerca de {mil(total_ob)} óbitos. Os seis grupos abaixo são os que mais concentram óbitos; os três primeiros respondem por {br(100*top3,0)}% do total.</p>
  <div class="doencas">
{''.join(principais)}
  </div>
  <p class="aviso">Os tratamentos listados são informação geral para contextualizar o trabalho. A conduta de cada caso é definida pela equipe de saúde.</p>
</div></section>

<footer><div class="wrap">
  <p>Dados: Ministério da Saúde, DATASUS, SIH/SUS, Morbidade Hospitalar do SUS por local de internação, Pernambuco.</p>
  <p>Tratamentos: INCA (tratamento do câncer), protocolos de sepse da rede Ebserh e linha de cuidado de infarto e AVC do Ministério da Saúde.</p>
  <p>Alto risco: taxa de mortalidade acima da mediana do estado ({br(meta['corte_alvo'])}%). Código e documentação: <a href="{REPO}">{REPO.replace('https://', '')}</a></p>
</div></footer>

<script>
const D={json.dumps(dados, ensure_ascii=False, separators=(',', ':'))};
const $=id=>document.getElementById(id);
const br=(v,c=2)=>v.toFixed(c).replace('.',',');
const mil=v=>v.toLocaleString('pt-BR');
function prob(d,f,p){{const r=D.grade[d+'-'+f],i=Math.min(199,Math.max(0,Math.round(p*10)-1));let v=r[1];for(let k=0;k<r.length;k+=2){{if(r[k]<=i)v=r[k+1];else break;}}return v;}}
function consulta(){{
  const d=+$('d').value,f=+$('fx').value,o=D.obs[d+'-'+f];
  $('p').value=o?o[2]:5;
  if(o){{
    $('tx').textContent=br(o[0])+'%'; $('tx').className='v '+(o[0]>D.corte?'alto':'baixo');
    $('txd').textContent=(o[0]>D.corte?'acima':'igual ou abaixo')+' da mediana do estado ('+br(D.corte)+'%)';
    $('in').textContent=mil(o[1]); $('ind').textContent='média de '+br(o[2],1)+' dias internado';
  }}else{{$('tx').textContent='sem dado';$('tx').className='v';$('txd').textContent='combinação não registrada na base';$('in').textContent='0';$('ind').textContent='';}}
  const pr=prob(d,f,o?o[2]:5),alto=pr>=50;
  $('cl').textContent=alto?'Alto risco':'Baixo risco'; $('cl').className='v '+(alto?'alto':'baixo');
  $('cld').textContent='probabilidade de alto risco: '+pr+'%';
  simula();
  let h='';
  D.faixas.forEach((nm,i)=>{{const x=D.obs[d+'-'+i];if(!x)return;const a=x[0]>D.corte;
    h+='<tr'+(i===f?' class="sel"':'')+'><td>'+nm+'</td><td class="n">'+mil(x[1])+'</td><td class="n">'+br(x[2],1)+'</td><td class="n">'+br(x[0])+'%</td><td class="'+(a?'alto':'baixo')+'">'+(a?'Alto':'Baixo')+'</td></tr>';}});
  $('tab').innerHTML=h;
}}
function simula(){{
  let p=parseFloat(String($('p').value).replace(',','.'));
  if(!(p>0)){{$('simr').textContent='';return;}} if(p>20)p=20;
  const v=prob(+$('d').value,+$('fx').value,p);
  $('simr').innerHTML='O modelo classificaria como <b class="'+(v>=50?'alto':'baixo')+'">'+(v>=50?'alto risco':'baixo risco')+'</b> ('+v+'%).';
}}
$('d').addEventListener('change',consulta); $('fx').addEventListener('change',consulta);
$('p').addEventListener('input',simula); consulta();
</script>
</body>
</html>
"""
(C.RAIZ / "index.html").write_text(html, encoding="utf-8")
print(f"[home] index.html gerado ({(C.RAIZ / 'index.html').stat().st_size/1024:.0f} KB)")
