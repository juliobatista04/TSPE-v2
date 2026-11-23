"""
Gera index.html na raiz do repositório: a página inicial do GitHub Pages.

A página tem três blocos: um resumo do projeto, o simulador e a lista do que
cada material do repositório contém. O simulador não depende da API: as
probabilidades do modelo final são pré-calculadas para cada combinação de
capítulo CID, faixa etária e média de permanência (0,1 a 20 dias, passo 0,1)
e embutidas na página.

Uso: python src/gerar_home.py
"""
import json

import joblib
import numpy as np
import pandas as pd

import config as C

REPO = "https://github.com/juliobatista04/TSPE-v2"
PAGES = "https://juliobatista04.github.io/TSPE-v2"

modelo = joblib.load(C.DIR_MODELOS / "melhor_modelo.joblib")
meta = json.load(open(C.DIR_MODELOS / "metadados.json", encoding="utf-8"))
R = json.load(open(C.DIR_RESULTADOS / "resultados_modelos.json", encoding="utf-8"))
A = json.load(open(C.DIR_RESULTADOS / "analise.json", encoding="utf-8"))

# ── Grade de probabilidades ───────────────────────────────────────────────
PERM = np.round(np.arange(1, 201) / 10, 1)          # 0,1 a 20,0 dias
med = meta["medianas_contexto"]
linhas = []
for d in meta["doencas"]:
    for i, f in enumerate(meta["faixas_etarias"]):
        cf = meta["contexto_por_faixa"].get(f, {})
        cd = meta["contexto_por_doenca"].get(d, {})
        for p in PERM:
            linhas.append({
                "faixa_etaria_ord": i, "doenca_cid": d, "media_permanencia": float(p),
                "media_perm_macro": cf.get("media_perm_macro", med["media_perm_macro"]),
                "tx_mort_macro": cf.get("tx_mort_macro", med["tx_mort_macro"]),
                "tx_mort_melhor_hosp": cd.get("tx_mort_melhor_hosp", med["tx_mort_melhor_hosp"]),
            })
X = pd.DataFrame(linhas)[meta["colunas"]]
prob = np.round(modelo.predict_proba(X)[:, 1] * 100).astype(int)

# compacta por trechos constantes: [indice_inicial, prob, indice_inicial, prob, ...]
grade, k = {}, 0
for d in range(len(meta["doencas"])):
    for f in range(len(meta["faixas_etarias"])):
        bloco = prob[k:k + len(PERM)]; k += len(PERM)
        rle, ant = [], None
        for j, v in enumerate(bloco):
            if v != ant:
                rle += [j, int(v)]; ant = v
        grade[f"{d}-{f}"] = rle

# combinações que existem na base, com a taxa observada (para mostrar ao lado)
df = pd.read_csv(C.ARQ_DATASET, sep=";")
obs = {f"{meta['doencas'].index(r.doenca_cid)}-{meta['faixas_etarias'].index(r.faixa_etaria)}":
       [round(float(r.tx_mortalidade), 2), int(r.internacoes), round(float(r.media_permanencia), 1)]
       for r in df.itertuples()}

melhor = R["melhor_modelo"]; mt = R["modelos"][melhor]["teste"]
fv = lambda v, d=3: f"{v:.{d}f}".replace(".", ",")
corte = fv(meta["corte_alvo"], 2)

dados = {"doencas": meta["doencas"], "faixas": meta["faixas_etarias"], "grade": grade, "obs": obs,
         "hospital": meta["hospital_por_doenca"], "corte": meta["corte_alvo"]}

MATERIAIS = [
    ("Interface de análise", f"{PAGES}/frontend/index.html", "frontend/index.html",
     "As quatro partes pedidas na diretriz: problema e base de dados; metodologia com a tabela de seleção de variáveis; gráficos de cada variável contra o alvo; resultados dos seis modelos com métricas, curvas ROC e matrizes de confusão."),
    ("Documentação", f"{REPO}/blob/main/docs/Documentacao.md", "docs/Documentacao.md",
     "Introdução, metodologia, resultados e referências. Inclui as tabelas de estatística descritiva, seleção de variáveis, hiperparâmetros e o teste sem as variáveis de contexto."),
    ("Guia passo a passo", f"{PAGES}/docs/Guia_TSPE_v2.html", "docs/Guia_TSPE_v2.html",
     "Como instalar, rodar o pipeline, subir a API e publicar. Tem uma tabela com as perguntas mais prováveis da banca e a resposta de cada uma."),
    ("Figura da metodologia", f"{REPO}/blob/main/docs/metodologia.png", "docs/metodologia.drawio",
     "As sete etapas do trabalho em um fluxo. O arquivo .drawio abre e edita no diagrams.net; o .png é a versão para colar em slide."),
    ("Código do pipeline", f"{REPO}/tree/main/src", "src/",
     "Um script por etapa, na ordem: tratamento.py (limpeza dos CSVs do TABNET), analise.py, selecao_variaveis.py, treinar.py, ablacao.py e os geradores da interface e da documentação. O rodar_tudo.py executa tudo."),
    ("Modelo treinado e API", f"{REPO}/blob/main/api.py", "modelos/ e api.py",
     f"O {melhor} já treinado (melhor_modelo.joblib) e a API Flask que responde a rota /prever."),
    ("Notebook do Colab", f"{REPO}/blob/main/notebooks/TSPE_v2_pipeline_colab.ipynb", "notebooks/",
     "Roda o projeto inteiro no Google Colab, sem instalar nada no computador."),
    ("Resultados", f"{REPO}/tree/main/resultados", "resultados/",
     "Os números gerados pelo pipeline em JSON e os gráficos em PNG usados na documentação."),
]

itens = "\n".join(
    f'<li><h3><a href="{url}">{titulo}</a> <span class="arq">{arq}</span></h3><p>{desc}</p></li>'
    for titulo, url, arq, desc in MATERIAIS)

# nomes por extenso (o TABNET abrevia alguns capítulos)
NOME = {
    "Algumas afec originadas no período perinatal": "Afecções originadas no período perinatal",
    "Doenças sangue órgãos hemat e transt imunitár": "Doenças do sangue e transtornos imunitários",
    "Doenças sist osteomuscular e tec conjuntivo": "Doenças do sistema osteomuscular e do tecido conjuntivo",
    "Lesões enven e alg out conseq causas externas": "Lesões, envenenamentos e causas externas",
    "Malf cong deformid e anomalias cromossômicas": "Malformações congênitas e anomalias cromossômicas",
    "Sint sinais e achad anorm ex clín e laborat": "Sintomas e achados anormais de exames",
    "Doenças endócrinas nutricionais e metabólicas": "Doenças endócrinas, nutricionais e metabólicas",
    "Gravidez parto e puerpério": "Gravidez, parto e puerpério",
    "Algumas doenças infecciosas e parasitárias": "Doenças infecciosas e parasitárias",
}
ordem = sorted(range(len(meta["doencas"])), key=lambda i: NOME.get(meta["doencas"][i], meta["doencas"][i]))
opt_d = "".join(
    f'<option value="{i}"{" selected" if meta["doencas"][i] == "Doenças do aparelho circulatório" else ""}>'
    f'{NOME.get(meta["doencas"][i], meta["doencas"][i])}</option>' for i in ordem)
opt_f = "".join(f'<option value="{i}"{" selected" if f == "60 a 69 anos" else ""}>{f}</option>' for i, f in enumerate(meta["faixas_etarias"]))

html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TSPE · Simulador de risco de mortalidade hospitalar</title>
<style>
:root{{--tinta:#1d1d1f;--suave:#55555c;--linha:#dcdce0;--fundo:#fafafa;--cor:#40005F;--alto:#9b2c2c;--baixo:#2f6b4f}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--tinta);background:var(--fundo);line-height:1.55;font-size:16px}}
.wrap{{max-width:860px;margin:0 auto;padding:0 20px}}
header{{border-bottom:1px solid var(--linha);background:#fff;padding:28px 0 22px}}
header .disc{{font-size:.85rem;color:var(--suave)}}
h1{{font-size:1.6rem;line-height:1.25;margin:.35rem 0 .5rem;font-weight:700}}
header .eq{{font-size:.9rem;color:var(--suave)}}
section{{padding:30px 0;border-bottom:1px solid var(--linha)}}
h2{{font-size:1.15rem;margin-bottom:.7rem;color:var(--cor)}}
p{{color:var(--suave)}}
p+p{{margin-top:.6rem}}
b{{color:var(--tinta)}}
a{{color:var(--cor)}}
.sim{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:14px}}
form{{background:#fff;border:1px solid var(--linha);border-radius:8px;padding:18px;display:grid;gap:12px}}
label{{font-size:.85rem;font-weight:600}}
select,input{{width:100%;margin-top:4px;padding:8px 10px;border:1px solid #c7c7cc;border-radius:6px;font:inherit;background:#fff}}
.res{{background:#fff;border:1px solid var(--linha);border-radius:8px;padding:18px}}
.rot{{font-size:1.5rem;font-weight:700;margin:.2rem 0}}
.rot.alto{{color:var(--alto)}}.rot.baixo{{color:var(--baixo)}}
.barra{{height:10px;background:#ececf0;border-radius:5px;overflow:hidden;margin:.5rem 0 .3rem}}
.barra i{{display:block;height:100%;background:var(--cor)}}
.peq{{font-size:.85rem;color:var(--suave)}}
.peq+.peq{{margin-top:.45rem}}
ol.mat{{list-style:none;counter-reset:m}}
ol.mat li{{counter-increment:m;position:relative;padding:12px 0 12px 38px;border-top:1px solid var(--linha)}}
ol.mat li:first-child{{border-top:0}}
ol.mat li::before{{content:counter(m);position:absolute;left:0;top:12px;width:26px;height:26px;border-radius:50%;background:#efe7f3;color:var(--cor);font-weight:700;font-size:.85rem;display:grid;place-items:center}}
ol.mat h3{{font-size:1rem}}
.arq{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.78rem;color:var(--suave);font-weight:400;margin-left:4px}}
table{{border-collapse:collapse;width:100%;font-size:.88rem;margin-top:10px;background:#fff}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--linha)}}
th{{font-weight:600}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
footer{{padding:24px 0 40px;font-size:.85rem;color:var(--suave)}}
@media(max-width:700px){{.sim{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header><div class="wrap">
  <div class="disc">WYD6399 Tópicos Avançados em Big Data com Python · UniFavip Wyden · Prof. Melquezedeque Lima</div>
  <h1>Classificação de risco de mortalidade hospitalar no SUS em Pernambuco</h1>
  <div class="eq">Julio César Batista Silva, Teves Gabriel, Luciano Filho, Jonathas Costa e Anderson Davi</div>
</div></header>

<section><div class="wrap">
  <h2>Resumo</h2>
  <p>Usamos dados públicos do SIH/SUS (DATASUS, TABNET) de Pernambuco para montar uma base com {A['resumo']['registros']} grupos de internação. Cada grupo é a combinação de um capítulo da CID-10 com uma faixa etária. O alvo é binário: <b>alto risco</b> quando a taxa de mortalidade hospitalar do grupo fica acima da mediana do estado ({corte}%) e <b>baixo risco</b> quando fica igual ou abaixo.</p>
  <p>Comparamos seis modelos de classificação com validação cruzada estratificada e Grid Search. O escolhido foi o <b>{melhor}</b>, com AUROC de {fv(R['modelos'][melhor]['cv_auroc_media'])} na validação cruzada e {fv(mt['auroc'])} no teste, acertando {fv(100*mt['acuracia'],0)}% dos grupos do conjunto de teste. O modelo classifica grupos, não pacientes. Serve para apoiar o planejamento de leitos e equipes.</p>
</div></section>

<section><div class="wrap">
  <h2>Simulador</h2>
  <p>Escolha a doença, a faixa etária e a média de dias de internação. O resultado vem do {melhor} treinado no projeto.</p>
  <div class="sim">
    <form id="f" onsubmit="return false">
      <div><label for="d">Capítulo da CID-10</label><select id="d">{opt_d}</select></div>
      <div><label for="fx">Faixa etária</label><select id="fx">{opt_f}</select></div>
      <div><label for="p">Média de permanência (dias)</label><input id="p" type="number" min="0.1" max="20" step="0.1" value="5"></div>
    </form>
    <div class="res" id="res"></div>
  </div>
</div></section>

<section><div class="wrap">
  <h2>O que tem em cada material</h2>
  <ol class="mat">
{itens}
  </ol>
</div></section>

<footer><div class="wrap">
  Fonte dos dados: Ministério da Saúde, DATASUS, SIH/SUS, Morbidade Hospitalar do SUS por local de internação, Pernambuco.
  Código: <a href="{REPO}">{REPO.replace('https://', '')}</a>
</div></footer>

<script>
const D={json.dumps(dados, ensure_ascii=False, separators=(',', ':'))};
const $=id=>document.getElementById(id);
const br=(v,c=1)=>v.toFixed(c).replace('.',',');
function prob(d,f,p){{
  const r=D.grade[d+'-'+f], i=Math.min(199,Math.max(0,Math.round(p*10)-1));
  let v=r[1]; for(let k=0;k<r.length;k+=2){{ if(r[k]<=i) v=r[k+1]; else break; }} return v;
}}
function atualiza(){{
  const d=+$('d').value, f=+$('fx').value; let p=parseFloat(String($('p').value).replace(',','.'));
  if(!(p>0)) {{ $('res').innerHTML='<p class="peq">Informe uma média de permanência maior que zero.</p>'; return; }}
  if(p>20) p=20;
  const v=prob(d,f,p), alto=v>=50, o=D.obs[d+'-'+f], h=D.hospital[D.doencas[d]];
  $('res').innerHTML=
    '<div class="peq">Classificação</div>'+
    '<div class="rot '+(alto?'alto':'baixo')+'">'+(alto?'Alto risco':'Baixo risco')+'</div>'+
    '<div class="barra"><i style="width:'+v+'%"></i></div>'+
    '<div class="peq">Probabilidade de alto risco: <b>'+v+'%</b></div>'+
    (o?'<div class="peq">Na base, este grupo teve '+o[1].toLocaleString('pt-BR')+' internações, '+br(o[2])+' dias de permanência média e taxa de mortalidade de '+br(o[0],2)+'% (mediana do estado: '+br(D.corte,2)+'%).</div>'
      :'<div class="peq">Esta combinação não aparece na base; o resultado é uma estimativa do modelo.</div>')+
    (h?'<div class="peq">Hospital com mais internações nesse capítulo: '+h+'.</div>':'');
}}
['d','fx','p'].forEach(id=>$(id).addEventListener('input',atualiza)); atualiza();
</script>
</body>
</html>
"""
(C.RAIZ / "index.html").write_text(html, encoding="utf-8")
print(f"[home] index.html gerado ({(C.RAIZ / 'index.html').stat().st_size/1024:.0f} KB), {len(grade)} combinações no simulador")
