# data/raw — arquivos brutos do TABNET

Coloque aqui os 12 CSVs exportados do TABNET (DATASUS → SIH/SUS → Morbidade Hospitalar do SUS por local de internação → Pernambuco), com exatamente estes nomes (são os mesmos do notebook original):

| Cruzamento | Medida | Nome do arquivo |
|---|---|---|
| Município × Faixa etária | Internações | `Municipio-idade-intern.csv` |
| Município × Faixa etária | Média de permanência | `municipio-idade-mediaPerm.csv` |
| Município × Faixa etária | Taxa de mortalidade | `municipio-idade-txMort.csv` |
| Município × Sexo | Internações | `municipio-sexo-intern.csv` |
| Município × Sexo | Média de permanência | `municipo-sexo-mediaPerm.csv` |
| Município × Sexo | Taxa de mortalidade | `municipio-sexo-txMort (2).csv` |
| Capítulo CID × Faixa etária | Internações | `doenca_idade_internacoes.csv` |
| Capítulo CID × Faixa etária | Taxa de mortalidade | `doenca_idade_TXmortalidade.csv` |
| Capítulo CID × Faixa etária | Média de permanência | `doenca_idade_mediaPermanencia.csv` |
| Estabelecimento × Capítulo CID | Internações | `hospital_doenca_internacoes.csv` |
| Estabelecimento × Capítulo CID | Taxa de mortalidade | `hospial_doenca_TXmortalidade.csv` |
| Estabelecimento × Capítulo CID | Média de permanência | `hospital_doenca_mediaPermanencia.csv` |

Como exportar no TABNET: escolha Linha (ex.: Capítulo CID-10), Coluna (ex.: Faixa etária), Conteúdo (Internações / Média permanência / Taxa mortalidade), período, e clique em "Mostra" → "Copia como CSV". O arquivo vem em latin1, separador `;`, com 3 linhas de cabeçalho e um rodapé "Fonte: ...". O script `src/tratamento.py` cuida disso.

Se a pasta estiver vazia, o pipeline usa `data/processado/dataset_ml_legado.csv` (o mesmo dataset do projeto anterior) e avisa no log. Para renomear os arquivos, edite `ARQUIVOS_RAW` em `src/config.py`.
