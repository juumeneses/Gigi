# Leitura do piloto de regioes

Execucao: piloto_20260929_v3. Amostra: 11 pares, 10 identidades de treino e uma de
validacao, dos anos 2017 e 2019. Nenhuma identidade do teste final. Nao representa
todos os anos ou todas as condicoes do dataset.

## Resultados tecnicos

- MediaPipe: 478 pontos em cada uma das 11 faces originais; deteccao em 77/77
  imagens contando as seis perturbacoes por face. Isto nao comprova anatomia.
- Cinco boxes faciais disponiveis em 11/11; a margem da box mandibular ultrapassa
  a imagem em 3/11. Nao significa automaticamente corte anatomico.
- Crânio: boxes orbitais em 11/11; nasal e central em 8/11. GP, MVXP e VDSR
  nao possuem todos os pontos necessarios. Nenhuma coordenada foi imputada.
- Box mandibular craniana: margem ultrapassa a imagem em 6/11 casos.
- Mascaras cranianas com primeiro plano em 77/77 variantes; cinco originais
  tocam bordas. Nem existencia nem estabilidade provam segmentacao correta.
- 18/77 variantes faciais tiveram alerta de corte de box; seu deslocamento
  agregado nao foi interpretado como instabilidade. Nas 59 restantes, incluindo
  os 11 originais com deslocamento zero, mediana das medianas = 0,674 px e
  maior P95 por variante = 3,459 px. Consultar CSV antes de agregar por regiao.
- Maior discordancia mediana normalizada entre propostas e MediaPipe: ponto
  candidato da glabela (2,24% da diagonal). E discordancia, nao erro anatomico.
- HTML: 44 imagens embutidas, resumo, estabilidade e discordancias por ponto.

## Inspecao visual e recomendacao

No cranio 2019_233_SFG, a imagem tem limite superior cortado e o contorno Otsu
inclui estrutura inferior de suporte. Nao usar area, altura total ou contorno
global como medida anatomica neste caso. Nao reconstruir a parte ausente.

As boxes orbitais cranianas sao faixas entre as margens propostas, com margem
minima; nao cobrem necessariamente toda a cavidade. Manter como recortes de
auditoria das propostas, nao como segmentacoes orbitais para novo treinamento.

Priorizar olhos e nariz para revisao regional. Na regiao nasal, manter apenas os
oito casos cranianos disponiveis ate revisar as ausencias. Glabela, zigios e
mandibula continuam exploratorios. A box mandibula/queixo e ampla e depende dos
zigios; nao equivale a uma segmentacao mandibular.

Na face 2019_260_WRV, o modelo retorna contorno e queixo apesar de a proposta
anterior do queixo estar ausente. Isso nao resolve visibilidade cutanea: avaliar
barba/oclusao separadamente. Nunca usar deteccao como substituto dessa revisao.

## Verificacao de execucao

O lote foi interrompido apos dois itens e retomado. Na verificacao independente
da versao anterior (mesmo processamento; HTML menos detalhado), os 66 JSONs por
imagem/etapa foram identicos byte a byte entre execucao completa e retomada.
Todos passaram na verificacao de integridade de payload e arquivos associados.
Os testes tambem preservam falhas no denominador e verificam hashes originais.

Nenhuma revisao humana foi preenchida automaticamente. As sessoes A/B estao
disponiveis no painel. Nao ha validacao anatomica independente, novo modelo de
marcacao ou novo matching. A proxima decisao e revisar as regioes, nao treinar.
