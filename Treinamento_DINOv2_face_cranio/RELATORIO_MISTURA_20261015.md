# Mistura de rankings e qualidade das imagens

## Resultado exploratorio

Foram reutilizados scores e checkpoints existentes. Nenhum novo treinamento
completo foi iniciado. O teste final permaneceu reservado.

Galeria completa: 103 pessoas de validacao, iguais nas duas fontes.

| Configuracao | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| A: InfoNCE, enquadramento, seed 42, neutra + lateral | 17,48% | 28,16% | 36,89% | 50,49% |
| B: triplet, original, ensemble 42/43/44, todas as vistas | 9,71% | 24,27% | 37,86% | 62,14% |
| Mistura A 65% / B 35%, RRF k=1 | 17,48% | 30,10% | 36,89% | 57,28% |

Run da mistura: `artifacts/fusions/20261015T025542080456Z`.
Busca: `artifacts/blend_searches/20261015T025541942759Z`.
Fonte A: `artifacts/fusions/20261015T015657235487Z`, pesos [.5,0,.2].
Fonte B: `artifacts/fusions/20261015T020309305867Z`, pesos [.5,.3,.2].

A busca examinou 105 opcoes (21 pesos x 5 constantes). Prioridade:
Recall@1, depois Recall@10, depois MRR. Os pesos foram escolhidos na mesma
validacao em que estes resultados sao apresentados: nao constitui confirmacao
independente. A mistura nao conservou todo o Recall@10 da fonte B.

Comparada com A, conservou os mesmos 18 acertos Top-1. No Top-10, nove
pessoas entraram e duas sairam, saldo de sete. Top-5 manteve o total, mas
cinco pessoas entraram e cinco sairam. Portanto o ganho agregado nao garante
melhora de cada consulta.

## Matching

Selecionar a execucao marcada **Mistura RRF** em Matching > Modelo e vistas.
Os quatro Recall sao apresentados antes do sorteio; parametros e fontes ficam
no expansor **Parametros do treino e protocolo**. Em seguida, sortear uma amostra
ou enviar as imagens e clicar **Comparar**.

Os parametros da mistura ficam fixos. Uma nova busca pode ser disparada em
Treinamentos > Misturar dois modelos. Modelos individuais continuam permitindo
alterar pesos e vistas, com recalculo das metricas nos scores salvos.

RRF soma `peso / (k + posicao)` em cada fonte. Nao e probabilidade nem cosseno.
O ranking depende da galeria. Gallery-10 e recalculada dentro de cada galeria
fixa, em vez de simplesmente restringir os scores RRF da galeria completa.

Foi executada uma consulta real com dez candidatos. Os scores do Matching
coincidiram com a recombinacao dos scores salvos para a mesma consulta/galeria.
Os mapas exibidos na mistura correspondem a um componente representativo da
fonte B, nao a uma explicacao integral da fusao.

## Qualidade

Triagem: `artifacts/quality_audits/20261015T025631788875Z`.
2975 imagens de treino/validacao, todas legiveis. Ha 165 fotos faciais com lado
menor que 224 pixels: 55 neutras, 55 sorrindo e 55 laterais, de 55 pessoas.
Destas imagens, 141 sao de treino e 24 de validacao. Nenhuma foi descartada.

A nitidez e medida por variancia do Laplaciano, apos reducao proporcional a
ate 256 pixels. O alerta usa o quartil inferior do treino de cada vista.
Contraste, fundo, renderizacao e conteudo influenciam essa proxy; os alertas
nao comprovam desfoque nem inadequacao anatomica. A triagem pode ser aberta
em Banco > Qualidade das imagens.

Proximos experimentos recomendados: recuperar originais de maior resolucao,
revisar enquadramento/orientacao, testar augmentacao leve de desfoque apenas
no treino e comparar erros por qualidade. Nao gerar detalhes anatomicos por
super-resolucao como se fossem observados. Ajuste parcial do extrator pode
ser experimentado depois, com regularizacao e selecao na validacao.

## Fontes metodologicas

- RRF: Cormack et al., SIGIR 2009, https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
  (metodo de recuperacao de informacao; nao valida matching face-cranio).
- Qualidade facial: NIST FATE, https://pages.nist.gov/frvt/html/frvt_quality.html
  (contexto de reconhecimento facial, nao prova especifica sobre este banco).
