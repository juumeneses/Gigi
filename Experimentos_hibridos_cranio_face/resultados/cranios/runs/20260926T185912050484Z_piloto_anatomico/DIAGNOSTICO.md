# Piloto de bordas CF v2: contornos internos

## Correcao e escopo

O extrator usava RETR_EXTERNAL: um anel escuro externo englobava os contornos
internos, ocultando as cavidades. O modo experimental usa RETR_TREE, mantendo
apenas niveis pares (componentes escuros) e excluindo buracos claros.
O comportamento padrao dos detectores anteriores permanece inalterado.
Foi acrescentado teste sintetico que reproduz esse anel externo.

## Resultado no mesmo lote de desenvolvimento

- 10/10 imagens processadas, nenhuma imagem original alterada.
- v1: 0/40 propostas. v2: 5/40 propostas, em 2/10 imagens.
- 2017_76_MVXP: duas propostas alares.
- 2021_94_RMA: duas propostas alares e uma orbital esquerda.
- Demais oito imagens: abstencao.
- Nenhuma anotacao anatomica validada, nenhum treinamento iniciado.

## Inspecao visual pelo assistente, nao por especialista

Os dois previews foram abertos e examinados. Os contornos nasais apresentam
prolongamentos inferiores e irregularidades, portanto nao devem ser usados como
mascaras de referencia. As propostas alares estao proximas das margens visiveis,
mas precisam de revisao da coordenada exata. A proposta orbital acompanha a
regiao lateral visivel; nao ha par orbital estavel completo neste lote.
Nada foi marcado automaticamente como Correto no painel.

## Decisao

Manter propostas apenas para auditoria, sem exportar labels. A mudanca resolve
uma falha de recuperacao de contornos, nao a precisao anatomica. Cobertura de
12,5% de propostas nao e acuracia. Proximo experimento deve localizar rebordos
orbitais em regioes locais e testar estabilidade de pose/iluminacao, mantendo
esta execucao como comparacao e o conjunto de aceitacao intacto.
