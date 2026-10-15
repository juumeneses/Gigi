# Piloto anatomico: resultado negativo

- Dez imagens originais CF, duas por ano, apenas desenvolvimento.
- Quatro pontos previstos no contrato por imagem: ec bilateral e al bilateral.
- Imagens processadas: 10/10. Propostas produzidas: 0/40.
- Nenhum ponto validado; nenhum treinamento iniciado; originais preservados.

Na primeira imagem inspecionada (2017_76_MVXP), o extrator multilim iares legado
nao retornou candidatos. Preencher as cavidades internas da mascara nao resolveu.
Logo, este piloto nao fornece ainda um lote de coordenadas para aprovacao.
Nao baixar limiares apenas para aumentar cobertura, nem aprovar centros como bordas.

Proximo passo tecnico: substituir a segmentacao de cavidades por busca local de
rebordos, ou obter detector externo supervisionado com anotacoes verificadas.
Revisar visualmente as propostas antes de qualquer exportacao de labels.
O guia e a infraestrutura de revisao estao preparados, mas a geracao automatica
de referencias anatomicas continua sem resolver. Nao confundir execucao concluida
com sucesso cientifico.
