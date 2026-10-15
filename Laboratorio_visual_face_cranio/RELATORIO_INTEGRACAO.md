# Integracao do laboratorio - 13 de outubro de 2026

## Entregas

Lote de 50 pessoas selecionado nos splits de treino/validacao, com dez por ano
(2017, 2019, 2021, 2022, 2023). Foram geradas 100 propostas frontais, face e CF,
sem falhas de processamento. Elas aguardam revisao; nenhuma foi aprovada
automaticamente ou incorporada como novo rotulo de treinamento.

Lote: `artifacts/annotation_batches/20261013T225939396950Z`.
Pedido e log: `artifacts/jobs/20261013T225937253970Z`.

O painel permite editar coordenadas e visibilidade, salvar autor/motivo e criar
nova versao das referencias. Originais e resultados legados foram preservados.

Matching inclui baseline Ridge, duas torres visuais ResNet18 e hibrido, perdas
InfoNCE/triplet, recortes e vistas opcionais, galerias conhecidas e uploads.
As projecoes sao treinadas; backbone congelado. Nao foi executado treinamento
completo do banco nesta entrega.

O modulo craniano de sexo biologico possui importacao de procedencia, treino
multivista e avaliacao independente. Permanece bloqueado no banco atual enquanto
nao houver referencia documental/especializada importada. A curadoria facial
e as estimativas FairFace/modelo facial continuam consultaveis.

## Verificacao

18 testes passaram: perdas mascaradas, coordenadas, geometria com ausencias,
retomada reproduzivel, integridade, splits, galerias estaveis, exclusoes com rank
original, revisoes e baseline craniano com tres vistas em dados sinteticos.

Tambem foi verificada a extracao de recortes com as tres vistas reais nas duas
modalidades: entradas por vista de 1029 componentes produziram embeddings de 64
dimensoes finitos. Pontos frontais nao foram transplantados para as laterais.

Um teste temporario com quatro pares reais executou uma epoca triplet com
ResNet18, ranking de quatro candidatos e avaliacao de duas identidades de
validacao. A execucao concluiu. Esse teste verifica integracao, nao desempenho
cientifico. Seus modelos temporarios foram descartados ao encerrar o teste.

Streamlit AppTest abriu Faces, Cranios, matching e registros sem excecoes.
Teste conhecido criou dez candidatos; Entrega final mostrou os uploads
agrupados; treino craniano ficou desabilitado sem referencias documentadas.

## Proxima execucao

Revisar o lote antes de treinar outra versao dos localizadores. O matching visual
pode ser treinado a partir dos pares existentes; o hibrido deve ser comparado
apos melhorar as referencias. Nenhuma equivalencia anatomica ou identificacao
foi validada por estes testes. Laterais ainda precisam de suas proprias
anotacoes para treinamento de pontos.
