# Laboratorio de pontos e regioes

Aplicacao independente usando `Banco` e referencias do piloto anterior.
As referencias sao pseudorrotulos de IA. O treino mede sua reproducao,
nao precisao anatomica independente. Laterais exigem referencias proprias.

## Abrir

Na raiz do repositorio:

```bash
Treinamento_faces/.venv/bin/python -m streamlit run Laboratorio_visual_face_cranio/app.py --server.port 8502
```

O ambiente existente contem Streamlit, OpenCV, NumPy, pandas, PyTorch e MediaPipe.
O painel usa esse mesmo interpretador para executar os comandos. O processamento
roda em subprocesso com log persistido, parada cooperativa e atualizacao da tela.

## Fluxo

1. Em Referencias e historico, preparar as referencias do piloto v3.
2. Em Faces ou Cranios, abrir a subaba Treinamento, selecionar a vista e abrir Treinar e avaliar pontos.
3. Ajustar parametros e clicar Treinar pontos. A selecao do checkpoint usa validacao.
4. Atualizar resultados, abrir Marcacoes, selecionar uma imagem e Aplicar modelo.
5. Selecionar a execucao em Camada exibida para ver as propostas e regioes.
6. Abrir Ampliar e revisar referencias de pontos: o lote inicial de 50 pessoas ja foi gerado.
7. Revisar a visibilidade de cada ponto, corrigir X/Y quando necessario, informar autor e salvar.
8. Incorporar as revisoes para criar novas referencias; selecionar essa versao antes de treinar novamente.
9. Em Matching cranio-face, abrir Treinamento e avaliacao do matching e escolher imagem, geometria ou hibrido.
10. Treinar, selecionar a execucao e usar Teste conhecido ou Entrega final. Os uploads sao agrupados por candidato CF/CE/CD.

Faces e Cranios contem as subabas Originais, Marcacoes, Treinamento e Sexo registrado.
A tabela de sexo reune a curadoria e as estimativas legadas por ID de cadastro,
preservando casos sem correspondencia. Essas fontes nao oferecem uma coluna oficial
documentada; valores curados automaticamente nao sao promovidos a referencia oficial.

Os originais nao sao regravados. Resultados novos ficam em `artifacts`:
`references`, `training`, `processing`, `reviews`, `point_reviews`, `annotation_batches`,
`matching`, `matching_evaluations`, `matching_comparisons`, `uploads`,
`sex_references`, `cranial_sex`, `cranial_sex_predictions`, `overlays` e `jobs`.
As importacoes conservam os splits e hashes originais. Imagem com mesmo nome e hash
diferente nao recebe automaticamente pontos. Na ausencia de copia identica em Banco,
a referencia legada continua explicita. Todas as falhas sao registradas.

O modelo CNN possui 11 saidas de coordenadas, por modalidade e vista, loss mascarada,
baseline medio ajustado no treino e avaliacao por ponto na validacao. Nao estima
visibilidade. Pontos sem supervisao nao recebem coordenadas. As regioes sao boxes
derivadas de grupos de pontos, nao um detector regional independente.

Retomar treinamento conserva a configuracao e aumenta o total de epocas.
O checkpoint `last.pt` inclui otimizador e historico; `best.pt` vem da menor perda
de validacao. O embaralhamento tem semente por epoca para retomada reproduzivel em CPU.
Reprocessamento verifica hash da entrada, configuracao, checkpoint e assets.

## CLI

Os botoes salvam pedidos JSON em `artifacts/jobs/<id>/request.json` e executam:

```bash
Treinamento_faces/.venv/bin/python -u Laboratorio_visual_face_cranio/cli.py --request CAMINHO_DO_PEDIDO.json
```

Pedidos tem `action` e `args`. Acoes: `prepare`, `train`, `evaluate`, `process`,
`expand`, `promote`, `train_matching`, `evaluate_matching`, `compare_matching`,
`train_cranial_sex` e `apply_cranial_sex`.
Exemplo de treino apos preparar referencias (caminho absoluto no campo reference):

```json
{"action":"train","args":{"reference":"/caminho/references.json","modality":"cranio","view":"CF","epochs":100,"lr":0.001,"batch_size":8,"size":128,"seed":42}}
```

Para retomar, acrescentar `resume` com o diretorio do treino e aumentar `epochs`.
Para parar, criar `stop.flag` na pasta do pedido. A parada ocorre no limite de uma
epoca ou imagem; logs e checkpoints ficam salvos. Nao reutilizar simultaneamente a
mesma pasta de pedido. Pedidos devem ser arquivos locais confiaveis.

## Testes

```bash
Treinamento_faces/.venv/bin/python -m unittest discover -s Laboratorio_visual_face_cranio/tests -v
```

## Modelos de matching

As torres independentes reutilizam ResNet18 de `Modelo_novo`. O backbone fica
congelado e as projecoes sao treinadas com InfoNCE ou triplet semi-hard.
O checkpoint vem da maior MRR de validacao, com early stopping. Os pesos ImageNet
devem existir em `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`;
o aplicativo nao baixa pesos automaticamente. Features por imagem sao cacheadas
com hashes. A retomada conserva manifesto, pesos, mascaras de revisao e parametros.

O baseline Ridge aprende face -> cranio com 55 distancias normalizadas e mascaras.
Comparacao geometrica exige pelo menos cinco medidas comuns e 60% da cobertura
da consulta. O hibrido combina geometria com imagem. Multivista usa slots fixos e
mascaras; recortes regionais usam as propostas frontais. Laterais sem referencias
podem fornecer imagem, mas nao recebem pontos frontais transplantados.

Ranking e score de similaridade nao sao probabilidade de identidade. O teste
congelado precisa de uma avaliacao explicita. Gallery-10 e avaliacao, nao treino.
Sobreposicao automatica exige correspondencias revisadas nao colineares;
os residuos sao em pixels e nao estimam espessura de tecido mole.

## Sexo registrado e estimativa craniana

FairFace e modelo facial permanecem como resultados historicos importados.
A tabela preserva curadoria manual e preenchimento automatico. Para treinar o
baseline craniano de sexo biologico, importar CSV com `id_pessoa`, `sexo`, `fonte`,
`tipo_referencia` (DOCUMENTAL ou ESPECIALISTA), autor e confirmacao de procedencia.
Sem essa referencia, o botao permanece desabilitado. Nao converter automaticamente
rotulos FairFace em sexo documentado.

O baseline usa silhuetas CF/CE/CD e regressao logistica, com normalizador ajustado
somente no treino. Limiar e agregacao media/mediana sao escolhidos na validacao.
Avaliacao no teste e explicita. Silhuetas podem incluir suporte; inspecao continua
necessaria. Novas previsoes aparecem na tabela conjunta e por vista.

O filtro no matching usa sexo informado/documentado; valores indefinidos nao
excluem candidatos. Posicao anterior ao filtro e exclusao do correto ficam visiveis.
