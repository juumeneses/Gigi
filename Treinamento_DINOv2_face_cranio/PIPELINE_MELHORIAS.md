# Pipeline de melhorias por Recall@1

## Executar pelo painel

Em **Treinamentos > Pipeline de melhorias: executar e comparar**:

1. Escolha o extrator e as losses. DINOv2 Base ja esta preparado.
2. Selecione os cenarios. Original entra sempre como controle.
3. Confira os pesos fixos, inicialmente 50% neutra, 30% sorriso e 20% lateral.
4. Use **Rodar somente ensemble existente** para combinar as tres seeds ja
   treinadas, ou **Iniciar pipeline completo** para novos treinamentos.
5. Acompanhe e interrompa em **Acompanhamento e logs**. **Retomar** continua
   na mesma pasta e verifica entradas, checkpoints, configuracoes e scores.

O plano completo tem 4 cenarios x 2 losses x 3 seeds x 3 vistas = 72 modelos
de projecao. DINOv2 continua congelado. Embeddings compativeis sao reutilizados.
Os parametros podem ser ajustados antes de iniciar. Nao ha busca ilimitada
de hiperparametros ou repeticoes automaticas ate conseguir um resultado.

## O que cada cenario testa

| Cenario | Imagem | Batches de treino |
| --- | --- | --- |
| Original | Original, EXIF/RGB/letterbox | Aleatorios |
| Candidatos dificeis | Original | Vizinhos buscados em todo o treino |
| Enquadramento | Recorte de bordas uniformes com margem | Aleatorios |
| Enquadramento + candidatos dificeis | Mesmo recorte | Vizinhos do treino |

Todos selecionam o checkpoint pelo Recall@1 de validacao, com MRR como
desempate. A mineracao usa somente pares de treino; as primeiras cinco
epocas, por padrao, usam batches aleatorios. Cada identidade entra uma vez
por epoca. A loss continua sendo InfoNCE ou triplet semi-hard.

O recorte e conservador: so remove bordas com fundo aproximadamente
uniforme e mantem margem de 10%, com minimo de quatro pixels. O conjunto
de pixels contrastantes e preservado; fundos heterogeneos, ausencia de
primeiro plano e caixas sem reducao relevante mantem a imagem original.
Nao identifica estruturas anatomicas. Suporte, cabelo ou texto podem
permanecer no recorte. Confira as previas antes de interpretar o resultado.

O cache do recorte inclui o modo e o hash de seu codigo. A inferencia aplica
o mesmo preparo do treinamento. Imagens originais e resultados anteriores
nao sao alterados.

## Comparacoes

**Resultados > Pipeline de melhorias** apresenta:

- Recall@1/3/5/10, MRR, falhas e comparacao com o Original da mesma loss.
- Media e desvio entre seeds para os modelos individuais.
- Ensemble: scores de cada vista padronizados com pares diferentes do
  treino, depois media das seeds e soma ponderada das vistas.
- Sete combinacoes de vistas, sem novos treinamentos.
- Diferencas de Recall@1 e intervalos por bootstrap de identidade; seeds
  sao promediadas antes do bootstrap, pois repetem as mesmas pessoas.
- Exemplos de acertos e erros, candidatos corretos e previas do recorte.

O historico antigo, quando disponivel, e contexto separado: os checkpoints
podem ter sido selecionados por MRR. Os novos quatro cenarios formam a
comparacao controlada. As mesmas pessoas e galerias sao mantidas, e falhas
continuam nos denominadores. O teste final nao e executado pelo pipeline.

Todos os pesos fixos e cenarios sao definidos antes da execucao. Ganhos
medidos nessa validacao sao exploratorios; o pipeline nao troca a
configuracao repetidamente para perseguir resultados nem promove um
modelo a resultado confirmado de identificacao.

## Matching

Os resultados individuais e ensembles aparecem em **Execucao dos modelos**.
O seletor indica o ensemble e a prioridade e Recall@1. Os pesos da execucao
sao carregados na selecao; ajustes manuais continuam disponiveis.

Uploads e teste conhecido usam a mesma inferencia. Um ensemble exige as
mesmas vistas nos modelos selecionados e preserva as falhas de seus
membros. Seus scores sao medias padronizadas, nao probabilidades.
O mapa de oclusao exibido detalha um membro identificado na interface;
nao e apresentado como mapa agregado de todas as seeds.

## CLI

Crie parametros em JSON com `manifest`, `losses`, `seeds`, `conditions`,
`weights` e os parametros de treinamento. Execute:

```sh
Treinamento_faces/.venv/bin/python Treinamento_DINOv2_face_cranio/cli.py improvement_pipeline --args parametros.json
```

Para retomar, use os mesmos parametros e `--run-dir CAMINHO --resume`.
Para outra execucao, omita a pasta ou use `--force`. As saidas ficam em
`artifacts/improvement_pipelines`, e os modelos combinados ficam em
`artifacts/fusions`. Sao salvos request, hashes, ledger de etapas,
resultados JSON/CSV, combinacoes de vistas, relatorio HTML e previas.

Os testes de funcionamento usam extratores sinteticos em pastas
temporarias. Nao representam resultado cientifico. A primeira execucao
real desta entrega reutilizou apenas modelos ja existentes, sem iniciar
novos treinamentos completos.
