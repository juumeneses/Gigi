# Laboratorio hibrido cranio-face

Projeto experimental paralelo. Os pipelines existentes em `Treinamento_cranios`,
`Treinamento_faces` e `Matching_cranio_face` nao sao sobrescritos.

## Execucao

Na raiz desta pasta:

```bash
python run_pipeline.py --stage all
```

Etapas isoladas:

```bash
python run_pipeline.py --stage dataset
python run_pipeline.py --stage cranios
python run_pipeline.py --stage faces
python run_pipeline.py --stage matching
python run_pipeline.py --stage relatorios
```

O split por identidade e criado uma vez e preservado, inclusive com `--force`.
As imagens em `dataset/` sao links relativos para o acervo original.

## Execucao auditavel dos cranios

Etapas isoladas nao executam as outras etapas. Execute `dataset` primeiro
se os manifestos ou splits ainda nao existirem. A etapa `cranios` salva
checkpoints individuais em `resultados/cranios/runs/<run_id>/`.
Reexecutar sem `--force` retoma a mesma run quando entradas, configuracao,
split e codigo sao iguais; saidas ausentes ou corrompidas sao recalculadas.
`--resume` explicita esse comportamento padrao. `--force` cria outra run.
Falhas de imagem e ponteiros LFS permanecem nas metricas e denominadores.

```bash
python run_pipeline.py --stage cranios --resume
streamlit run revisao_cranios.py
```

A tela de revisao inicial usa somente treino/validacao e vincula julgamentos
as coordenadas, hash da imagem e run. O detector ainda e V3: esta entrega
nao certifica os pontos nem libera o hibrido. A evolucao regional do detector,
a integracao desta revisao ao app principal e a validacao anatomica continuam
pendentes. Os resultados legados permanecem disponiveis.

## Resultados

Abra `resultados/relatorios_html/index.html` no navegador. CSVs auditaveis ficam
em `dataset/manifestos`, `resultados/cranios`, `resultados/faces` e
`resultados/matching`.

Os pontos cranianos sao referencias geometrico-anatomicas automaticas. Eles nao
devem ser descritos como landmarks periciais validados.

## Two-tower

O treino neural e deliberadamente separado da execucao exploratoria:

```bash
python scripts/treinar_two_tower.py --weights imagenet
```

O script usa apenas treino/validacao para escolher o checkpoint. O teste fica
reservado para a avaliacao congelada.

Baseline PyTorch disponivel sem downloads adicionais, usando o ResNet18
ImageNet ja presente no cache local e os 61 componentes PCA faciais:

```bash
python scripts/treinar_two_tower_torch.py
python run_pipeline.py --stage matching
```
