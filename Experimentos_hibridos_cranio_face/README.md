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
