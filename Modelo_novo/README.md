# Modelo novo — matching face–crânio

Pipeline experimental e portável para preparar dados, treinar e avaliar retrieval entre faces frontais e crânios. O módulo não altera nem sobrescreve `Treinamento_faces`, `Treinamento_cranios` ou `Matching_cranio_face`.

O experimento multivista opcional usa as três fotografias reais disponíveis por participante: frontal neutra (`FFN`), frontal sorrindo (`FFS`) e lateral (`FL`).

## Estado inicial auditado

- 810 chaves na união das fontes.
- 723 pares válidos e com arquivos existentes.
- Splits congelados: 507 treino, 108 validação e 108 teste.
- Chave única: `ano::id_pessoa`.
- Caminhos persistidos relativamente à raiz do repositório; caminhos absolutos legados de outras máquinas são recuperados automaticamente.
- Folds leave-one-year-out para 2017, 2019, 2021, 2022 e 2023.

O termo correto para a saída é **score de matching/similaridade aprendida**, não percentual pericial de compatibilidade.

## Instalação

Na raiz do repositório:

```bash
python3 -m venv .venv-modelo-novo
source .venv-modelo-novo/bin/activate
pip install -r Modelo_novo/requirements.txt
```

## Pipeline reproduzível

```bash
python Modelo_novo/cli.py build-index
python Modelo_novo/cli.py create-splits
python Modelo_novo/cli.py preprocess --limit 10
python Modelo_novo/cli.py preprocess
python Modelo_novo/cli.py baselines
python Modelo_novo/cli.py train
```

## Estudos sem contaminar o teste original

Comparar uma vista por vez (recomendado para acompanhar duração):

```bash
python Modelo_novo/cli.py ablate-views --combination CF
python Modelo_novo/cli.py ablate-views --combination CE_CD
python Modelo_novo/cli.py ablate-views --combination CF_CE
python Modelo_novo/cli.py ablate-views --combination CF_CD
python Modelo_novo/cli.py ablate-views --combination CF_CE_CD
```

Ou executar todas sequencialmente:

```bash
python Modelo_novo/cli.py ablate-views --combination all
```

Esses experimentos selecionam a configuração usando apenas validação. O teste original não é carregado.

Generalização por ano, um treinamento por comando:

```bash
python Modelo_novo/cli.py loyo --year 2017
python Modelo_novo/cli.py loyo --year 2019
python Modelo_novo/cli.py loyo --year 2021
python Modelo_novo/cli.py loyo --year 2022
python Modelo_novo/cli.py loyo --year 2023
```

Em cada fold, o ano escolhido é exclusivamente teste; a validação é retirada dos outros quatro anos.

Simular a aplicação final com uma face e dez crânios, sem novo treinamento:

```bash
python Modelo_novo/cli.py simulate-gallery --size 10 --repeats 100 --variant domain_normalized_hard_negatives
```

São avaliados candidatos aleatórios, do mesmo sexo e difíceis, além de galerias em que o correto está ausente. A variante normalizada é usada por padrão; passe `--variant original` para reproduzir a comparação anterior. Os relatórios são separados por variante para evitar sobrescrita.

### Treinamento com negativos difíceis

Use a configuração dedicada para penalizar, a partir da quarta época, os três crânios incorretos mais semelhantes de cada lote:

```bash
python3 Modelo_novo/cli.py --config Modelo_novo/configs/hard_negatives.yaml loyo --year 2017
```

Repita para 2019, 2021, 2022 e 2023. Essa variante é registrada como `domain_normalized_hard_negatives` e não substitui os resultados anteriores.

### Estudo avançado reproduzível

A configuração `research_advanced.yaml` combina fusão aprendida por atenção das vistas CF/CE/CD e mineração global de negativos difíceis. Para executar o protocolo LOYO com três sementes, reutilizando os splits já congelados:

```bash
.venv-modelo-novo/bin/python Modelo_novo/cli.py \
  --config Modelo_novo/configs/research_advanced.yaml \
  multiseed-loyo --years all --seeds 42 123 456
```

O comando grava cada execução imediatamente e pula combinações de ano/semente já concluídas quando for chamado novamente. Resultados individuais, médias e desvios-padrão ficam em `Modelo_novo/studies/multiseed_loyo/domain_normalized_global_hard_attention/`.

Para separar o efeito das duas mudanças, execute as ablações com os mesmos splits, sementes e 24 épocas:

```bash
.venv-modelo-novo/bin/python Modelo_novo/cli.py --config Modelo_novo/configs/research_attention_only.yaml multiseed-loyo --years 2017 --seeds 42 123 456
.venv-modelo-novo/bin/python Modelo_novo/cli.py --config Modelo_novo/configs/research_global_hard_only.yaml multiseed-loyo --years 2017 --seeds 42 123 456
```

### Modelo híbrido visual–geométrico

Os 64 pontos do contorno facial e os 64 pseudolandmarks do crânio frontal possuem cobertura completa dos 723 pares. A configuração híbrida usa duas torres geométricas independentes (128 coordenadas cada) e funde seus embeddings com as torres visuais, mantendo a média robusta das vistas CF/CE/CD:

```bash
.venv-modelo-novo/bin/python Modelo_novo/cli.py \
  --config Modelo_novo/configs/research_hybrid_shape64.yaml \
  multiseed-loyo --years 2017 --seeds 42 123 456
```

Os pontos são representações de contorno normalizadas; não devem ser descritos como landmarks anatômicos homólogos sem validação antropológica específica.

### Ensemble multissemente

Depois de concluir as três sementes dos cinco anos, combine os escores por média simples e simule dez candidatos:

```bash
.venv-modelo-novo/bin/python Modelo_novo/cli.py ensemble-rankings --source-variant domain_normalized_hard_negatives
.venv-modelo-novo/bin/python Modelo_novo/cli.py simulate-gallery --size 10 --repeats 100 --variant ensemble_domain_normalized_hard_negatives
```

A agregação é fixa (`raw_score_mean`) para evitar selecionar o método diretamente pelo teste.

### Robustez de domínio sem consultar o teste

Compare o controle com faces/crânios em escala de cinza equalizada e augmentations de captura. Ambos usam somente o split congelado de treino e validação:

```bash
.venv-modelo-novo/bin/python Modelo_novo/cli.py --config Modelo_novo/configs/research_validation_control.yaml multiseed-validation --seeds 42 123 456
.venv-modelo-novo/bin/python Modelo_novo/cli.py --config Modelo_novo/configs/research_domain_robust.yaml multiseed-validation --seeds 42 123 456
```

O teste não é carregado por esse comando. Somente se a variante robusta vencer de forma consistente ela deve ser congelada e levada a uma nova avaliação LOYO exploratória.

Para otimizar temperatura, margem e peso dos negativos especificamente para dez candidatos, ainda sem teste:

```bash
.venv-modelo-novo/bin/python Modelo_novo/cli.py --config Modelo_novo/configs/research_validation_control.yaml tune-gallery10 --seeds 42 123 456
```

A busca testa sete combinações na primeira semente e confirma somente o controle e as duas melhores nas outras sementes. As galerias amostradas e a regra de checkpoint usam semente fixa e média recíproca de posição (`gallery10_mrr`).

A configuração selecionada pela validação foi congelada em `research_gallery10_tuned.yaml` (temperatura 0,07; margem 0,20; peso 0,25). Sua avaliação LOYO final é executada sem novos ajustes:

```bash
.venv-modelo-novo/bin/python Modelo_novo/cli.py --config Modelo_novo/configs/research_gallery10_tuned.yaml multiseed-loyo --years all --seeds 42 123 456
```

Auditar diferenças de captura entre anos e gerar galerias visuais:

```bash
python Modelo_novo/cli.py audit-domains
```

Gerar um novo conjunto normalizado sem alterar originais:

```bash
python Modelo_novo/cli.py normalize-images --size 224
```

Para preparar e treinar o experimento com as três fotografias faciais, mantendo os splits e o restante do protocolo avançado:

```bash
.venv-modelo-novo/bin/python Modelo_novo/cli.py --config Modelo_novo/configs/research_multiview_faces.yaml normalize-images --size 224
.venv-modelo-novo/bin/python Modelo_novo/cli.py --config Modelo_novo/configs/research_multiview_faces.yaml multiseed-validation --seeds 42 123 456
```

A seleção inicial deve permanecer restrita a treino/validação. Somente se a fusão multivista superar consistentemente o controle, execute o protocolo LOYO congelado. A `FFN` reutiliza os landmarks frontais; `FFS` e `FL` recebem normalização conservadora sem landmarks, pois aplicar pontos frontais à vista lateral seria geometricamente inválido.

As faces são recortadas por landmarks, alinhadas pelos olhos, mascaradas e padronizadas. Os crânios são centralizados pelo conteúdo, redimensionados e recebem contraste uniforme. As saídas ficam em `Modelo_novo/data/domain_normalized/`.

Para usar esse conjunto em um experimento, passe a configuração dedicada antes do comando:

```bash
python Modelo_novo/cli.py --config Modelo_novo/configs/domain_normalized.yaml loyo --year 2019
```

Interface:

```bash
streamlit run Modelo_novo/app.py
```

Na área **Comparar 10 crânios**, envie uma face frontal e as vistas CF, CE e CD de cada um dos dez candidatos. A interface combina os cinco checkpoints `domain_normalized_hard_negatives` e apresenta um ranking de similaridade, sem converter o score em probabilidade pericial.

## Proteções metodológicas

- Split por pessoa, estratificado aproximadamente por ano e sexo.
- Teste temporal leave-one-year-out.
- Torres ResNet18 independentes para as duas modalidades.
- Crânio multivista: embeddings CF, CE e CD combinados no espaço latente.
- Backbone congelado nas três primeiras épocas; somente o último bloco é liberado depois.
- Learning rates separados para projeções e backbones.
- Augmentations conservadoras, sem espelhamento que destrua a lateralidade.
- Dropout, label smoothing e clipping de gradiente para reduzir overfitting.
- InfoNCE simétrica por padrão; cada batch fornece negativos cruzados.
- Seleção de checkpoint por MRR na validação.
- Teste consultado somente após a seleção do checkpoint.
- Métricas finais estratificadas por ano e sexo.
- Sexo fora da loss principal e peso zero por padrão.
- Função separada para penalização probabilística por sexo, permitindo ablação.
- Baselines aleatório, apenas sexo e shape 64 atual.

As probabilidades cranianas de sexo existentes podem ter sido treinadas com indivíduos que aparecem nestes splits. Por isso são usadas apenas no baseline/ablação, e não como entrada do two-tower. Uma avaliação conclusiva com sexo exige probabilidades out-of-fold alinhadas aos splits deste módulo.

## Artefatos

- `data/manifests/metadata_pares.csv`: união auditável, incluindo excluídos e motivos.
- `data/manifests/metadata_pares.audit.json`: hash, contagens e erros.
- `data/splits/*.csv`: splits congelados.
- `data/splits/leave_one_year_out/`: testes de generalização por ano.
- `reports/baselines/`: métricas e rankings dos baselines.
- `runs/<run_id>/`: histórico, checkpoint, métricas e ranking neural.

## Critério de sucesso

O two-tower sem sexo deve superar consistentemente o baseline `shape64_atual`, não apenas o acaso ou `apenas_sexo`, em pessoas nunca vistas. O desempenho também deve ser examinado por ano e no protocolo leave-one-year-out.

## Limitações

Os 723 pares são adequados para prova de conceito, mas pequenos para fine-tuning integral de duas CNNs. O sistema não deve ser apresentado como identificação pericial nem como probabilidade calibrada sem validação externa independente.
