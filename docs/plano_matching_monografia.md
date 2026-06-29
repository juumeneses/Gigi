# Plano V2 - Matching cranio-face baseado na monografia

## Fontes

- Monografia: "Automatizacao do reconhecimento facial: uma abordagem computacional baseada em tomografias do cranio".
- Aula 12: reconhecimento de faces, redes siamesas, triplet loss e classificacao binaria.
- Aula 13: one-shot learning com rede siamesa e triplet loss.
- Artigo "Sex estimation using skull silhouette images..." fica restrito ao modulo de sexo craniano, nao ao matching.

## O que a monografia faz

A monografia faz matching cranio-face. Ela nao reconstrói a face a partir do cranio. O objetivo e estimar a probabilidade de correspondencia entre uma tomografia craniana e uma fotografia facial.

Fluxo central:

1. Face:
   - usa fotografia facial frontal;
   - extrai 468 landmarks com MediaPipe Face Mesh;
   - normaliza os pontos por centro dos olhos, rotacao da linha ocular, distancia interpupilar e altura facial robusta;
   - vetoriza os 468 pontos em 936 coordenadas x/y;
   - reduz dimensionalidade com PCA;
   - na monografia, o PCA final preserva 99,95% da variancia e chega a 23 componentes.

2. Cranio:
   - usa imagem/tomografia frontal do cranio;
   - pre-processa em escala de cinza e equalizacao de histograma;
   - nao marca landmarks cranianos manuais;
   - usa uma CNN pre-treinada, especialmente VGG16, como extratora de features da imagem craniana.

3. Rede two-tower:
   - torre craniana: imagem do cranio -> VGG16 congelada -> Global Average Pooling -> Dense com L2 -> embedding normalizado L2;
   - torre facial: PCA dos landmarks faciais -> Dense -> embedding normalizado L2;
   - as duas torres projetam cranio e face para o mesmo espaco latente.

4. Comparacao:
   - similaridade por cosseno entre embeddings normalizados;
   - score de correspondencia usado como probabilidade;
   - limiar inicial 0,5 para par correspondente ou nao correspondente.

5. Treinamento:
   - pares positivos: cranio e face do mesmo individuo;
   - pares negativos: mesmo cranio com face de outro individuo;
   - negative sampling 1:1;
   - perda principal: entropia cruzada binaria, nao triplet loss;
   - triplet loss aparece como referencia de one-shot learning, mas a monografia escolhe classificacao binaria por estabilidade em dominios heterogeneos.

6. Genero:
   - genero entra como informacao auxiliar;
   - concatena um escalar de genero ao embedding/vetor facial;
   - negative sampling pode ser restrito a individuos do mesmo genero, para evitar que o modelo aprenda apenas diferencas sexuais obvias;
   - no nosso projeto, o genero da face vem da curadoria/fairface e o sexo craniano vem do modulo do artigo.

## Proxima etapa recomendada

### V2.1 - Dataset pareado para matching

Criar exportacao unica:

- `Matching_cranio_face/dados/dataset_pares_matching.csv`
- colunas minimas:
  - `ano`
  - `id_pessoa`
  - `face_id`
  - `face_path`
  - `cranio_cf_path`
  - `genero_final_face`
  - `sexo_cranio_predito`
  - `pca_face_*`
  - `usar_matching_face`
  - `usar_matching_cranio`

Regras:

- so entra se face e cranio CF existem;
- preferir itens com `usar_matching=True`;
- manter genero como feature auxiliar, nunca filtro absoluto na inferencia.

### V2.2 - Treino two-tower

Criar script:

- `Matching_cranio_face/scripts/02_treinar_two_tower_matching.py`

Entradas:

- PCA facial final de `Treinamento_faces`;
- imagem CF do cranio ou silhueta/normalizada;
- genero curado/estimado;
- pares positivos e negativos.

Arquitetura:

- torre craniana VGG16 congelada;
- torre facial MLP simples sobre PCA;
- normalizacao L2 nas duas saidas;
- similaridade cosseno;
- cabeca sigmoid para `P(match)`.

Defaults iniciais:

- `image_size=224`
- `batch_size=16` no Mac M4, reduzir para 8 se faltar memoria;
- `embedding_dim=512`
- `dropout=0.0` para ranking;
- `dropout=0.2` para triagem conservadora;
- `epochs=60`
- `early_stopping_patience=10`
- `learning_rate=1e-4` na cabeca e projeções.

### V2.3 - Avaliacao

Salvar:

- `metricas_two_tower_matching.csv`
- `matriz_confusao_two_tower_matching.csv`
- `ranking_teste_two_tower.csv`
- `historico_runs_matching.csv`

Metricas:

- accuracy;
- F1;
- recall;
- false negative rate;
- false positive rate;
- top-k accuracy no ranking;
- matriz por genero;
- distribuicao de scores para erros.

### V2.4 - Painel

Na pasta `Matching_cranio_face`:

- aba `Treino`;
- aba `Runs`;
- aba `Ranking`;
- aba `Erros`;
- aba `Inferencia`.

Na inferencia:

- upload de 1 face;
- upload de N cranios;
- extrair landmarks/PCA da face;
- extrair embedding do cranio;
- usar genero como peso auxiliar;
- gerar ranking por `P(match)` e score composto;
- mostrar previews e exportar CSV.

## O que nao fazer nesta etapa

- Nao misturar o artigo de sexo craniano com o matching; ele serve para treinar o classificador de sexo do cranio.
- Nao exigir landmarks anatomicos manuais no cranio; a monografia explicitamente usa CNN para evitar essa dependencia.
- Nao usar triplet loss como primeira opcao; a monografia escolhe binary cross-entropy por estabilidade.
- Nao usar genero como filtro absoluto, porque pode haver erro de classificacao ou curadoria incompleta.
