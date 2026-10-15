# Pipeline Top-10

## Executar pelo painel

1. Atualizar a pagina e abrir Treinamentos.
2. Abrir "Pipeline Top-10: DINOv2 x ConvNeXt e parametros".
3. Manter os dois extratores e a busca de parametros selecionados.
4. Clicar "Iniciar pipeline Top-10".

O primeiro uso prepara os pesos oficiais do ConvNeXt Tiny, se ausentes. Requer
rede e espaco em disco. Depois, os pesos e as caracteristicas ficam locais.
Cada clique inicia outra execucao. Interromper/Retomar preservam etapas validas.

## Sequencia padrao

- DINOv2 Base e ConvNeXt Tiny, ambos congelados, com projecoes treinaveis.
- Triplet, imagens 224 x 224, proporcao preservada, sem CE.
- Neutra x CF, sorrindo x CF e lateral facial x CD.
- Pesos fixos 50/30/20. Sem ajuste de pesos durante a busca de parametros.
- Baseline: lr 0,001, margem 0,2, batch 32, ate 100 epocas, paciencia 15.
- Seeds 42, 43 e 44 para comparar os extratores por Recall@10 medio.
- No extrator escolhido: lr 0,001 / 0,0003 e margem 0,1 / 0,2 / 0,3.
- Triagem das seis configuracoes na seed 42; confirmacao da escolhida nas
  outras duas seeds. Modelos identicos dentro da execucao sao reutilizados.
- Ensembles dos baselines e da configuracao confirmada; recomendacao por
  Recall@10, com MRR e Recall@1 como desempates.

Sao ate 39 modelos de projecao por vista, nao 39 extratores treinados.
Desmarcar a busca de parametros executa somente a comparacao inicial (18).
O checkpoint de cada vista usa Recall@10 e MRR como desempate. Top-1/3/5
permanecem nos relatorios. Nenhum resultado e garantido antecipadamente.

## Resultados e Matching

Resultados > Pipeline Top-10 apresenta metricas, ganhos e intervalos,
parametros, sete combinacoes por modelo e exemplos visuais.
As linhas de uma seed sao explicitamente provisorias. As medias individuais
e os ensembles sao apresentados separadamente. Cada consulta usa a galeria
completa da validacao; Top-10 em uma galeria de dez nao mede esta meta.

Ao concluir, clicar "Usar recomendado Top-10 no Matching". O painel seleciona
o ensemble recomendado; nao usa automaticamente um modelo de outro protocolo.
A tabela por modelos e combinacoes continua disponivel nas consultas.

O teste final fica reservado. A validacao orienta a selecao e a triagem, entao
os ganhos e intervalos apresentados sao exploratorios, nao confirmacao
independente. Seeds repetem as mesmas pessoas.

## CLI

Usar `cli.py retrieval_pipeline --args parametros.json`. O JSON pode conter:

```json
{
  "manifest": "/caminho/para/manifest.json",
  "backbones": ["dinov2_vitb14", "convnext_tiny"],
  "seeds": [42, 43, 44],
  "learning_rates": [0.001, 0.0003],
  "margins": [0.2, 0.1, 0.3],
  "tune": true
}
```

Retomar com os mesmos argumentos e `--run-dir /pasta/da/execucao --resume`.
`--force` cria outra execucao. Configuracao, hashes, checkpoints, metricas,
rankings, arquivos CSV e relatorio HTML ficam em `artifacts/retrieval_pipelines`;
os modelos combinados ficam em `artifacts/fusions` para uso no Matching.
