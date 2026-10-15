# Pesos das vistas na validacao

A prioridade atual e Recall@1 (primeiro colocado), com MRR como desempate.
Novas buscas e novos checkpoints seguem esse criterio, registrado nas
configuracoes. Os numeros abaixo preservam a primeira busca, feita por MRR.
No painel, confira o criterio de cada busca antes de comparar resultados.

Busca executada em 14/10/2026 para DINOv2 Base, com seeds 42, 43 e 44,
separadamente para InfoNCE e triplet. Foram avaliadas 232 combinacoes por
loss: pesos em passos de 5%, incluindo vistas com peso zero e pesos iguais.
O criterio de escolha foi a MRR media na galeria completa de 103 pessoas.
As normalizacoes existentes, ajustadas nos pares diferentes do treino,
foram reutilizadas. Nenhum novo treinamento ou teste final foi executado.

| Loss | Neutra | Sorriso | Lateral | MRR igual | MRR escolhida | Ganho |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Triplet | 40% | 30% | 30% | 0,2363 | 0,2378 | +0,0015 |
| InfoNCE | 10% | 50% | 40% | 0,2260 | 0,2364 | +0,0104 |

Esses ganhos sao descritivos: os pesos foram escolhidos na propria
validacao. A busca tambem selecionou pesos em duas seeds e avaliou a
terceira, repetindo para as tres seeds. Nesse procedimento, a MRR media
foi 0,2181 no triplet e 0,2274 no InfoNCE. As identidades de validacao
continuam sendo as mesmas; isso avalia variacao entre treinamentos.

O ganho do triplet e pequeno e nao se manteve entre seeds. Pesos iguais
continuam como referencia recomendada para ele. Os pesos InfoNCE podem
ser explorados, com ganho menor ao excluir uma seed da escolha.
Melhorar MRR nao implica melhorar todos os Recalls: compare as colunas
antes de adotar uma combinacao.

## Painel

1. Treinamentos > Buscar pesos das vistas na validacao: selecione o
   extrator/loss e clique em Buscar melhores pesos. Scores sao reutilizados.
2. Resultados > Pesos das vistas: consulte os pesos, a comparacao com
   pesos iguais, os resultados por seed e a estabilidade.
3. Matching > Execucao dos modelos: selecione uma execucao marcada
   pesos ajustados. Os pesos sao carregados automaticamente; podem ser
   editados e restaurados com Usar pesos desta execucao.

Cada busca salva configuracao, hashes, grid CSV/JSON, resultados por seed,
checagem por seed excluida da escolha e relatorio HTML em
artifacts/weight_searches. As combinacoes escolhidas aparecem como novas
execucoes em artifacts/fusions, com rankings e exemplos visuais.
Os checkpoints e scores de origem permanecem preservados.
