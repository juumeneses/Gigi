# Auditoria do pipeline concluido

Execucao: 20261015T014757386011Z. Duracao registrada: 26 minutos e 2 segundos.
Todas as 64 etapas e os 72 modelos terminaram. Os hashes das saidas
concluidas e das 2.975 entradas usadas foram conferidos, sem alteracoes.

Foram usadas 492 identidades de treino e 103 de validacao. As 106 de teste
permaneceram reservadas. Os checkpoints correspondem ao maior Recall@1
registrado, com MRR como desempate. Os treinos terminaram entre 22 e 63
epocas. Os 36 registros de mineracao usam somente identidades de treino,
sem duplicacoes nos batches da ultima epoca auditada.

## Comparacao principal

Todos os valores abaixo sao Recall@1 na galeria completa de 103 pessoas,
com pesos neutra/sorriso/lateral de 50/30/20. Individual representa a media
das seeds 42, 43 e 44; ensemble e o ranking unico que combina essas seeds.

| Cenario | InfoNCE individual | InfoNCE ensemble | Triplet individual | Triplet ensemble |
| --- | ---: | ---: | ---: | ---: |
| Original | 8,7% | 13,6% | 12,0% | 9,7% |
| Candidatos dificeis | 11,0% | 10,7% | 9,7% | 9,7% |
| Enquadramento | 10,0% | 10,7% | 10,0% | 11,7% |
| Enquadramento + dificeis | 10,0% | 10,7% | 8,4% | 8,7% |

O melhor ensemble com as tres vistas foi o Original InfoNCE: 14 acertos
em 103 consultas (13,6%). O ganho de mineracao e enquadramento nao foi
consistente nesse protocolo de tres vistas. O teste de enquadramento
removeu bordas em 245 CF, 242 CD e 19 FL entre 595 pessoas de treino e
validacao; FFN e FFS permaneceram inteiras. Portanto, esse experimento
testou principalmente enquadramento craniano, nao uma normalizacao
anatomica das faces.

## Selecionando vistas

- Maior resultado de uma seed: InfoNCE, enquadramento, seed 42,
  neutra + lateral: 18/103 (17,5%). As seeds 43 e 44 tiveram 9,7% e 12,6%.
  A media foi 13,3%, contra 8,4% do Original com as mesmas vistas.
  Diferenca exploratoria de +4,85 p.p.; IC95% aproximadamente 0 a
  +10,03 p.p. O limite toca zero: ganho inconclusivo.
- InfoNCE com candidatos dificeis, ensemble, sorriso + lateral:
  15/103 (14,6%). Original com as mesmas vistas: 10/103 (9,7%).
  IC95% da diferenca: 0 a +10,68 p.p.; ganho inconclusivo.
- InfoNCE com enquadramento + dificeis, ensemble, sorriso + lateral:
  tambem 15/103; o intervalo inclui zero.

As sete combinacoes foram exploradas na mesma validacao; a melhor
configuracao observada nao constitui estimativa independente de precisao.
O resultado de 17,5% exige neutra + lateral, nao as tres vistas.

## Correcao numerica da analise

Um limite inferior de bootstrap de aproximadamente 7e-18 e numericamente
zero. A classificacao agora arredonda os limites a 12 casas antes de
decidir se o intervalo inclui zero. Foi adicionado um teste para esse caso.
O resumo anterior esta preservado em summary_before_ci_rounding_audit.json.
Os scores, pesos, checkpoints e metricas de ranking nao foram alterados.

## Recomendacao

Manter Original InfoNCE ensemble como referencia de tres vistas.
Investigar neutra + lateral com enquadramento e sorriso + lateral com
candidatos dificeis como candidatos experimentais, antes de novos treinos.
A precisao absoluta permanece baixa para identificar pessoas; estes
resultados sustentam experimentacao e analise de falhas, nao identificacao
confirmada. O teste final deve ser executado somente apos congelar a
configuracao escolhida.
